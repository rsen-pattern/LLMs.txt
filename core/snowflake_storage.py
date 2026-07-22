"""Persist per-domain llms.txt configs in Snowflake (VIBE_CODE_DB)."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

CONFIG_TABLE = "LLMSTXT_CONFIGS"


def ensure_config_table(conn) -> None:
    """Create the configs table if it does not exist."""
    cursor = conn.cursor()
    try:
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {CONFIG_TABLE} (
                DOMAIN VARCHAR(255) NOT NULL PRIMARY KEY,
                CONFIG VARIANT NOT NULL,
                UPDATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
            )
        """)
    finally:
        cursor.close()


def save_crawl_config(conn, domain: str, config: Dict[str, Any]) -> None:
    """Upsert crawl config for a domain."""
    ensure_config_table(conn)
    config_json = json.dumps(config)
    cursor = conn.cursor()
    try:
        cursor.execute(
            f"""
            MERGE INTO {CONFIG_TABLE} AS target
            USING (SELECT %s AS domain, PARSE_JSON(%s) AS config) AS source
            ON target.DOMAIN = source.domain
            WHEN MATCHED THEN
                UPDATE SET CONFIG = source.config, UPDATED_AT = CURRENT_TIMESTAMP()
            WHEN NOT MATCHED THEN
                INSERT (DOMAIN, CONFIG, UPDATED_AT)
                VALUES (source.domain, source.config, CURRENT_TIMESTAMP())
            """,
            (domain, config_json),
        )
    finally:
        cursor.close()


def load_crawl_config(conn, domain: str) -> Optional[Dict[str, Any]]:
    """Load previously saved crawl config for a domain."""
    cursor = conn.cursor()
    try:
        cursor.execute(
            f"SELECT CONFIG FROM {CONFIG_TABLE} WHERE DOMAIN = %s LIMIT 1",
            (domain,),
        )
        row = cursor.fetchone()
        if not row:
            return None
        raw = row[0]
        if isinstance(raw, str):
            return json.loads(raw)
        if isinstance(raw, dict):
            return raw
        return json.loads(str(raw))
    except Exception as e:
        logger.warning("Failed to load config for %s: %s", domain, e)
        return None
    finally:
        cursor.close()


def list_saved_domains(conn) -> List[str]:
    """List domains with saved configs, most recent first."""
    cursor = conn.cursor()
    try:
        ensure_config_table(conn)
        cursor.execute(
            f"""
            SELECT DOMAIN FROM {CONFIG_TABLE}
            ORDER BY UPDATED_AT DESC
            LIMIT 50
            """
        )
        return [row[0] for row in cursor.fetchall()]
    except Exception as e:
        logger.warning("Failed to list saved domains: %s", e)
        return []
    finally:
        cursor.close()
