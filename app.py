"""LLMs.txt Generator — Flask app for Pattern Forklift deployment."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlparse

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from dotenv import load_dotenv
from flask import (
    Flask,
    Response,
    flash,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

try:
    from snowflake.connector import connect
    HAS_SNOWFLAKE = True
except ImportError:
    HAS_SNOWFLAKE = False
    connect = None  # type: ignore

from core.llms_generator import (
    PATTERN_CATALOG,
    PATTERN_ECOMMERCE,
    PATTERN_INDEX_EXPORT,
    PATTERN_LABELS,
    PATTERN_WORKFLOW,
    LLMsTextGenerator,
    _estimate_tokens,
    _rebuild_llmstxt,
    deduplicate_entries,
    filter_near_duplicates,
    filter_thin_content,
    parse_screaming_frog_csv,
)
from core.snowflake_storage import list_saved_domains, save_crawl_config

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-only-change-me")

_results_store: Dict[str, Dict[str, Any]] = {}

PATTERN_CHOICES = [
    (PATTERN_CATALOG, PATTERN_LABELS[PATTERN_CATALOG]),
    (PATTERN_WORKFLOW, PATTERN_LABELS[PATTERN_WORKFLOW]),
    (PATTERN_INDEX_EXPORT, PATTERN_LABELS[PATTERN_INDEX_EXPORT]),
    (PATTERN_ECOMMERCE, PATTERN_LABELS[PATTERN_ECOMMERCE]),
]


@app.before_request
def reload_env():
    if app.debug:
        load_dotenv(override=True)


def get_private_key():
    private_key_path = os.getenv("SNOWFLAKE_PRIVATE_KEY_PATH")
    private_key_content = os.getenv("SNOWFLAKE_PRIVATE_KEY")
    passphrase = os.getenv("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE")

    if passphrase:
        passphrase = passphrase.strip().strip('"').strip("'")

    if private_key_path:
        private_key_path = private_key_path.strip().strip('"').strip("'")
        if "\r" in private_key_path and "\r" != os.linesep:
            private_key_path = private_key_path.replace("\r", "\\r")
        private_key_path = os.path.normpath(private_key_path)
        if not os.path.isfile(private_key_path):
            raise Exception(f"Private key file not found at path: {private_key_path}")
        with open(private_key_path, "rb") as key_file:
            p_key = serialization.load_pem_private_key(
                key_file.read(),
                password=passphrase.encode() if passphrase else None,
                backend=default_backend(),
            )
    elif private_key_content:
        key_content = private_key_content.strip().strip('"').strip("'").replace("\\n", "\n")
        p_key = serialization.load_pem_private_key(
            key_content.encode(),
            password=passphrase.encode() if passphrase else None,
            backend=default_backend(),
        )
    else:
        raise Exception("Either SNOWFLAKE_PRIVATE_KEY_PATH or SNOWFLAKE_PRIVATE_KEY must be set")

    return p_key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


def get_snowflake_connection():
    if not HAS_SNOWFLAKE:
        raise Exception(
            "Snowflake connector is not installed in this environment. "
            "It is included in the Docker image for Forklift deployment."
        )
    try:
        conn_params = {
            "user": os.getenv("SNOWFLAKE_USERNAME"),
            "private_key": get_private_key(),
            "account": os.getenv("SNOWFLAKE_ACCOUNT"),
            "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
            "database": os.getenv("SNOWFLAKE_DATABASE"),
            "schema": os.getenv("SNOWFLAKE_SCHEMA", "PUBLIC"),
        }
        role = os.getenv("SNOWFLAKE_ROLE")
        if role:
            conn_params["role"] = role
        return connect(**conn_params)
    except Exception as e:
        raise Exception(f"Failed to connect to Snowflake: {str(e)}")


def _api_keys() -> tuple[str, str]:
    return os.getenv("BIFROST_API_KEY", ""), os.getenv("FIRECRAWL_API_KEY", "")


def _store_result(result: Dict[str, Any], site_url: str, single_file_mode: bool) -> str:
    result_id = str(uuid.uuid4())
    result["_url"] = site_url
    result["_single_file_mode"] = single_file_mode
    result["excluded_urls"] = []
    _results_store[result_id] = result
    session["result_id"] = result_id
    return result_id


def _current_result() -> Optional[Dict[str, Any]]:
    result_id = session.get("result_id")
    if result_id and result_id in _results_store:
        return _results_store[result_id]
    return None


def _parse_settings() -> Dict[str, Any]:
    return {
        "pattern": request.form.get("pattern", PATTERN_CATALOG),
        "max_urls": int(request.form.get("max_urls", 20)),
        "generate_full": request.form.get("generate_full") == "on",
        "single_file_mode": request.form.get("single_file_mode") == "on",
        "dedup_enabled": request.form.get("dedup_enabled") == "on",
        "near_dupes_enabled": request.form.get("near_dupes_enabled") == "on",
        "near_dupe_threshold": float(request.form.get("near_dupe_threshold", 90)),
        "thin_content_enabled": request.form.get("thin_content_enabled") == "on",
        "min_word_count": int(request.form.get("min_word_count", 50)),
    }


def _result_stats(result: Dict[str, Any]) -> Dict[str, Any]:
    llmstxt_text = result.get("llmstxt", "")
    fulltxt_text = result.get("llms_fulltxt", "") or ""
    size_kb = len(llmstxt_text.encode("utf-8")) / 1024
    full_size_kb = len(fulltxt_text.encode("utf-8")) / 1024 if fulltxt_text else 0
    total_words = sum(r.get("word_count", 0) for r in result.get("results", []))
    return {
        "size_kb": round(size_kb, 1),
        "full_size_kb": round(full_size_kb, 1),
        "toc_tokens": _estimate_tokens(llmstxt_text),
        "full_tokens": _estimate_tokens(fulltxt_text) if fulltxt_text else 0,
        "total_words": total_words,
    }


def _snowflake_ready() -> bool:
    return HAS_SNOWFLAKE and bool(
        os.getenv("SNOWFLAKE_USERNAME")
        and (
            os.getenv("SNOWFLAKE_PRIVATE_KEY_PATH")
            or os.getenv("SNOWFLAKE_PRIVATE_KEY")
        )
    )


def _list_saved_domains_safe() -> List[str]:
    if not _snowflake_ready():
        return []
    conn = None
    try:
        conn = get_snowflake_connection()
        return list_saved_domains(conn)
    except Exception:
        return []
    finally:
        if conn:
            conn.close()


@app.route("/")
def index():
    result = _current_result()
    saved_domains = _list_saved_domains_safe()
    bifrost_key, firecrawl_key = _api_keys()
    return render_template(
        "index.html",
        result=result,
        stats=_result_stats(result) if result else None,
        pattern_choices=PATTERN_CHOICES,
        saved_domains=saved_domains,
        has_bifrost=bool(bifrost_key),
        has_firecrawl=bool(firecrawl_key),
        has_snowflake=_snowflake_ready(),
    )


@app.route("/generate/firecrawl", methods=["POST"])
def generate_firecrawl():
    site_url = request.form.get("site_url", "").strip()
    settings = _parse_settings()
    bifrost_key, firecrawl_key = _api_keys()

    if not site_url:
        flash("Please enter a website URL.", "error")
        return redirect(url_for("index"))
    if not firecrawl_key:
        flash("FIRECRAWL_API_KEY is required for Firecrawl mode.", "error")
        return redirect(url_for("index"))
    if not bifrost_key:
        flash("BIFROST_API_KEY is required for AI summaries.", "error")
        return redirect(url_for("index"))

    generator = LLMsTextGenerator(
        firecrawl_api_key=firecrawl_key,
        bifrost_api_key=bifrost_key,
    )
    try:
        result = generator.generate_from_firecrawl(
            site_url,
            max_urls=settings["max_urls"],
            generate_full=settings["generate_full"],
            pattern=settings["pattern"],
        )
        _store_result(result, site_url, settings["single_file_mode"])
        flash(
            f"Processed {result['num_urls_processed']} of {result['num_urls_total']} URLs.",
            "success",
        )
    except Exception as e:
        flash(f"Generation failed: {e}", "error")
    return redirect(url_for("index"))


@app.route("/generate/csv", methods=["POST"])
def generate_csv():
    site_url = request.form.get("site_url", "").strip()
    settings = _parse_settings()
    use_ai = request.form.get("use_ai") == "on"
    scrape_content = request.form.get("scrape_content") == "on"
    bifrost_key, firecrawl_key = _api_keys()
    uploaded = request.files.get("csv_file")

    if not site_url:
        flash("Please enter the website URL.", "error")
        return redirect(url_for("index"))
    if not uploaded or not uploaded.filename:
        flash("Please upload a Screaming Frog CSV file.", "error")
        return redirect(url_for("index"))
    if use_ai and not bifrost_key:
        flash("BIFROST_API_KEY is required when AI summaries are enabled.", "error")
        return redirect(url_for("index"))
    if scrape_content and not firecrawl_key:
        flash("FIRECRAWL_API_KEY is required for scraping.", "error")
        return redirect(url_for("index"))

    generator = LLMsTextGenerator(
        firecrawl_api_key=firecrawl_key if scrape_content else None,
        bifrost_api_key=bifrost_key if use_ai else None,
    )

    try:
        file_contents = uploaded.read().decode("utf-8-sig")
        csv_entries = parse_screaming_frog_csv(file_contents, settings["max_urls"])
    except Exception as e:
        flash(f"Failed to parse CSV: {e}", "error")
        return redirect(url_for("index"))

    if not csv_entries:
        flash("No valid URLs found in the CSV.", "error")
        return redirect(url_for("index"))

    filter_messages: List[str] = []
    if settings["dedup_enabled"]:
        csv_entries, dupes = deduplicate_entries(csv_entries)
        if dupes:
            filter_messages.append(f"Removed {len(dupes)} duplicate pages")

    if settings["near_dupes_enabled"]:
        csv_entries, near_dupes = filter_near_duplicates(
            csv_entries, settings["near_dupe_threshold"]
        )
        if near_dupes:
            filter_messages.append(f"Removed {len(near_dupes)} near-duplicate pages")

    if settings["thin_content_enabled"]:
        csv_entries, thin = filter_thin_content(csv_entries, settings["min_word_count"])
        if thin:
            filter_messages.append(
                f"Removed {len(thin)} thin-content pages (<{settings['min_word_count']} words)"
            )

    if filter_messages:
        flash("Filters: " + "; ".join(filter_messages), "info")

    if not csv_entries:
        flash("No pages remaining after filtering.", "error")
        return redirect(url_for("index"))

    try:
        result = generator.generate_from_csv(
            csv_entries=csv_entries,
            site_url=site_url,
            scrape=scrape_content,
            generate_full=settings["generate_full"],
            use_ai=use_ai,
            pattern=settings["pattern"],
        )
        _store_result(result, site_url, settings["single_file_mode"])
        flash(
            f"Processed {result['num_urls_processed']} of {result['num_urls_total']} URLs.",
            "success",
        )
    except Exception as e:
        flash(f"Generation failed: {e}", "error")
    return redirect(url_for("index"))


@app.route("/regenerate", methods=["POST"])
def regenerate():
    result = _current_result()
    if not result:
        flash("No generation result to update.", "error")
        return redirect(url_for("index"))

    site_url = result.get("_url", "")
    edited_name = request.form.get("site_name", result.get("site_name", "")).strip()
    edited_summary = request.form.get("site_summary", result.get("site_summary", "")).strip()
    all_urls = set(request.form.getlist("all_urls"))
    included = set(request.form.getlist("included"))
    excluded_urls = all_urls - included

    new_llmstxt = _rebuild_llmstxt(
        site_url,
        edited_name,
        edited_summary,
        result.get("results", []),
        result.get("pattern", PATTERN_CATALOG),
        excluded_urls,
    )
    result["llmstxt"] = new_llmstxt
    result["site_name"] = edited_name
    result["site_summary"] = edited_summary
    result["excluded_urls"] = list(excluded_urls)
    flash("Regenerated llms.txt with your edits.", "success")
    return redirect(url_for("index"))


@app.route("/save-config", methods=["POST"])
def save_config():
    result = _current_result()
    if not result:
        flash("No generation result to save.", "error")
        return redirect(url_for("index"))
    if not _snowflake_ready():
        flash("Snowflake is not configured. Set SNOWFLAKE_* variables in .env.", "error")
        return redirect(url_for("index"))

    site_url = result.get("_url", "")
    domain = urlparse(site_url).netloc.replace("www.", "")
    config = {
        "site_name": request.form.get("site_name", result.get("site_name", "")),
        "site_summary": request.form.get("site_summary", result.get("site_summary", "")),
        "pattern": result.get("pattern", PATTERN_CATALOG),
        "excluded_urls": result.get("excluded_urls", []),
        "updated_at": datetime.utcnow().isoformat(),
        "num_pages": result.get("num_urls_processed", 0),
    }
    conn = None
    try:
        conn = get_snowflake_connection()
        save_crawl_config(conn, domain, config)
        flash(f"Config saved to Snowflake for {domain}.", "success")
    except Exception as e:
        flash(f"Failed to save config: {e}", "error")
    finally:
        if conn:
            conn.close()
    return redirect(url_for("index"))


@app.route("/clear", methods=["POST"])
def clear_result():
    result_id = session.pop("result_id", None)
    if result_id:
        _results_store.pop(result_id, None)
    flash("Cleared results.", "info")
    return redirect(url_for("index"))


@app.route("/download/<filetype>")
def download(filetype: str):
    result = _current_result()
    if not result:
        flash("No file to download.", "error")
        return redirect(url_for("index"))

    site_url = result.get("_url", "https://example.com")
    domain = urlparse(site_url).netloc.replace("www.", "") or "site"

    if filetype == "llmstxt":
        content = result.get("llmstxt", "")
        filename = f"{domain}-llms.txt"
    elif filetype == "full":
        content = result.get("llms_fulltxt", "")
        filename = f"{domain}-llms-full.txt"
    elif filetype == "combined":
        full = result.get("llms_fulltxt", "") or ""
        content = result.get("llmstxt", "") + "\n\n---\n\n" + full
        filename = f"{domain}-llms.txt"
    else:
        flash("Unknown download type.", "error")
        return redirect(url_for("index"))

    if not content:
        flash("No content available for that file.", "error")
        return redirect(url_for("index"))

    return Response(
        content,
        mimetype="text/plain",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=3000)
