#!/usr/bin/env python3
"""
Generate llms.txt and llms-full.txt files for a website (CLI version).

Supports two modes for URL discovery:
1. Firecrawl API: Automatically maps and scrapes a website
2. Screaming Frog CSV: Import a Screaming Frog "Internal All" CSV export

Uses Patterns Bifrost API (OpenAI-compatible) for AI-generated summaries.

Output follows the llms.txt spec (https://llmstxt.org):
  # Site Name
  > Summary
  ## Section
  - [Title](url): Description
  ## Optional
  - [Title](url): Description
"""

import csv
import io
import json
import logging
import os
import sys
import time
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests
from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BIFROST_BASE_URL = "https://bifrost.pattern.com"
BIFROST_MODEL = "openai/gpt-4o-mini"
OPTIONAL_DEPTH_THRESHOLD = 4
OPTIONAL_INLINKS_THRESHOLD = 1

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_int(val: str) -> int:
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return 0


def _url_to_section(url: str) -> str:
    path = urlparse(url).path.strip("/")
    if not path:
        return "Main"
    first_segment = path.split("/")[0]
    return first_segment.replace("-", " ").replace("_", " ").title()


def _group_into_sections(
    results: List[Dict],
) -> Tuple[Dict[str, List[Dict]], List[Dict]]:
    sections: Dict[str, List[Dict]] = defaultdict(list)
    optional: List[Dict] = []

    for r in results:
        crawl_depth = r.get("crawl_depth", 0)
        unique_inlinks = r.get("unique_inlinks", 0)

        is_optional = (
            crawl_depth >= OPTIONAL_DEPTH_THRESHOLD
            and unique_inlinks <= OPTIONAL_INLINKS_THRESHOLD
        )

        if is_optional:
            optional.append(r)
        else:
            sections[_url_to_section(r["url"])].append(r)

    return dict(sections), optional


def _format_spec_llmstxt(
    site_url: str,
    site_name: str,
    site_summary: str,
    sections: Dict[str, List[Dict]],
    optional: List[Dict],
) -> str:
    lines = [f"# {site_name}\n"]
    if site_summary:
        lines.append(f"> {site_summary}\n")

    section_order = sorted(sections.keys(), key=lambda s: (s != "Main", s))
    for section_name in section_order:
        items = sections[section_name]
        lines.append(f"\n## {section_name}\n")
        for r in items:
            lines.append(f"- [{r['title']}]({r['url']}): {r['description']}")

    if optional:
        lines.append("\n## Optional\n")
        for r in optional:
            lines.append(f"- [{r['title']}]({r['url']}): {r['description']}")

    return "\n".join(lines) + "\n"


def _format_spec_llms_full(site_name: str, results: List[Dict]) -> str:
    lines = [f"# {site_name}\n"]
    for r in results:
        if r.get("markdown"):
            lines.append(f"\n---\n\n## {r['title']}\n\nSource: {r['url']}\n")
            lines.append(r["markdown"])
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Screaming Frog CSV parser
# ---------------------------------------------------------------------------


def parse_screaming_frog_csv(csv_path: str, max_urls: int = 0) -> List[Dict]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    results = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV file has no headers")

        header_map = {h.lower().strip(): h for h in reader.fieldnames}

        def get_field(row, *names):
            for name in names:
                key = header_map.get(name.lower().strip())
                if key and row.get(key):
                    return row[key].strip()
            return ""

        for row in reader:
            address = get_field(row, "Address", "URL")
            if not address:
                continue
            status_code = get_field(row, "Status Code")
            content_type = get_field(row, "Content Type")
            indexability = get_field(row, "Indexability")

            if status_code and status_code != "200":
                continue
            if content_type and "text/html" not in content_type.lower():
                continue
            if indexability and indexability.lower() == "non-indexable":
                continue

            title = get_field(row, "Title 1", "Title")
            description = get_field(row, "Meta Description 1", "Meta Description", "Description 1")
            h1 = get_field(row, "H1-1", "H1")
            word_count = get_field(row, "Word Count")
            crawl_depth = get_field(row, "Crawl Depth")
            inlinks = get_field(row, "Inlinks")
            unique_inlinks = get_field(row, "Unique Inlinks")

            results.append({
                "url": address,
                "title": title or h1 or "",
                "description": description or "",
                "word_count": _safe_int(word_count),
                "crawl_depth": _safe_int(crawl_depth),
                "inlinks": _safe_int(inlinks),
                "unique_inlinks": _safe_int(unique_inlinks),
            })

            if max_urls and len(results) >= max_urls:
                break

    logger.info(f"Parsed {len(results)} URLs from Screaming Frog CSV")
    return results


# ---------------------------------------------------------------------------
# Generator class
# ---------------------------------------------------------------------------


class LLMsTextGenerator:
    def __init__(
        self,
        firecrawl_api_key: Optional[str] = None,
        bifrost_api_key: Optional[str] = None,
    ):
        self.firecrawl_api_key = firecrawl_api_key
        self.llm_client = (
            OpenAI(base_url=BIFROST_BASE_URL, api_key=bifrost_api_key)
            if bifrost_api_key else None
        )
        self.firecrawl_base_url = "https://api.firecrawl.dev/v1"
        self.firecrawl_headers = (
            {"Authorization": f"Bearer {firecrawl_api_key}", "Content-Type": "application/json"}
            if firecrawl_api_key else {}
        )

    # -- Firecrawl ----------------------------------------------------------

    def map_website(self, url: str, limit: int = 100) -> List[str]:
        response = requests.post(
            f"{self.firecrawl_base_url}/map",
            headers=self.firecrawl_headers,
            json={"url": url, "limit": limit, "includeSubdomains": False, "ignoreSitemap": False},
        )
        response.raise_for_status()
        data = response.json()
        return data["links"] if data.get("success") and data.get("links") else []

    def scrape_url(self, url: str) -> Optional[Dict]:
        if not self.firecrawl_api_key:
            return None
        try:
            response = requests.post(
                f"{self.firecrawl_base_url}/scrape",
                headers=self.firecrawl_headers,
                json={"url": url, "formats": ["markdown"], "onlyMainContent": True, "timeout": 30000},
            )
            response.raise_for_status()
            data = response.json()
            if data.get("success") and data.get("data"):
                return {"url": url, "markdown": data["data"].get("markdown", "")}
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
        return None

    # -- AI helpers ---------------------------------------------------------

    def _call_llm(self, prompt: str, system: str) -> dict:
        resp = self.llm_client.chat.completions.create(
            model=BIFROST_MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=100,
        )
        return json.loads(resp.choices[0].message.content)

    def generate_description(self, url: str, markdown: str) -> Tuple[str, str]:
        if not self.llm_client:
            return "Page", "No description available"
        prompt = (
            f"Generate a 9-10 word description and a 3-4 word title for: {url}\n\n"
            f'Return JSON: {{"title": "3-4 word title", "description": "9-10 word description"}}\n\n'
            f"Page content:\n{markdown[:4000]}"
        )
        try:
            r = self._call_llm(prompt, "You generate concise titles and descriptions for web pages.")
            return r.get("title", "Page"), r.get("description", "No description available")
        except Exception as e:
            logger.error(f"Error generating description: {e}")
            return "Page", "No description available"

    def generate_description_from_metadata(self, url: str, title: str, desc: str) -> Tuple[str, str]:
        if not self.llm_client:
            return (" ".join(title.split()[:4]) if title else "Page",
                     " ".join(desc.split()[:10]) if desc else "No description available")
        prompt = (
            f"Generate a 3-4 word title and 9-10 word description.\n"
            f"URL: {url}\nTitle: {title}\nDescription: {desc}\n\n"
            f'Return JSON: {{"title": "...", "description": "..."}}'
        )
        try:
            r = self._call_llm(prompt, "You generate concise titles and descriptions for web pages.")
            return r.get("title", "Page"), r.get("description", "No description available")
        except Exception as e:
            logger.error(f"Error: {e}")
            return (" ".join(title.split()[:4]) if title else "Page",
                     " ".join(desc.split()[:10]) if desc else "No description available")

    def generate_site_summary(self, site_url: str, page_titles: List[str]) -> Tuple[str, str]:
        domain = urlparse(site_url).netloc.replace("www.", "")
        if not self.llm_client:
            return domain, ""
        titles_sample = "\n".join(page_titles[:30])
        prompt = (
            f"Given this URL and page titles, generate:\n"
            f"1. A short site name (1-3 words)\n"
            f"2. A one-sentence summary (15-25 words)\n\n"
            f"URL: {site_url}\nTitles:\n{titles_sample}\n\n"
            f'Return JSON: {{"name": "Site Name", "summary": "Summary."}}'
        )
        try:
            r = self._call_llm(prompt, "You generate site names and summaries for llms.txt files.")
            return r.get("name", domain), r.get("summary", "")
        except Exception as e:
            logger.error(f"Error generating site summary: {e}")
            return domain, ""

    # -- Processors ---------------------------------------------------------

    def process_url_firecrawl(self, url: str, index: int) -> Optional[Dict]:
        scraped = self.scrape_url(url)
        if not scraped or not scraped.get("markdown"):
            return None
        title, desc = self.generate_description(url, scraped["markdown"])
        return {"url": url, "title": title, "description": desc,
                "markdown": scraped["markdown"], "index": index,
                "crawl_depth": 0, "inlinks": 0, "unique_inlinks": 0}

    def process_url_csv(self, entry: Dict, index: int, scrape: bool = False) -> Optional[Dict]:
        url = entry["url"]
        markdown = ""
        if scrape:
            scraped = self.scrape_url(url)
            if scraped and scraped.get("markdown"):
                markdown = scraped["markdown"]
        if markdown and self.llm_client:
            title, desc = self.generate_description(url, markdown)
        else:
            title, desc = self.generate_description_from_metadata(
                url, entry.get("title", ""), entry.get("description", ""))
        return {"url": url, "title": title, "description": desc,
                "markdown": markdown, "index": index,
                "crawl_depth": entry.get("crawl_depth", 0),
                "inlinks": entry.get("inlinks", 0),
                "unique_inlinks": entry.get("unique_inlinks", 0)}

    # -- Generation ---------------------------------------------------------

    def generate(
        self, site_url: str, urls: Optional[List[str]] = None,
        csv_entries: Optional[List[Dict]] = None,
        max_urls: int = 100, scrape: bool = False,
        generate_full: bool = True, use_ai: bool = True,
    ) -> Dict:
        """Unified generation method for both modes."""
        if csv_entries:
            items = csv_entries
        elif urls:
            items = urls[:max_urls]
        else:
            items = self.map_website(site_url, max_urls)
            if not items:
                raise ValueError("No URLs found for the website")

        is_csv = csv_entries is not None
        batch_size = 10
        all_results = []

        # Process: no-AI CSV shortcut
        if is_csv and not use_ai and not scrape:
            for i, entry in enumerate(items):
                title = entry.get("title", "") or "Page"
                if len(title.split()) > 6:
                    title = " ".join(title.split()[:6])
                desc = entry.get("description", "") or "No description available"
                if len(desc.split()) > 15:
                    desc = " ".join(desc.split()[:12])
                all_results.append({
                    "url": entry["url"], "title": title, "description": desc,
                    "markdown": "", "index": i,
                    "crawl_depth": entry.get("crawl_depth", 0),
                    "inlinks": entry.get("inlinks", 0),
                    "unique_inlinks": entry.get("unique_inlinks", 0),
                })
        else:
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                logger.info(f"Processing batch {i // batch_size + 1}/{(len(items) + batch_size - 1) // batch_size}")

                with ThreadPoolExecutor(max_workers=5) as executor:
                    if is_csv:
                        futures = {executor.submit(self.process_url_csv, entry, i + j, scrape): entry
                                   for j, entry in enumerate(batch)}
                    else:
                        futures = {executor.submit(self.process_url_firecrawl, url, i + j): url
                                   for j, url in enumerate(batch)}

                    for future in as_completed(futures):
                        try:
                            result = future.result()
                            if result:
                                all_results.append(result)
                        except Exception as e:
                            logger.error(f"Failed: {e}")

                if i + batch_size < len(items):
                    time.sleep(1)

        all_results.sort(key=lambda x: x["index"])

        # Site summary
        page_titles = [r["title"] for r in all_results if r.get("title")]
        site_name, site_summary = self.generate_site_summary(site_url, page_titles)

        # Group into sections
        sections, optional = _group_into_sections(all_results)

        llmstxt = _format_spec_llmstxt(site_url, site_name, site_summary, sections, optional)

        llms_fulltxt = ""
        if generate_full and any(r.get("markdown") for r in all_results):
            llms_fulltxt = _format_spec_llms_full(site_name, all_results)

        return {
            "llmstxt": llmstxt,
            "llms_fulltxt": llms_fulltxt,
            "num_urls_processed": len(all_results),
            "num_urls_total": len(items),
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate spec-compliant llms.txt and llms-full.txt files"
    )
    parser.add_argument("url", help="The website URL to process")
    parser.add_argument("--csv", dest="csv_path",
                        help="Path to Screaming Frog 'Internal All' CSV export")
    parser.add_argument("--scrape", action="store_true",
                        help="When using --csv, also scrape URLs via Firecrawl")
    parser.add_argument("--no-ai", action="store_true",
                        help="Skip AI generation, use CSV metadata directly")
    parser.add_argument("--max-urls", type=int, default=20,
                        help="Max URLs to process (default: 20, 0 for unlimited)")
    parser.add_argument("--output-dir", default=".",
                        help="Directory to save output files")
    parser.add_argument("--firecrawl-api-key", default=os.getenv("FIRECRAWL_API_KEY"))
    parser.add_argument("--bifrost-api-key", default=os.getenv("BIFROST_API_KEY"))
    parser.add_argument("--no-full-text", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    using_csv = args.csv_path is not None
    needs_firecrawl = not using_csv or args.scrape
    needs_bifrost = not args.no_ai

    if needs_firecrawl and not args.firecrawl_api_key:
        logger.error("Firecrawl API key not provided. Set FIRECRAWL_API_KEY or use --firecrawl-api-key")
        sys.exit(1)
    if needs_bifrost and not args.bifrost_api_key:
        if using_csv:
            logger.warning("Bifrost API key not provided. Using CSV metadata directly.")
            args.no_ai = True
        else:
            logger.error("Bifrost API key not provided. Set BIFROST_API_KEY or use --bifrost-api-key")
            sys.exit(1)

    generator = LLMsTextGenerator(
        firecrawl_api_key=args.firecrawl_api_key if needs_firecrawl else None,
        bifrost_api_key=args.bifrost_api_key if not args.no_ai else None,
    )

    try:
        csv_entries = parse_screaming_frog_csv(args.csv_path, args.max_urls) if using_csv else None

        result = generator.generate(
            site_url=args.url,
            csv_entries=csv_entries,
            max_urls=args.max_urls,
            scrape=args.scrape,
            generate_full=not args.no_full_text,
            use_ai=not args.no_ai,
        )

        os.makedirs(args.output_dir, exist_ok=True)
        domain = urlparse(args.url).netloc.replace("www.", "")

        llmstxt_path = os.path.join(args.output_dir, f"{domain}-llms.txt")
        with open(llmstxt_path, "w", encoding="utf-8") as f:
            f.write(result["llmstxt"])
        logger.info(f"Saved llms.txt to {llmstxt_path}")

        if result.get("llms_fulltxt"):
            fulltxt_path = os.path.join(args.output_dir, f"{domain}-llms-full.txt")
            with open(fulltxt_path, "w", encoding="utf-8") as f:
                f.write(result["llms_fulltxt"])
            logger.info(f"Saved llms-full.txt to {fulltxt_path}")

        print(f"\nSuccess! Processed {result['num_urls_processed']} of {result['num_urls_total']} URLs")
        print(f"Files saved to {os.path.abspath(args.output_dir)}/")

    except Exception as e:
        logger.error(f"Failed to generate llms.txt: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
