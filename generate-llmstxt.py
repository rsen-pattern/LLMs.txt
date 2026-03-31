#!/usr/bin/env python3
"""
Generate llms.txt and llms-full.txt files for a website using Firecrawl and OpenAI.

Supports two modes for URL discovery:
1. Firecrawl API: Automatically maps and scrapes a website
2. Screaming Frog CSV: Import a Screaming Frog "Internal All" CSV export

This script:
1. Discovers URLs via Firecrawl /map endpoint or Screaming Frog CSV import
2. Scrapes each URL to get markdown content (via Firecrawl)
3. Uses OpenAI to generate titles and descriptions (or uses existing metadata from CSV)
4. Creates llms.txt (index with descriptions) and llms-full.txt (full content)
"""

import os
import sys
import json
import csv
import time
import argparse
import logging
import re
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

import requests
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_screaming_frog_csv(csv_path: str, max_urls: int = 0) -> List[Dict]:
    """
    Parse a Screaming Frog 'Internal All' CSV export.

    Expected columns (case-insensitive matching):
    - Address: The URL
    - Status Code: HTTP status code
    - Content Type: MIME type
    - Title 1: Page title
    - Meta Description 1: Meta description
    - H1-1: First H1 heading
    - Word Count: Number of words on the page
    - Indexability: Whether the page is indexable

    Returns a list of dicts with url, title, description, and other metadata.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    results = []

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        # Normalize header names to lowercase for flexible matching
        if reader.fieldnames is None:
            raise ValueError("CSV file has no headers")

        header_map = {h.lower().strip(): h for h in reader.fieldnames}

        def get_field(row: Dict, *field_names: str) -> str:
            """Try multiple possible column names, return first match."""
            for name in field_names:
                key = header_map.get(name.lower().strip())
                if key and row.get(key):
                    return row[key].strip()
            return ""

        for row in reader:
            address = get_field(row, "Address", "URL", "address", "url")
            if not address:
                continue

            status_code = get_field(row, "Status Code", "status code")
            content_type = get_field(row, "Content Type", "content type")
            indexability = get_field(row, "Indexability", "indexability")

            # Filter: only include HTML pages that returned 200
            if status_code and status_code != "200":
                continue
            if content_type and "text/html" not in content_type.lower():
                continue
            if indexability and indexability.lower() == "non-indexable":
                continue

            title = get_field(row, "Title 1", "Title", "title 1", "title")
            description = get_field(
                row,
                "Meta Description 1",
                "Meta Description",
                "meta description 1",
                "meta description",
                "Description 1",
            )
            h1 = get_field(row, "H1-1", "H1", "h1-1", "h1")
            word_count = get_field(row, "Word Count", "word count")

            results.append(
                {
                    "url": address,
                    "title": title or h1 or "",
                    "description": description or "",
                    "word_count": int(word_count) if word_count.isdigit() else 0,
                }
            )

            if max_urls and len(results) >= max_urls:
                break

    logger.info(f"Parsed {len(results)} URLs from Screaming Frog CSV")
    return results


class LLMsTextGenerator:
    """Generate llms.txt files using Firecrawl API and/or Screaming Frog CSV data."""

    def __init__(
        self,
        firecrawl_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
    ):
        """Initialize the generator with API keys."""
        self.firecrawl_api_key = firecrawl_api_key
        self.openai_client = OpenAI(api_key=openai_api_key) if openai_api_key else None
        self.firecrawl_base_url = "https://api.firecrawl.dev/v1"
        self.headers = (
            {
                "Authorization": f"Bearer {self.firecrawl_api_key}",
                "Content-Type": "application/json",
            }
            if firecrawl_api_key
            else {}
        )

    def map_website(self, url: str, limit: int = 100) -> List[str]:
        """Map a website to get all URLs using Firecrawl."""
        if not self.firecrawl_api_key:
            raise ValueError("Firecrawl API key required for website mapping")

        logger.info(f"Mapping website: {url} (limit: {limit})")

        try:
            response = requests.post(
                f"{self.firecrawl_base_url}/map",
                headers=self.headers,
                json={
                    "url": url,
                    "limit": limit,
                    "includeSubdomains": False,
                    "ignoreSitemap": False,
                },
            )
            response.raise_for_status()

            data = response.json()
            if data.get("success") and data.get("links"):
                urls = data["links"]
                logger.info(f"Found {len(urls)} URLs")
                return urls
            else:
                logger.error(f"Failed to map website: {data}")
                return []

        except Exception as e:
            logger.error(f"Error mapping website: {e}")
            return []

    def scrape_url(self, url: str) -> Optional[Dict]:
        """Scrape a single URL using Firecrawl."""
        if not self.firecrawl_api_key:
            return None

        logger.debug(f"Scraping URL: {url}")

        try:
            response = requests.post(
                f"{self.firecrawl_base_url}/scrape",
                headers=self.headers,
                json={
                    "url": url,
                    "formats": ["markdown"],
                    "onlyMainContent": True,
                    "timeout": 30000,
                },
            )
            response.raise_for_status()

            data = response.json()
            if data.get("success") and data.get("data"):
                return {
                    "url": url,
                    "markdown": data["data"].get("markdown", ""),
                    "metadata": data["data"].get("metadata", {}),
                }
            else:
                logger.error(f"Failed to scrape {url}: {data}")
                return None

        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return None

    def generate_description(self, url: str, markdown: str) -> Tuple[str, str]:
        """Generate title and description using OpenAI."""
        if not self.openai_client:
            raise ValueError("OpenAI API key required for description generation")

        logger.debug(f"Generating description for: {url}")

        prompt = f"""Generate a 9-10 word description and a 3-4 word title of the entire page based on ALL the content one will find on the page for this url: {url}. This will help in a user finding the page for its intended purpose.

Return the response in JSON format:
{{
    "title": "3-4 word title",
    "description": "9-10 word description"
}}"""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that generates concise titles and descriptions for web pages.",
                    },
                    {
                        "role": "user",
                        "content": f"{prompt}\n\nPage content:\n{markdown[:4000]}",
                    },
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=100,
            )

            result = json.loads(response.choices[0].message.content)
            return result.get("title", "Page"), result.get(
                "description", "No description available"
            )

        except Exception as e:
            logger.error(f"Error generating description: {e}")
            return "Page", "No description available"

    def generate_description_from_metadata(
        self, url: str, title: str, description: str
    ) -> Tuple[str, str]:
        """Generate or refine title/description using OpenAI from existing metadata."""
        if not self.openai_client:
            # Fall back to existing metadata
            short_title = " ".join(title.split()[:4]) if title else "Page"
            short_desc = (
                " ".join(description.split()[:10])
                if description
                else "No description available"
            )
            return short_title, short_desc

        logger.debug(f"Generating description from metadata for: {url}")

        prompt = f"""Based on the following page metadata, generate a concise 3-4 word title and a 9-10 word description. This will help users find the page for its intended purpose.

URL: {url}
Existing Title: {title}
Existing Description: {description}

Return the response in JSON format:
{{
    "title": "3-4 word title",
    "description": "9-10 word description"
}}"""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that generates concise titles and descriptions for web pages.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=100,
            )

            result = json.loads(response.choices[0].message.content)
            return result.get("title", "Page"), result.get(
                "description", "No description available"
            )

        except Exception as e:
            logger.error(f"Error generating description from metadata: {e}")
            short_title = " ".join(title.split()[:4]) if title else "Page"
            short_desc = (
                " ".join(description.split()[:10])
                if description
                else "No description available"
            )
            return short_title, short_desc

    def process_url_firecrawl(self, url: str, index: int) -> Optional[Dict]:
        """Process a single URL via Firecrawl: scrape and generate description."""
        scraped_data = self.scrape_url(url)
        if not scraped_data or not scraped_data.get("markdown"):
            return None

        title, description = self.generate_description(
            url, scraped_data["markdown"]
        )

        return {
            "url": url,
            "title": title,
            "description": description,
            "markdown": scraped_data["markdown"],
            "index": index,
        }

    def process_url_csv(
        self, entry: Dict, index: int, scrape: bool = False
    ) -> Optional[Dict]:
        """Process a single URL from CSV data, optionally scraping for full content."""
        url = entry["url"]
        markdown = ""

        if scrape:
            scraped_data = self.scrape_url(url)
            if scraped_data and scraped_data.get("markdown"):
                markdown = scraped_data["markdown"]

        # Generate or refine title/description
        if markdown and self.openai_client:
            title, description = self.generate_description(url, markdown)
        else:
            title, description = self.generate_description_from_metadata(
                url, entry.get("title", ""), entry.get("description", "")
            )

        return {
            "url": url,
            "title": title,
            "description": description,
            "markdown": markdown,
            "index": index,
        }

    def generate_from_firecrawl(
        self, url: str, max_urls: int = 100, show_full_text: bool = True
    ) -> Dict[str, str]:
        """Generate llms.txt using Firecrawl for URL discovery and scraping."""
        logger.info(f"Generating llms.txt for {url} via Firecrawl")

        urls = self.map_website(url, max_urls)
        if not urls:
            raise ValueError("No URLs found for the website")

        urls = urls[:max_urls]
        return self._process_urls_firecrawl(url, urls, show_full_text)

    def generate_from_csv(
        self,
        csv_path: str,
        site_url: str,
        max_urls: int = 0,
        scrape: bool = False,
        show_full_text: bool = True,
        use_ai: bool = True,
    ) -> Dict[str, str]:
        """Generate llms.txt using Screaming Frog CSV for URL data."""
        logger.info(f"Generating llms.txt from Screaming Frog CSV: {csv_path}")

        csv_entries = parse_screaming_frog_csv(csv_path, max_urls)
        if not csv_entries:
            raise ValueError("No valid URLs found in CSV file")

        return self._process_urls_csv(
            site_url, csv_entries, scrape, show_full_text, use_ai
        )

    def _process_urls_firecrawl(
        self, site_url: str, urls: List[str], show_full_text: bool
    ) -> Dict[str, str]:
        """Process URLs discovered via Firecrawl."""
        llmstxt = f"# {site_url} llms.txt\n\n"
        llms_fulltxt = f"# {site_url} llms-full.txt\n\n"

        batch_size = 10
        all_results = []

        for i in range(0, len(urls), batch_size):
            batch = urls[i : i + batch_size]
            logger.info(
                f"Processing batch {i // batch_size + 1}/"
                f"{(len(urls) + batch_size - 1) // batch_size}"
            )

            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {
                    executor.submit(self.process_url_firecrawl, url, i + j): (
                        url,
                        i + j,
                    )
                    for j, url in enumerate(batch)
                }

                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result:
                            all_results.append(result)
                    except Exception as e:
                        url, idx = futures[future]
                        logger.error(f"Failed to process {url}: {e}")

            if i + batch_size < len(urls):
                time.sleep(1)

        all_results.sort(key=lambda x: x["index"])

        for i, result in enumerate(all_results, 1):
            llmstxt += (
                f"- [{result['title']}]({result['url']}): {result['description']}\n"
            )
            if show_full_text:
                llms_fulltxt += (
                    f"<|page-{i}-llmstxt|>\n"
                    f"## {result['title']}\n"
                    f"{result['markdown']}\n\n"
                )

        return {
            "llmstxt": llmstxt,
            "llms_fulltxt": llms_fulltxt if show_full_text else "",
            "num_urls_processed": len(all_results),
            "num_urls_total": len(urls),
        }

    def _process_urls_csv(
        self,
        site_url: str,
        csv_entries: List[Dict],
        scrape: bool,
        show_full_text: bool,
        use_ai: bool,
    ) -> Dict[str, str]:
        """Process URLs from Screaming Frog CSV data."""
        llmstxt = f"# {site_url} llms.txt\n\n"
        llms_fulltxt = f"# {site_url} llms-full.txt\n\n"

        batch_size = 10
        all_results = []

        for i in range(0, len(csv_entries), batch_size):
            batch = csv_entries[i : i + batch_size]
            logger.info(
                f"Processing batch {i // batch_size + 1}/"
                f"{(len(csv_entries) + batch_size - 1) // batch_size}"
            )

            if use_ai or scrape:
                with ThreadPoolExecutor(max_workers=5) as executor:
                    futures = {
                        executor.submit(
                            self.process_url_csv, entry, i + j, scrape
                        ): (entry["url"], i + j)
                        for j, entry in enumerate(batch)
                    }

                    for future in as_completed(futures):
                        try:
                            result = future.result()
                            if result:
                                all_results.append(result)
                        except Exception as e:
                            url, idx = futures[future]
                            logger.error(f"Failed to process {url}: {e}")
            else:
                # No API calls needed - use CSV metadata directly
                for j, entry in enumerate(batch):
                    title = entry.get("title", "Page")
                    if not title:
                        title = "Page"
                    # Truncate long titles to ~4 words
                    title_words = title.split()
                    if len(title_words) > 4:
                        title = " ".join(title_words[:4])

                    description = entry.get("description", "No description available")
                    if not description:
                        description = "No description available"
                    # Truncate long descriptions to ~10 words
                    desc_words = description.split()
                    if len(desc_words) > 12:
                        description = " ".join(desc_words[:10])

                    all_results.append(
                        {
                            "url": entry["url"],
                            "title": title,
                            "description": description,
                            "markdown": "",
                            "index": i + j,
                        }
                    )

            if (use_ai or scrape) and i + batch_size < len(csv_entries):
                time.sleep(1)

        all_results.sort(key=lambda x: x["index"])

        for i, result in enumerate(all_results, 1):
            llmstxt += (
                f"- [{result['title']}]({result['url']}): {result['description']}\n"
            )
            if show_full_text and result.get("markdown"):
                llms_fulltxt += (
                    f"<|page-{i}-llmstxt|>\n"
                    f"## {result['title']}\n"
                    f"{result['markdown']}\n\n"
                )

        has_full_content = any(r.get("markdown") for r in all_results)

        return {
            "llmstxt": llmstxt,
            "llms_fulltxt": llms_fulltxt if show_full_text and has_full_content else "",
            "num_urls_processed": len(all_results),
            "num_urls_total": len(csv_entries),
        }


def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate llms.txt and llms-full.txt files for a website "
            "using Firecrawl API and/or Screaming Frog CSV export"
        )
    )
    parser.add_argument("url", help="The website URL to process")
    parser.add_argument(
        "--csv",
        dest="csv_path",
        help="Path to Screaming Frog 'Internal All' CSV export (use instead of Firecrawl mapping)",
    )
    parser.add_argument(
        "--scrape",
        action="store_true",
        help="When using --csv, also scrape URLs via Firecrawl for full markdown content",
    )
    parser.add_argument(
        "--no-ai",
        action="store_true",
        help="When using --csv, skip AI title/description generation and use CSV metadata directly",
    )
    parser.add_argument(
        "--max-urls",
        type=int,
        default=20,
        help="Maximum number of URLs to process (default: 20, 0 for unlimited)",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to save output files (default: current directory)",
    )
    parser.add_argument(
        "--firecrawl-api-key",
        default=os.getenv("FIRECRAWL_API_KEY"),
        help="Firecrawl API key (default: from FIRECRAWL_API_KEY env var)",
    )
    parser.add_argument(
        "--openai-api-key",
        default=os.getenv("OPENAI_API_KEY"),
        help="OpenAI API key (default: from OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--no-full-text",
        action="store_true",
        help="Don't generate llms-full.txt file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Validate API keys based on mode
    using_csv = args.csv_path is not None
    needs_firecrawl = not using_csv or args.scrape
    needs_openai = not args.no_ai

    if needs_firecrawl and not args.firecrawl_api_key:
        logger.error(
            "Firecrawl API key not provided. "
            "Set FIRECRAWL_API_KEY environment variable or use --firecrawl-api-key"
        )
        sys.exit(1)

    if needs_openai and not args.openai_api_key:
        if using_csv:
            logger.warning(
                "OpenAI API key not provided. Using CSV metadata directly for titles/descriptions. "
                "Use --openai-api-key for AI-generated summaries."
            )
            args.no_ai = True
        else:
            logger.error(
                "OpenAI API key not provided. "
                "Set OPENAI_API_KEY environment variable or use --openai-api-key"
            )
            sys.exit(1)

    # Create generator
    generator = LLMsTextGenerator(
        firecrawl_api_key=args.firecrawl_api_key if needs_firecrawl else None,
        openai_api_key=args.openai_api_key if not args.no_ai else None,
    )

    try:
        if using_csv:
            result = generator.generate_from_csv(
                csv_path=args.csv_path,
                site_url=args.url,
                max_urls=args.max_urls,
                scrape=args.scrape,
                show_full_text=not args.no_full_text,
                use_ai=not args.no_ai,
            )
        else:
            result = generator.generate_from_firecrawl(
                args.url,
                args.max_urls,
                not args.no_full_text,
            )

        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)

        # Extract domain from URL for filename
        domain = urlparse(args.url).netloc.replace("www.", "")

        # Save llms.txt
        llmstxt_path = os.path.join(args.output_dir, f"{domain}-llms.txt")
        with open(llmstxt_path, "w", encoding="utf-8") as f:
            f.write(result["llmstxt"])
        logger.info(f"Saved llms.txt to {llmstxt_path}")

        # Save llms-full.txt if available
        if result.get("llms_fulltxt"):
            llms_fulltxt_path = os.path.join(
                args.output_dir, f"{domain}-llms-full.txt"
            )
            with open(llms_fulltxt_path, "w", encoding="utf-8") as f:
                f.write(result["llms_fulltxt"])
            logger.info(f"Saved llms-full.txt to {llms_fulltxt_path}")

        # Print summary
        print(
            f"\nSuccess! Processed {result['num_urls_processed']} "
            f"out of {result['num_urls_total']} URLs"
        )
        print(f"Files saved to {os.path.abspath(args.output_dir)}/")

    except Exception as e:
        logger.error(f"Failed to generate llms.txt: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
