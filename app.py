"""
LLMs.txt Generator — Streamlit App

Generate llms.txt and llms-full.txt files for any website using:
- Firecrawl API for automatic URL discovery and scraping
- Screaming Frog "Internal All" CSV import for URL discovery
- Patterns Bifrost API (OpenAI-compatible) for AI-generated summaries
"""

import csv
import io
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests
import streamlit as st
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bifrost / LLM client helpers
# ---------------------------------------------------------------------------

BIFROST_BASE_URL = "https://bifrost.pattern.com"
BIFROST_MODEL = "openai/gpt-4o-mini"


def get_llm_client(api_key: str) -> OpenAI:
    """Return an OpenAI-compatible client pointed at Patterns Bifrost."""
    return OpenAI(base_url=BIFROST_BASE_URL, api_key=api_key)


# ---------------------------------------------------------------------------
# Screaming Frog CSV parser
# ---------------------------------------------------------------------------


def parse_screaming_frog_csv(file_contents: str, max_urls: int = 0) -> List[Dict]:
    """Parse a Screaming Frog 'Internal All' CSV from an in-memory string."""
    reader = csv.DictReader(io.StringIO(file_contents))

    if reader.fieldnames is None:
        raise ValueError("CSV file has no headers")

    header_map = {h.lower().strip(): h for h in reader.fieldnames}

    def get_field(row: Dict, *field_names: str) -> str:
        for name in field_names:
            key = header_map.get(name.lower().strip())
            if key and row.get(key):
                return row[key].strip()
        return ""

    results: List[Dict] = []
    for row in reader:
        address = get_field(row, "Address", "URL", "address", "url")
        if not address:
            continue

        status_code = get_field(row, "Status Code", "status code")
        content_type = get_field(row, "Content Type", "content type")
        indexability = get_field(row, "Indexability", "indexability")

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

    return results


# ---------------------------------------------------------------------------
# Core generator class
# ---------------------------------------------------------------------------


class LLMsTextGenerator:
    """Generate llms.txt files using Firecrawl + Patterns Bifrost."""

    def __init__(
        self,
        firecrawl_api_key: Optional[str] = None,
        bifrost_api_key: Optional[str] = None,
    ):
        self.firecrawl_api_key = firecrawl_api_key
        self.llm_client = get_llm_client(bifrost_api_key) if bifrost_api_key else None
        self.firecrawl_base_url = "https://api.firecrawl.dev/v1"
        self.firecrawl_headers = (
            {
                "Authorization": f"Bearer {firecrawl_api_key}",
                "Content-Type": "application/json",
            }
            if firecrawl_api_key
            else {}
        )

    # -- Firecrawl helpers --------------------------------------------------

    def map_website(self, url: str, limit: int = 100) -> List[str]:
        response = requests.post(
            f"{self.firecrawl_base_url}/map",
            headers=self.firecrawl_headers,
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
            return data["links"]
        return []

    def scrape_url(self, url: str) -> Optional[Dict]:
        if not self.firecrawl_api_key:
            return None
        try:
            response = requests.post(
                f"{self.firecrawl_base_url}/scrape",
                headers=self.firecrawl_headers,
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
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
        return None

    # -- AI description helpers ---------------------------------------------

    def _call_llm(self, prompt: str, system: str) -> dict:
        """Make a chat completion call via Bifrost and return parsed JSON."""
        response = self.llm_client.chat.completions.create(
            model=BIFROST_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=100,
        )
        return json.loads(response.choices[0].message.content)

    def generate_description(self, url: str, markdown: str) -> Tuple[str, str]:
        if not self.llm_client:
            return "Page", "No description available"

        prompt = (
            f"Generate a 9-10 word description and a 3-4 word title of the entire "
            f"page based on ALL the content for this url: {url}. "
            f"This will help a user find the page for its intended purpose.\n\n"
            f'Return JSON: {{"title": "3-4 word title", "description": "9-10 word description"}}\n\n'
            f"Page content:\n{markdown[:4000]}"
        )
        try:
            result = self._call_llm(
                prompt,
                "You are a helpful assistant that generates concise titles and descriptions for web pages.",
            )
            return result.get("title", "Page"), result.get(
                "description", "No description available"
            )
        except Exception as e:
            logger.error(f"Error generating description: {e}")
            return "Page", "No description available"

    def generate_description_from_metadata(
        self, url: str, title: str, description: str
    ) -> Tuple[str, str]:
        if not self.llm_client:
            short_title = " ".join(title.split()[:4]) if title else "Page"
            short_desc = (
                " ".join(description.split()[:10])
                if description
                else "No description available"
            )
            return short_title, short_desc

        prompt = (
            f"Based on the following page metadata, generate a concise 3-4 word title "
            f"and a 9-10 word description.\n\n"
            f"URL: {url}\nExisting Title: {title}\nExisting Description: {description}\n\n"
            f'Return JSON: {{"title": "3-4 word title", "description": "9-10 word description"}}'
        )
        try:
            result = self._call_llm(
                prompt,
                "You are a helpful assistant that generates concise titles and descriptions for web pages.",
            )
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

    # -- URL processors -----------------------------------------------------

    def process_url_firecrawl(self, url: str, index: int) -> Optional[Dict]:
        scraped = self.scrape_url(url)
        if not scraped or not scraped.get("markdown"):
            return None
        title, desc = self.generate_description(url, scraped["markdown"])
        return {
            "url": url,
            "title": title,
            "description": desc,
            "markdown": scraped["markdown"],
            "index": index,
        }

    def process_url_csv(
        self, entry: Dict, index: int, scrape: bool = False
    ) -> Optional[Dict]:
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
                url, entry.get("title", ""), entry.get("description", "")
            )

        return {
            "url": url,
            "title": title,
            "description": desc,
            "markdown": markdown,
            "index": index,
        }

    # -- Main generation methods --------------------------------------------

    def generate_from_firecrawl(
        self,
        url: str,
        max_urls: int = 100,
        generate_full: bool = True,
        progress_callback=None,
    ) -> Dict[str, str]:
        urls = self.map_website(url, max_urls)
        if not urls:
            raise ValueError("No URLs found for the website")
        urls = urls[:max_urls]
        return self._process_urls(
            site_url=url,
            items=[(u, i) for i, u in enumerate(urls)],
            processor=lambda item: self.process_url_firecrawl(item[0], item[1]),
            generate_full=generate_full,
            progress_callback=progress_callback,
        )

    def generate_from_csv(
        self,
        csv_entries: List[Dict],
        site_url: str,
        scrape: bool = False,
        generate_full: bool = True,
        use_ai: bool = True,
        progress_callback=None,
    ) -> Dict[str, str]:
        if not csv_entries:
            raise ValueError("No valid URLs found in CSV data")

        if not use_ai and not scrape:
            return self._build_from_metadata(site_url, csv_entries, generate_full)

        return self._process_urls(
            site_url=site_url,
            items=[(entry, i) for i, entry in enumerate(csv_entries)],
            processor=lambda item: self.process_url_csv(item[0], item[1], scrape),
            generate_full=generate_full,
            progress_callback=progress_callback,
        )

    # -- Internal helpers ---------------------------------------------------

    def _process_urls(
        self, site_url, items, processor, generate_full, progress_callback=None
    ) -> Dict[str, str]:
        batch_size = 10
        all_results = []
        total = len(items)

        for i in range(0, total, batch_size):
            batch = items[i : i + batch_size]

            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {executor.submit(processor, item): item for item in batch}
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result:
                            all_results.append(result)
                    except Exception as e:
                        logger.error(f"Failed to process item: {e}")

            if progress_callback:
                progress_callback(min(i + batch_size, total), total)

            if i + batch_size < total:
                time.sleep(1)

        all_results.sort(key=lambda x: x["index"])
        return self._format_output(site_url, all_results, generate_full, total)

    def _build_from_metadata(
        self, site_url: str, csv_entries: List[Dict], generate_full: bool
    ) -> Dict[str, str]:
        """Build output directly from CSV metadata — no API calls."""
        all_results = []
        for i, entry in enumerate(csv_entries):
            title = entry.get("title", "") or "Page"
            title_words = title.split()
            if len(title_words) > 4:
                title = " ".join(title_words[:4])

            description = entry.get("description", "") or "No description available"
            desc_words = description.split()
            if len(desc_words) > 12:
                description = " ".join(desc_words[:10])

            all_results.append(
                {
                    "url": entry["url"],
                    "title": title,
                    "description": description,
                    "markdown": "",
                    "index": i,
                }
            )
        return self._format_output(
            site_url, all_results, generate_full, len(csv_entries)
        )

    @staticmethod
    def _format_output(
        site_url: str, results: List[Dict], generate_full: bool, total: int
    ) -> Dict[str, str]:
        llmstxt = f"# {site_url} llms.txt\n\n"
        llms_fulltxt = f"# {site_url} llms-full.txt\n\n"

        has_full_content = False
        for i, r in enumerate(results, 1):
            llmstxt += f"- [{r['title']}]({r['url']}): {r['description']}\n"
            if generate_full and r.get("markdown"):
                has_full_content = True
                llms_fulltxt += (
                    f"<|page-{i}-llmstxt|>\n## {r['title']}\n{r['markdown']}\n\n"
                )

        return {
            "llmstxt": llmstxt,
            "llms_fulltxt": llms_fulltxt if generate_full and has_full_content else "",
            "num_urls_processed": len(results),
            "num_urls_total": total,
        }


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------


def main():
    st.set_page_config(page_title="LLMs.txt Generator", page_icon="📄", layout="wide")

    st.title("📄 LLMs.txt Generator")
    st.markdown(
        "Generate **llms.txt** and **llms-full.txt** files for any website using "
        "Firecrawl API or a Screaming Frog CSV export, powered by "
        "[Patterns Bifrost](https://docs.getbifrost.ai/) for AI summaries."
    )

    # ---- Sidebar: API keys & settings ------------------------------------
    with st.sidebar:
        st.header("API Keys")

        # Read defaults from Streamlit secrets or environment variables
        default_bifrost = st.secrets.get("BIFROST_API_KEY", os.environ.get("BIFROST_API_KEY", ""))
        default_firecrawl = st.secrets.get("FIRECRAWL_API_KEY", os.environ.get("FIRECRAWL_API_KEY", ""))

        bifrost_key = st.text_input(
            "Patterns Bifrost API Key",
            value=default_bifrost,
            type="password",
            help="Your Bifrost virtual key (sk-bf-...)",
        )
        firecrawl_key = st.text_input(
            "Firecrawl API Key",
            value=default_firecrawl,
            type="password",
            help="Required for Firecrawl mode or scraping with CSV mode",
        )

        st.divider()
        st.header("Settings")

        max_urls = st.number_input(
            "Max URLs to process",
            min_value=1,
            max_value=500,
            value=20,
            step=5,
        )
        generate_full = st.checkbox("Generate llms-full.txt", value=True)

    # ---- Main area: input mode -------------------------------------------
    tab_firecrawl, tab_csv = st.tabs(["🔥 Firecrawl (Auto-Crawl)", "📊 Screaming Frog CSV"])

    # -- Firecrawl tab -----------------------------------------------------
    with tab_firecrawl:
        st.markdown(
            "Enter a URL and we'll automatically discover pages via Firecrawl, "
            "scrape their content, and generate AI summaries via Bifrost."
        )
        fc_url = st.text_input(
            "Website URL",
            placeholder="https://example.com",
            key="fc_url",
        )

        if st.button("Generate llms.txt", key="fc_generate", type="primary"):
            if not fc_url:
                st.error("Please enter a website URL.")
            elif not firecrawl_key:
                st.error("Firecrawl API key is required for this mode.")
            elif not bifrost_key:
                st.error("Bifrost API key is required for AI summaries.")
            else:
                _run_firecrawl(fc_url, firecrawl_key, bifrost_key, max_urls, generate_full)

    # -- CSV tab -----------------------------------------------------------
    with tab_csv:
        st.markdown(
            "Upload a Screaming Frog **Internal All** CSV export. "
            "The tool reads `Address`, `Status Code`, `Content Type`, `Title 1`, "
            "`Meta Description 1`, and `Indexability` columns."
        )

        csv_url = st.text_input(
            "Website URL (used as header in output)",
            placeholder="https://example.com",
            key="csv_url",
        )
        uploaded_file = st.file_uploader(
            "Upload Screaming Frog CSV",
            type=["csv"],
            help="Export via Screaming Frog: File > Export > Internal > All",
        )

        col1, col2 = st.columns(2)
        with col1:
            use_ai = st.checkbox(
                "Use AI for titles/descriptions",
                value=True,
                help="Uses Bifrost API. Uncheck to use raw CSV metadata.",
            )
        with col2:
            scrape_content = st.checkbox(
                "Scrape full page content via Firecrawl",
                value=False,
                help="Fetches markdown content for llms-full.txt generation.",
            )

        if st.button("Generate llms.txt", key="csv_generate", type="primary"):
            if not csv_url:
                st.error("Please enter the website URL.")
            elif not uploaded_file:
                st.error("Please upload a Screaming Frog CSV file.")
            elif use_ai and not bifrost_key:
                st.error("Bifrost API key is required when AI summaries are enabled.")
            elif scrape_content and not firecrawl_key:
                st.error("Firecrawl API key is required for scraping.")
            else:
                _run_csv(
                    csv_url,
                    uploaded_file,
                    firecrawl_key,
                    bifrost_key,
                    max_urls,
                    generate_full,
                    use_ai,
                    scrape_content,
                )


# ---------------------------------------------------------------------------
# Runner functions
# ---------------------------------------------------------------------------


def _run_firecrawl(url, firecrawl_key, bifrost_key, max_urls, generate_full):
    """Execute Firecrawl-based generation and display results."""
    generator = LLMsTextGenerator(
        firecrawl_api_key=firecrawl_key,
        bifrost_api_key=bifrost_key,
    )

    progress_bar = st.progress(0, text="Mapping website...")
    status = st.empty()

    def on_progress(done, total):
        progress_bar.progress(done / total, text=f"Processing URLs... ({done}/{total})")

    try:
        status.info("Discovering URLs via Firecrawl...")
        result = generator.generate_from_firecrawl(
            url, max_urls, generate_full, progress_callback=on_progress
        )
        progress_bar.progress(1.0, text="Done!")
        _display_results(result, url)
    except Exception as e:
        st.error(f"Generation failed: {e}")


def _run_csv(url, uploaded_file, firecrawl_key, bifrost_key, max_urls, generate_full, use_ai, scrape):
    """Execute CSV-based generation and display results."""
    generator = LLMsTextGenerator(
        firecrawl_api_key=firecrawl_key if scrape else None,
        bifrost_api_key=bifrost_key if use_ai else None,
    )

    try:
        file_contents = uploaded_file.getvalue().decode("utf-8-sig")
        csv_entries = parse_screaming_frog_csv(file_contents, max_urls)
    except Exception as e:
        st.error(f"Failed to parse CSV: {e}")
        return

    if not csv_entries:
        st.error("No valid URLs found in the CSV (filtered to 200 status, text/html, indexable).")
        return

    st.info(f"Found **{len(csv_entries)}** valid URLs in CSV.")

    progress_bar = st.progress(0, text="Processing URLs...")

    def on_progress(done, total):
        progress_bar.progress(done / total, text=f"Processing URLs... ({done}/{total})")

    try:
        result = generator.generate_from_csv(
            csv_entries=csv_entries,
            site_url=url,
            scrape=scrape,
            generate_full=generate_full,
            use_ai=use_ai,
            progress_callback=on_progress if (use_ai or scrape) else None,
        )
        progress_bar.progress(1.0, text="Done!")
        _display_results(result, url)
    except Exception as e:
        st.error(f"Generation failed: {e}")


def _display_results(result: Dict, url: str):
    """Render the generated output and download buttons."""
    domain = urlparse(url).netloc.replace("www.", "")

    st.success(
        f"Processed **{result['num_urls_processed']}** of "
        f"**{result['num_urls_total']}** URLs"
    )

    # -- llms.txt ----------------------------------------------------------
    st.subheader("llms.txt")
    st.code(result["llmstxt"], language="markdown")
    st.download_button(
        label="Download llms.txt",
        data=result["llmstxt"],
        file_name=f"{domain}-llms.txt",
        mime="text/plain",
    )

    # -- llms-full.txt -----------------------------------------------------
    if result.get("llms_fulltxt"):
        st.subheader("llms-full.txt")
        with st.expander("Preview llms-full.txt", expanded=False):
            st.code(result["llms_fulltxt"][:5000] + ("\n..." if len(result["llms_fulltxt"]) > 5000 else ""), language="markdown")
        st.download_button(
            label="Download llms-full.txt",
            data=result["llms_fulltxt"],
            file_name=f"{domain}-llms-full.txt",
            mime="text/plain",
        )


if __name__ == "__main__":
    main()
