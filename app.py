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


def _safe_int(val: str) -> int:
    """Parse a string to int, returning 0 on failure."""
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return 0


def _safe_float(val: str) -> float:
    """Parse a string to float, returning 0.0 on failure."""
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def parse_screaming_frog_csv(file_contents: str, max_urls: int = 0) -> List[Dict]:
    """Parse a Screaming Frog 'Internal All' CSV from an in-memory string.

    Extracts all useful columns for importance ranking, deduplication,
    section grouping, and content quality filtering.
    """
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

        # -- Core metadata --
        title = get_field(row, "Title 1", "Title", "title 1", "title")
        description = get_field(
            row, "Meta Description 1", "Meta Description",
            "meta description 1", "meta description", "Description 1",
        )
        h1 = get_field(row, "H1-1", "H1", "h1-1", "h1")

        # -- Content quality --
        word_count = get_field(row, "Word Count", "word count")
        text_ratio = get_field(row, "Text Ratio", "text ratio")

        # -- Link / importance signals --
        crawl_depth = get_field(row, "Crawl Depth", "crawl depth")
        folder_depth = get_field(row, "Folder Depth", "folder depth")
        inlinks = get_field(row, "Inlinks", "inlinks")
        unique_inlinks = get_field(row, "Unique Inlinks", "unique inlinks")
        outlinks = get_field(row, "Outlinks", "outlinks")
        external_outlinks = get_field(row, "External Outlinks", "external outlinks")
        link_score = get_field(row, "Link Score", "link score")

        # -- Deduplication --
        content_hash = get_field(row, "Hash", "hash")
        canonical = get_field(
            row, "Canonical Link Element 1", "canonical link element 1",
        )
        closest_similarity = get_field(
            row, "Closest Similarity Match", "closest similarity match",
        )

        # -- Performance --
        response_time = get_field(row, "Response Time", "response time")

        results.append(
            {
                "url": address,
                "title": title or h1 or "",
                "description": description or "",
                "word_count": _safe_int(word_count),
                "text_ratio": _safe_float(text_ratio.rstrip("%")) if text_ratio else 0.0,
                "crawl_depth": _safe_int(crawl_depth),
                "folder_depth": _safe_int(folder_depth),
                "inlinks": _safe_int(inlinks),
                "unique_inlinks": _safe_int(unique_inlinks),
                "outlinks": _safe_int(outlinks),
                "external_outlinks": _safe_int(external_outlinks),
                "link_score": _safe_int(link_score),
                "hash": content_hash,
                "canonical": canonical,
                "closest_similarity": _safe_float(closest_similarity.rstrip("%")) if closest_similarity else 0.0,
                "response_time": _safe_float(response_time),
            }
        )

        if max_urls and len(results) >= max_urls:
            break

    return results


def deduplicate_entries(entries: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Remove duplicate pages using canonical URLs and content hashes.

    Returns (deduplicated_entries, removed_duplicates).
    """
    seen_canonicals: Dict[str, int] = {}  # canonical_url -> index of first entry
    seen_hashes: Dict[str, int] = {}      # hash -> index of first entry
    kept: List[Dict] = []
    removed: List[Dict] = []

    for entry in entries:
        url = entry["url"]

        # 1. Canonical dedup: if canonical differs from URL, skip this URL
        #    (the canonical version should be the one in the output)
        canonical = entry.get("canonical", "")
        if canonical and canonical != url:
            # Check if we already have the canonical URL
            if canonical in seen_canonicals:
                removed.append(entry)
                continue
            # Use canonical as the key — this URL is the canonical
            seen_canonicals[canonical] = len(kept)
        else:
            seen_canonicals[url] = len(kept)

        # 2. Hash dedup: skip exact duplicate content
        content_hash = entry.get("hash", "")
        if content_hash:
            if content_hash in seen_hashes:
                removed.append(entry)
                continue
            seen_hashes[content_hash] = len(kept)

        kept.append(entry)

    return kept, removed


def filter_thin_content(
    entries: List[Dict], min_word_count: int = 50
) -> Tuple[List[Dict], List[Dict]]:
    """Filter out thin-content pages below the word count threshold.

    Returns (kept, removed).
    """
    kept: List[Dict] = []
    removed: List[Dict] = []
    for entry in entries:
        wc = entry.get("word_count", 0)
        if wc > 0 and wc < min_word_count:
            removed.append(entry)
        else:
            kept.append(entry)
    return kept, removed


def filter_near_duplicates(
    entries: List[Dict], similarity_threshold: float = 90.0
) -> Tuple[List[Dict], List[Dict]]:
    """Filter pages with near-duplicate similarity above threshold.

    Keeps the page with more inlinks when a near-duplicate is detected.
    Returns (kept, removed).
    """
    kept: List[Dict] = []
    removed: List[Dict] = []
    for entry in entries:
        sim = entry.get("closest_similarity", 0.0)
        if sim >= similarity_threshold:
            removed.append(entry)
        else:
            kept.append(entry)
    return kept, removed


# ---------------------------------------------------------------------------
# Section grouping & spec-compliant formatting
# ---------------------------------------------------------------------------

# Pages at crawl depth >= this threshold go into the "## Optional" section
OPTIONAL_DEPTH_THRESHOLD = 4
# Pages with unique inlinks <= this go into Optional (low internal importance)
OPTIONAL_INLINKS_THRESHOLD = 1
# Pages with link score <= this are candidates for Optional (0-100 scale, like PageRank)
OPTIONAL_LINK_SCORE_THRESHOLD = 5

# Output pattern types
PATTERN_CATALOG = "catalog"
PATTERN_WORKFLOW = "workflow"
PATTERN_INDEX_EXPORT = "index_export"

# Semantic section templates per pattern
CATALOG_SECTIONS = [
    "Getting Started", "Core Concepts", "Guides", "API Reference",
    "Integrations", "Resources", "Optional",
]
WORKFLOW_SECTIONS = [
    "Quickstart", "Setup & Configuration", "Features", "Workflows",
    "Troubleshooting", "Reference", "Optional",
]
INDEX_EXPORT_SECTIONS = [
    "Overview", "Documentation", "Tutorials", "API", "Examples", "Optional",
]


def _url_to_section(url: str) -> str:
    """Derive a section name from the first path segment of a URL.

    Used as a fallback when AI grouping is not available.
    """
    path = urlparse(url).path.strip("/")
    if not path:
        return "Main"
    first_segment = path.split("/")[0]
    return first_segment.replace("-", " ").replace("_", " ").title()


def _page_importance_score(r: Dict) -> float:
    """Compute a composite importance score for sorting within sections.

    Higher = more important. Combines Link Score, Unique Inlinks,
    and inverse Crawl Depth.
    """
    link_score = r.get("link_score", 0)
    unique_inlinks = r.get("unique_inlinks", 0)
    crawl_depth = r.get("crawl_depth", 0)
    word_count = r.get("word_count", 0)

    # Normalize each signal to roughly 0-100 range, then weight
    depth_score = max(0, 100 - crawl_depth * 20)  # shallow = high score
    inlinks_score = min(unique_inlinks * 10, 100)
    content_score = min(word_count / 10, 100) if word_count > 0 else 50

    return (link_score * 0.4) + (inlinks_score * 0.3) + (depth_score * 0.2) + (content_score * 0.1)


def _is_optional_page(r: Dict) -> bool:
    """Determine if a page should go in the Optional section."""
    crawl_depth = r.get("crawl_depth", 0)
    unique_inlinks = r.get("unique_inlinks", 0)
    link_score = r.get("link_score", 0)

    is_deep = crawl_depth >= OPTIONAL_DEPTH_THRESHOLD
    is_low_importance = (
        unique_inlinks <= OPTIONAL_INLINKS_THRESHOLD
        or (link_score > 0 and link_score <= OPTIONAL_LINK_SCORE_THRESHOLD)
    )
    return is_deep and is_low_importance


def _group_into_sections_by_url(
    results: List[Dict],
) -> Tuple[Dict[str, List[Dict]], List[Dict]]:
    """Fallback: group by URL path segments."""
    sections: Dict[str, List[Dict]] = {}
    optional: List[Dict] = []

    for r in results:
        if _is_optional_page(r):
            optional.append(r)
        else:
            section = _url_to_section(r["url"])
            sections.setdefault(section, []).append(r)

    # Sort pages within each section by importance
    for section_name in sections:
        sections[section_name].sort(key=_page_importance_score, reverse=True)
    optional.sort(key=_page_importance_score, reverse=True)

    return sections, optional


def _format_spec_llmstxt(
    site_url: str,
    site_name: str,
    site_summary: str,
    sections: Dict[str, Dict],
    optional: List[Dict],
    pattern: str = PATTERN_CATALOG,
) -> str:
    """Build an llms.txt string following the spec with section descriptions.

    sections is a dict of {name: {"description": str, "pages": [...]}}
    """
    lines = [f"# {site_name}\n"]

    if site_summary:
        lines.append(f"> {site_summary}\n")

    # Determine section ordering based on pattern template
    if pattern == PATTERN_WORKFLOW:
        template_order = WORKFLOW_SECTIONS
    elif pattern == PATTERN_INDEX_EXPORT:
        template_order = INDEX_EXPORT_SECTIONS
    else:
        template_order = CATALOG_SECTIONS

    # Order: template sections first (in template order), then remaining alphabetically
    template_names = [s for s in template_order if s in sections and s != "Optional"]
    remaining = sorted(
        [s for s in sections if s not in template_order and s != "Optional"]
    )
    section_order = template_names + remaining

    for section_name in section_order:
        section_data = sections[section_name]
        pages = section_data.get("pages", [])
        description = section_data.get("description", "")

        if not pages:
            continue

        lines.append(f"\n## {section_name}")
        if description:
            lines.append(f"\n{description}\n")
        else:
            lines.append("")

        for r in pages:
            lines.append(f"- [{r['title']}]({r['url']}): {r['description']}")

    if optional:
        lines.append("\n## Optional\n")
        for r in optional:
            lines.append(f"- [{r['title']}]({r['url']}): {r['description']}")

    return "\n".join(lines) + "\n"


def _format_spec_llms_full(
    site_name: str,
    results: List[Dict],
) -> str:
    """Build llms-full.txt with full markdown content per page."""
    lines = [f"# {site_name}\n"]

    for i, r in enumerate(results, 1):
        if r.get("markdown"):
            lines.append(f"\n---\n\n## {r['title']}\n\nSource: {r['url']}\n")
            lines.append(r["markdown"])

    content = "\n".join(lines) + "\n"
    return content


def _validate_llmstxt(content: str, results: List[Dict]) -> List[Dict]:
    """Run validation checks on the generated llms.txt and return issues."""
    issues = []

    # Check file size
    size_kb = len(content.encode("utf-8")) / 1024
    if size_kb > 50:
        issues.append({
            "level": "warning",
            "message": f"File size is {size_kb:.1f} KB — recommended to keep under 50 KB for LLM context efficiency.",
        })

    # Check for required H1
    if not content.startswith("# "):
        issues.append({
            "level": "error",
            "message": "Missing required H1 title at the top of the file.",
        })

    # Check for relative URLs
    for r in results:
        if r["url"].startswith("/") or not r["url"].startswith("http"):
            issues.append({
                "level": "error",
                "message": f"Relative URL found: {r['url']} — all URLs must be absolute.",
            })
            break  # one warning is enough

    # Check for blockquote summary
    if "\n>" not in content and not content.split("\n", 2)[1].startswith(">"):
        issues.append({
            "level": "info",
            "message": "No blockquote summary found. Consider adding a > summary line for better LLM context.",
        })

    # Check for sections
    if "## " not in content:
        issues.append({
            "level": "info",
            "message": "No H2 sections found. Grouping pages under sections improves LLM navigation.",
        })

    return issues


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
            f"Generate a concise title and an action-oriented description for this web page.\n\n"
            f"URL: {url}\n\n"
            f"Rules:\n"
            f"- Title: 3-5 words, descriptive (not generic like 'Documentation' or 'Page')\n"
            f"- Description: 8-12 words, action-oriented, explains what the user can learn or do\n"
            f"- Good example: title='Payment Methods API', description='Learn how to accept and manage different payment methods'\n"
            f"- Bad example: title='Page', description='This page contains information about the topic'\n\n"
            f'Return JSON: {{"title": "Descriptive Title", "description": "Action-oriented description of the page."}}\n\n'
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
            short_title = " ".join(title.split()[:5]) if title else "Page"
            short_desc = (
                " ".join(description.split()[:12])
                if description
                else "No description available"
            )
            return short_title, short_desc

        prompt = (
            f"Rewrite this page's title and description to be concise and action-oriented.\n\n"
            f"URL: {url}\nExisting Title: {title}\nExisting Description: {description}\n\n"
            f"Rules:\n"
            f"- Title: 3-5 words, descriptive (not generic)\n"
            f"- Description: 8-12 words, action-oriented, explains what the user can learn or do\n"
            f"- Good: 'Simulate payments to test your integration'\n"
            f"- Bad: 'Information about testing'\n\n"
            f'Return JSON: {{"title": "Descriptive Title", "description": "Action-oriented description."}}'
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

    # -- Site summary & section grouping via AI --------------------------------

    def generate_site_summary(
        self, site_url: str, page_titles: List[str]
    ) -> Tuple[str, str]:
        """Generate a site name and one-line summary using AI."""
        domain = urlparse(site_url).netloc.replace("www.", "")

        if not self.llm_client:
            return domain, ""

        titles_sample = "\n".join(page_titles[:30])
        prompt = (
            f"Given this website URL and a sample of its page titles, generate:\n"
            f"1. A short site/project name (1-3 words, like a brand name)\n"
            f"2. A one-sentence summary (15-25 words) describing what this site/product does\n\n"
            f"URL: {site_url}\n"
            f"Page titles:\n{titles_sample}\n\n"
            f'Return JSON: {{"name": "Site Name", "summary": "One sentence summary."}}'
        )
        try:
            result = self._call_llm(
                prompt,
                "You generate concise site names and summaries for llms.txt files.",
            )
            return (
                result.get("name", domain),
                result.get("summary", ""),
            )
        except Exception as e:
            logger.error(f"Error generating site summary: {e}")
            return domain, ""

    def generate_semantic_sections(
        self, results: List[Dict], pattern: str = PATTERN_CATALOG
    ) -> Tuple[Dict[str, Dict], List[Dict]]:
        """Use AI to group pages into semantic sections with descriptions.

        Returns ({section_name: {"description": str, "pages": [...]}}, optional_pages)
        """
        # Separate optional pages first
        optional = [r for r in results if _is_optional_page(r)]
        main_pages = [r for r in results if not _is_optional_page(r)]

        if not self.llm_client or not main_pages:
            # Fallback: URL-based grouping wrapped in new format
            url_sections, url_optional = _group_into_sections_by_url(results)
            sections = {
                name: {"description": "", "pages": pages}
                for name, pages in url_sections.items()
            }
            return sections, url_optional

        # Build page summaries for AI
        page_list = []
        for i, r in enumerate(main_pages[:80]):  # Cap at 80 to fit context
            page_list.append(
                f'{i}: "{r["title"]}" - {r["url"]} - {r["description"]}'
            )
        pages_text = "\n".join(page_list)

        if pattern == PATTERN_WORKFLOW:
            template_hint = "Organize around workflows and tasks: Quickstart, Setup & Configuration, Features, Workflows, Troubleshooting, Reference."
        elif pattern == PATTERN_INDEX_EXPORT:
            template_hint = "Organize as a documentation index: Overview, Documentation, Tutorials, API, Examples."
        else:
            template_hint = "Organize as a product catalog: Getting Started, Core Concepts, Guides, API Reference, Integrations, Resources."

        prompt = (
            f"Group these web pages into 3-7 semantic sections for an llms.txt file.\n\n"
            f"Pattern guidance: {template_hint}\n\n"
            f"Pages:\n{pages_text}\n\n"
            f"For each section, provide:\n"
            f"- A clear section name (2-3 words)\n"
            f"- A one-sentence description of what this section covers\n"
            f"- The page indices that belong to it\n\n"
            f"Return JSON: {{"
            f'"sections": ['
            f'{{"name": "Section Name", "description": "One sentence about this section.", "page_indices": [0, 1, 2]}}'
            f"]"
            f"}}"
        )

        try:
            result = self._call_llm(
                prompt,
                "You organize web pages into logical sections for llms.txt files. "
                "Create meaningful groupings that help AI tools navigate the content. "
                "Every page index must appear in exactly one section.",
            )

            ai_sections = result.get("sections", [])
            if not ai_sections:
                raise ValueError("No sections returned")

            # Build section dict
            assigned_indices = set()
            sections: Dict[str, Dict] = {}
            for s in ai_sections:
                name = s.get("name", "Other")
                desc = s.get("description", "")
                indices = s.get("page_indices", [])
                pages = []
                for idx in indices:
                    if 0 <= idx < len(main_pages) and idx not in assigned_indices:
                        pages.append(main_pages[idx])
                        assigned_indices.add(idx)
                if pages:
                    pages.sort(key=_page_importance_score, reverse=True)
                    sections[name] = {"description": desc, "pages": pages}

            # Add any unassigned pages to an "Other" section
            unassigned = [
                main_pages[i]
                for i in range(len(main_pages))
                if i not in assigned_indices
            ]
            if unassigned:
                unassigned.sort(key=_page_importance_score, reverse=True)
                if "Other" in sections:
                    sections["Other"]["pages"].extend(unassigned)
                else:
                    sections["Other"] = {"description": "", "pages": unassigned}

            optional.sort(key=_page_importance_score, reverse=True)
            return sections, optional

        except Exception as e:
            logger.error(f"AI section grouping failed, falling back to URL-based: {e}")
            url_sections, url_optional = _group_into_sections_by_url(results)
            sections = {
                name: {"description": "", "pages": pages}
                for name, pages in url_sections.items()
            }
            return sections, url_optional

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
            "crawl_depth": entry.get("crawl_depth", 0),
            "inlinks": entry.get("inlinks", 0),
            "unique_inlinks": entry.get("unique_inlinks", 0),
        }

    # -- Main generation methods --------------------------------------------

    def generate_from_firecrawl(
        self,
        url: str,
        max_urls: int = 100,
        generate_full: bool = True,
        pattern: str = PATTERN_CATALOG,
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
            pattern=pattern,
            progress_callback=progress_callback,
        )

    def generate_from_csv(
        self,
        csv_entries: List[Dict],
        site_url: str,
        scrape: bool = False,
        generate_full: bool = True,
        use_ai: bool = True,
        pattern: str = PATTERN_CATALOG,
        progress_callback=None,
    ) -> Dict[str, str]:
        if not csv_entries:
            raise ValueError("No valid URLs found in CSV data")

        if not use_ai and not scrape:
            return self._build_from_metadata(
                site_url, csv_entries, generate_full, pattern
            )

        return self._process_urls(
            site_url=site_url,
            items=[(entry, i) for i, entry in enumerate(csv_entries)],
            processor=lambda item: self.process_url_csv(item[0], item[1], scrape),
            generate_full=generate_full,
            pattern=pattern,
            progress_callback=progress_callback,
        )

    # -- Internal helpers ---------------------------------------------------

    def _process_urls(
        self, site_url, items, processor, generate_full,
        pattern=PATTERN_CATALOG, progress_callback=None,
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

        # Generate site name & summary via AI
        page_titles = [r["title"] for r in all_results if r.get("title")]
        site_name, site_summary = self.generate_site_summary(site_url, page_titles)

        return self._format_output(
            site_url, site_name, site_summary, all_results, generate_full,
            total, pattern,
        )

    def _build_from_metadata(
        self, site_url: str, csv_entries: List[Dict], generate_full: bool,
        pattern: str = PATTERN_CATALOG,
    ) -> Dict[str, str]:
        """Build output directly from CSV metadata — no API calls."""
        all_results = []
        for i, entry in enumerate(csv_entries):
            title = entry.get("title", "") or "Page"
            title_words = title.split()
            if len(title_words) > 6:
                title = " ".join(title_words[:6])

            description = entry.get("description", "") or "No description available"
            desc_words = description.split()
            if len(desc_words) > 15:
                description = " ".join(desc_words[:12])

            all_results.append(
                {
                    "url": entry["url"],
                    "title": title,
                    "description": description,
                    "markdown": "",
                    "index": i,
                    "crawl_depth": entry.get("crawl_depth", 0),
                    "inlinks": entry.get("inlinks", 0),
                    "unique_inlinks": entry.get("unique_inlinks", 0),
                    "link_score": entry.get("link_score", 0),
                    "word_count": entry.get("word_count", 0),
                }
            )

        domain = urlparse(site_url).netloc.replace("www.", "")
        return self._format_output(
            site_url, domain, "", all_results, generate_full, len(csv_entries),
            pattern,
        )

    def _format_output(
        self,
        site_url: str,
        site_name: str,
        site_summary: str,
        results: List[Dict],
        generate_full: bool,
        total: int,
        pattern: str = PATTERN_CATALOG,
    ) -> Dict[str, str]:
        """Produce spec-compliant llms.txt with semantic sections."""
        # Use AI grouping when available, fallback to URL-based
        sections, optional = self.generate_semantic_sections(results, pattern)

        llmstxt = _format_spec_llmstxt(
            site_url, site_name, site_summary, sections, optional, pattern
        )

        llms_fulltxt = ""
        if generate_full:
            llms_fulltxt = _format_spec_llms_full(site_name, results)
            if not any(r.get("markdown") for r in results):
                llms_fulltxt = ""

        validation = _validate_llmstxt(llmstxt, results)

        return {
            "llmstxt": llmstxt,
            "llms_fulltxt": llms_fulltxt,
            "num_urls_processed": len(results),
            "num_urls_total": total,
            "validation": validation,
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
        st.header("General Settings")
        st.caption("Applies to both Firecrawl and Screaming Frog CSV modes")

        pattern = st.selectbox(
            "Output pattern",
            options=[
                ("Catalog (multi-product, API-centric)", PATTERN_CATALOG),
                ("Workflow (dev tools, IDE integrations)", PATTERN_WORKFLOW),
                ("Index + Export (dense docs, AI-native)", PATTERN_INDEX_EXPORT),
            ],
            format_func=lambda x: x[0],
            index=0,
            help=(
                "**Catalog**: Groups by product area — Getting Started, Core Concepts, "
                "Guides, API Reference. Best for multi-product platforms (Stripe, Cloudflare).\n\n"
                "**Workflow**: Groups by developer tasks — Quickstart, Setup, Features, "
                "Troubleshooting. Best for dev tools (Cursor, Windsurf).\n\n"
                "**Index + Export**: Groups as doc index — Overview, Documentation, "
                "Tutorials, API, Examples. Best for dense docs (Anthropic, LangGraph)."
            ),
        )[1]

        max_urls = st.number_input(
            "Max URLs to process",
            min_value=1,
            max_value=500,
            value=20,
            step=5,
        )
        generate_full = st.checkbox("Generate llms-full.txt", value=True)

        st.divider()
        st.header("Screaming Frog Filters")
        st.caption("Only applies when using the Screaming Frog CSV tab")

        dedup_enabled = st.checkbox(
            "Remove duplicates",
            value=True,
            help="Removes pages with duplicate canonical URLs or identical content hashes.",
        )
        near_dupes_enabled = st.checkbox(
            "Remove near-duplicates",
            value=False,
            help="Removes pages above the similarity threshold. Requires 'Enable Near Duplicates' in Screaming Frog config.",
        )
        near_dupe_threshold = 90.0
        if near_dupes_enabled:
            near_dupe_threshold = st.slider(
                "Similarity threshold %",
                min_value=50,
                max_value=100,
                value=90,
                step=5,
                help="Pages with similarity at or above this value are removed.",
            )

        thin_content_enabled = st.checkbox(
            "Remove thin content",
            value=False,
            help="Removes pages below a minimum word count.",
        )
        min_word_count = 50
        if thin_content_enabled:
            min_word_count = st.number_input(
                "Min word count",
                min_value=10,
                max_value=500,
                value=50,
                step=10,
            )

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
                _run_firecrawl(fc_url, firecrawl_key, bifrost_key, max_urls, generate_full, pattern)

    # -- CSV tab -----------------------------------------------------------
    with tab_csv:
        st.markdown(
            "Upload a Screaming Frog **Internal All** CSV export. "
            "The tool uses **20+ columns** including link metrics, content hashes, "
            "and canonical URLs for deduplication, importance ranking, and automatic "
            "`## Optional` section detection. See the sidebar for content filters."
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
                    pattern=pattern,
                    dedup_enabled=dedup_enabled,
                    near_dupes_enabled=near_dupes_enabled,
                    near_dupe_threshold=float(near_dupe_threshold),
                    thin_content_enabled=thin_content_enabled,
                    min_word_count=min_word_count,
                )


# ---------------------------------------------------------------------------
# Runner functions
# ---------------------------------------------------------------------------


def _run_firecrawl(url, firecrawl_key, bifrost_key, max_urls, generate_full, pattern=PATTERN_CATALOG):
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
            url, max_urls, generate_full, pattern=pattern, progress_callback=on_progress
        )
        progress_bar.progress(1.0, text="Done!")
        _display_results(result, url)
    except Exception as e:
        st.error(f"Generation failed: {e}")


def _run_csv(
    url, uploaded_file, firecrawl_key, bifrost_key, max_urls, generate_full,
    use_ai, scrape, pattern=PATTERN_CATALOG, dedup_enabled=True,
    near_dupes_enabled=False, near_dupe_threshold=90.0,
    thin_content_enabled=False, min_word_count=50,
):
    """Execute CSV-based generation with filtering and display results."""
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

    total_parsed = len(csv_entries)

    # -- Apply content filters ---------------------------------------------
    filter_log = []

    if dedup_enabled:
        csv_entries, dupes = deduplicate_entries(csv_entries)
        if dupes:
            filter_log.append(f"Removed **{len(dupes)}** duplicate pages (canonical/hash)")

    if near_dupes_enabled:
        csv_entries, near_dupes = filter_near_duplicates(csv_entries, near_dupe_threshold)
        if near_dupes:
            filter_log.append(
                f"Removed **{len(near_dupes)}** near-duplicate pages "
                f"(>={near_dupe_threshold:.0f}% similarity)"
            )

    if thin_content_enabled:
        csv_entries, thin = filter_thin_content(csv_entries, min_word_count)
        if thin:
            filter_log.append(
                f"Removed **{len(thin)}** thin-content pages (<{min_word_count} words)"
            )

    # -- Show filter summary -----------------------------------------------
    if filter_log:
        with st.expander("Content Filtering", expanded=True):
            st.markdown(f"**{total_parsed}** pages parsed from CSV")
            for msg in filter_log:
                st.markdown(f"- {msg}")
            st.markdown(f"**{len(csv_entries)}** pages remaining after filters")
    else:
        st.info(f"Found **{total_parsed}** valid URLs in CSV.")

    if not csv_entries:
        st.error("No pages remaining after filtering.")
        return

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
            pattern=pattern,
            progress_callback=on_progress if (use_ai or scrape) else None,
        )
        progress_bar.progress(1.0, text="Done!")
        _display_results(result, url)
    except Exception as e:
        st.error(f"Generation failed: {e}")


def _display_results(result: Dict, url: str):
    """Render the generated output, validation, and download buttons."""
    domain = urlparse(url).netloc.replace("www.", "")

    st.success(
        f"Processed **{result['num_urls_processed']}** of "
        f"**{result['num_urls_total']}** URLs"
    )

    # -- Validation panel --------------------------------------------------
    validation = result.get("validation", [])
    if validation:
        with st.expander("Validation Results", expanded=True):
            for issue in validation:
                level = issue["level"]
                msg = issue["message"]
                if level == "error":
                    st.error(msg)
                elif level == "warning":
                    st.warning(msg)
                else:
                    st.info(msg)
    else:
        st.info("All validation checks passed.")

    # -- File size stats ---------------------------------------------------
    size_kb = len(result["llmstxt"].encode("utf-8")) / 1024
    full_size_kb = (
        len(result["llms_fulltxt"].encode("utf-8")) / 1024
        if result.get("llms_fulltxt")
        else 0
    )
    cols = st.columns(3)
    cols[0].metric("llms.txt size", f"{size_kb:.1f} KB")
    if full_size_kb:
        cols[1].metric("llms-full.txt size", f"{full_size_kb:.1f} KB")
    cols[2].metric("Pages included", result["num_urls_processed"])

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
            preview = result["llms_fulltxt"][:5000]
            if len(result["llms_fulltxt"]) > 5000:
                preview += "\n..."
            st.code(preview, language="markdown")
        st.download_button(
            label="Download llms-full.txt",
            data=result["llms_fulltxt"],
            file_name=f"{domain}-llms-full.txt",
            mime="text/plain",
        )


if __name__ == "__main__":
    main()
