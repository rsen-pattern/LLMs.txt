# LLMs.txt Generator

A **Streamlit** web app that generates `llms.txt` and `llms-full.txt` files for any website using:

- **Firecrawl API** for automatic URL discovery and content scraping
- **Screaming Frog CSV** import ("Internal All" export) for URL discovery
- **[Patterns Bifrost](https://bifrost.pattern.com)** API for AI-generated summaries (OpenAI-compatible gateway)

## What is llms.txt?

`llms.txt` is a [standardized format](https://llmstxt.org/) proposed by Jeremy Howard for making website content accessible to LLMs. This tool generates **spec-compliant** output:

```markdown
# Site Name

> One-sentence summary of what this site/product does.

## Docs

- [Quick Start](https://example.com/docs/quickstart): Getting started guide for new users

## Blog

- [Release Notes](https://example.com/blog/releases): Latest product updates and changelogs

## Optional

- [Careers](https://example.com/careers): Open roles at the company
```

- **`llms.txt`** — Curated index with H1 title, blockquote summary, H2 sections, and `## Optional` for lower-priority pages
- **`llms-full.txt`** — Complete markdown content of all pages

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the app

```bash
streamlit run app.py
```

### 3. Open in browser

The app will open at `http://localhost:8501`. Enter your API keys in the sidebar and start generating.

## Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository and select `app.py` as the main file
4. Add your API keys as **Secrets** in the Streamlit Cloud dashboard:
   ```toml
   BIFROST_API_KEY = "sk-bf-..."
   FIRECRAWL_API_KEY = "fc-..."
   ```

## API Keys

| Key | Required For | Get One |
|-----|-------------|---------|
| **Patterns Bifrost** | AI-generated titles/descriptions | Your Bifrost dashboard |
| **Firecrawl** | Auto-crawl mode or CSV + scrape mode | [firecrawl.dev](https://firecrawl.dev) |

Keys are entered in the app sidebar. No keys are stored — they live only in your browser session.

## Usage Modes

### Mode 1: Firecrawl (Auto-Crawl)

Enter a URL and the app will:
1. Discover all pages via Firecrawl's `/map` endpoint
2. Scrape each page for markdown content
3. Generate AI titles/descriptions via Bifrost
4. Output `llms.txt` and `llms-full.txt`

**Requires:** Bifrost key + Firecrawl key

### Mode 2: Screaming Frog CSV

Upload a Screaming Frog "Internal All" CSV export:

1. **CSV only (no AI)** — Uses existing `Title 1` and `Meta Description 1` columns directly. No API keys needed.
2. **CSV + AI** — Refines titles/descriptions via Bifrost. Requires Bifrost key.
3. **CSV + Scrape** — Uses CSV for URL list, Firecrawl for full content. Requires both keys.

### Screaming Frog Setup Guide

Follow these steps to get the best results from your Screaming Frog export.

#### Step 1: Configure the Spider

Before crawling, adjust these settings in **Configuration > Spider**:

- **Crawl tab**: Ensure "Crawl Internal Links" is checked, "Crawl External Links" is unchecked
- **Limits tab**: Set a **Crawl Depth** limit (e.g. 5-8) to avoid crawling endlessly deep pages
- **Rendering tab**: Set to **JavaScript Rendering** if your site is an SPA or uses client-side rendering (otherwise "Text Only" is fine)

#### Step 2: Enable Required Columns

Some columns need to be explicitly enabled:

| Setting Location | What to Enable |
|-----------------|----------------|
| **Configuration > Spider > Crawl** | Check "Crawl Internal Links" |
| **Configuration > Content > Duplicates** | Check "Enable Near Duplicates" (required for similarity detection) |
| **Configuration > Content > Area** | Check "Hash" (enables content hash for exact dedup) |

**Link Score** is calculated automatically under the **Internal** tab — no extra config needed.

#### Step 3: Crawl the Website

1. Enter your full website URL (e.g. `https://example.com`) in the URL bar
2. Click **Start**
3. Wait for the crawl to finish (the progress bar at the bottom shows status)

#### Step 4: Export the CSV

1. Click on the **Internal** tab at the top
2. Select **Filter: HTML** from the dropdown (this pre-filters to HTML pages)
3. Go to **File > Export** (or click the Export button)
4. Save as **CSV**
5. Upload this file to the app

#### Recommended Crawl Settings by Site Size

| Site Size | Max URLs | Crawl Depth | Rendering | Notes |
|-----------|----------|-------------|-----------|-------|
| Small (<100 pages) | Unlimited | No limit | Text Only | Full crawl recommended |
| Medium (100-1,000) | 500 | 6 | Text Only | Use filters to stay under 500 |
| Large (1,000-10,000) | 1,000 | 5 | Text Only | Focus on key sections |
| Very Large (10,000+) | 2,000 | 4 | Text Only | Consider crawling subfolders separately |

### CSV Columns Used

The tool reads **20+ columns** from the Screaming Frog export (all case-insensitive):

#### Core Metadata (always used)

| Column | Purpose |
|--------|---------|
| `Address` | The page URL |
| `Status Code` | Filters to 200 OK responses only |
| `Content Type` | Filters to `text/html` pages only |
| `Indexability` | Skips non-indexable pages |
| `Title 1` | Page title — used as link text in llms.txt |
| `Meta Description 1` | Page description — used as link description |
| `H1-1` | Fallback title if Title 1 is empty |

#### Importance & Ranking Signals

| Column | Purpose |
|--------|---------|
| `Crawl Depth` | Pages at depth 4+ with low importance go to `## Optional` |
| `Folder Depth` | URL path depth for section grouping |
| `Inlinks` | Total internal links pointing to the page |
| `Unique Inlinks` | Unique pages linking to this page — primary importance metric |
| `Outlinks` | Outgoing internal links (identifies hub pages) |
| `External Outlinks` | External links (identifies resource/reference pages) |
| `Link Score` | Screaming Frog's 0-100 PageRank-like metric — scores <=5 at depth 4+ go to `## Optional` |

#### Deduplication

| Column | Purpose |
|--------|---------|
| `Canonical Link Element 1` | Uses canonical URL to skip duplicate versions of the same page |
| `Hash` | MD5 content hash — removes exact duplicate pages |
| `Closest Similarity Match` | Percentage similarity — filters near-duplicates above threshold (default 90%) |

#### Content Quality

| Column | Purpose |
|--------|---------|
| `Word Count` | Filters thin-content pages below minimum word count |
| `Text Ratio` | Text-to-HTML ratio (extracted for reference) |
| `Response Time` | Page load time (extracted for reference) |

### Content Filters

Filters are available in the Streamlit sidebar and as CLI flags:

| Filter | Streamlit | CLI Flag | Default |
|--------|-----------|----------|---------|
| **Deduplication** (canonical + hash) | "Remove duplicates" checkbox | `--no-dedup` to disable | Enabled |
| **Near-duplicate removal** | "Remove near-duplicates" checkbox + slider | `--filter-near-dupes 90` | Disabled |
| **Thin content removal** | "Remove thin content" checkbox + word count input | `--filter-thin 50` | Disabled |

#### How Deduplication Works

1. **Canonical dedup**: If a page's `Canonical Link Element 1` differs from its URL, the page is skipped (the canonical version is kept)
2. **Hash dedup**: If two pages share the same content `Hash`, only the first is kept
3. **Near-duplicate dedup**: Pages with `Closest Similarity Match` above the threshold are removed

### Output Patterns

Choose from three output patterns inspired by real-world llms.txt files:

| Pattern | Inspired By | Best For | Sections |
|---------|------------|----------|----------|
| **Catalog** (default) | Stripe, Cloudflare | Large docs sites with diverse content | Getting Started, Core Concepts, Guides, API Reference, Integrations, Resources |
| **Workflow** | Cursor, Windsurf | Developer tools with setup-oriented docs | Quickstart, Setup & Configuration, Features, Workflows, Troubleshooting, Reference |
| **Index + Export** | Anthropic, LangGraph | Sites with tutorials and examples | Overview, Documentation, Tutorials, API, Examples |

Select the pattern in the Streamlit sidebar under **General Settings**, or use `--pattern` in the CLI.

### AI Semantic Section Grouping (Streamlit)

When using the Streamlit app with a Bifrost API key, pages are grouped into meaningful sections using AI — producing categories like "Getting Started" and "Core Concepts" instead of mechanical URL-path groups. Each section gets a brief description displayed under the H2 heading.

### URL-Based Section Grouping (Fallback / CLI)

Pages are automatically grouped under **H2 sections** based on their URL path:

- `https://example.com/docs/setup` -> `## Docs`
- `https://example.com/blog/post` -> `## Blog`
- `https://example.com/pricing` -> `## Pricing`
- `https://example.com/` -> `## Main`

Pages within each section are sorted by importance (composite of Link Score, Unique Inlinks, Crawl Depth, and Word Count).

### Optional Section Detection

Pages are placed in `## Optional` (per the llms.txt spec) if **both** conditions are met:

1. **Crawl Depth >= 4** (deep in the site hierarchy)
2. **Low importance**: Unique Inlinks <= 1 OR Link Score <= 5

This ensures secondary/deep content can be skipped when shorter LLM context is needed.

### Validation

The tool validates your generated llms.txt and warns about:
- File size over 50 KB (recommended max for LLM context efficiency)
- Missing H1 title (required by spec)
- Relative URLs (must be absolute)
- Missing blockquote summary
- Missing H2 sections

## Output Format

### llms.txt (spec-compliant)

```markdown
# Site Name

> One-sentence summary of the site.

## Docs

- [Quick Start](https://example.com/docs/quickstart): Getting started guide for new users

## Blog

- [Release Notes](https://example.com/blog/releases): Latest updates and changelogs

## Optional

- [Deep Nested Page](https://example.com/docs/advanced/internals/api): Low-traffic internal API docs
```

### llms-full.txt

```markdown
# Site Name

---

## Quick Start

Source: https://example.com/docs/quickstart

Full markdown content of the page...
```

## CLI Usage

The CLI script supports all the same features:

```bash
# Firecrawl mode (auto-crawl)
python generate-llmstxt.py https://example.com

# Screaming Frog CSV mode
python generate-llmstxt.py https://example.com --csv internal_all.csv

# CSV without AI (no API keys needed)
python generate-llmstxt.py https://example.com --csv internal_all.csv --no-ai

# CSV with content filters
python generate-llmstxt.py https://example.com --csv internal_all.csv \
  --filter-thin 50 \
  --filter-near-dupes 90

# Disable deduplication
python generate-llmstxt.py https://example.com --csv internal_all.csv --no-dedup

# Use a specific output pattern
python generate-llmstxt.py https://example.com --csv internal_all.csv --pattern workflow

# Full example with all options
python generate-llmstxt.py https://docs.example.com \
  --csv internal_all.csv \
  --scrape \
  --max-urls 100 \
  --pattern catalog \
  --filter-thin 50 \
  --output-dir ./output \
  --verbose
```

Run `python generate-llmstxt.py --help` for all options.

## Project Structure

```
├── app.py                  # Streamlit web app (main entry point)
├── generate-llmstxt.py     # CLI script
├── requirements.txt        # Python dependencies
├── .env.example            # API key template
├── .streamlit/
│   └── config.toml         # Streamlit theme config
└── README.md
```
