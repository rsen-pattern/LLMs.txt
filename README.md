# LLMs.txt Generator

A **Streamlit** web app that generates `llms.txt` and `llms-full.txt` files for any website using:

- **Firecrawl API** for automatic URL discovery and content scraping
- **Screaming Frog CSV** import ("Internal All" export) for URL discovery
- **[Patterns Bifrost](https://bifrost.pattern.com)** API for AI-generated summaries (OpenAI-compatible gateway)

## What is llms.txt?

`llms.txt` is a standardized format for making website content more accessible to Large Language Models (LLMs):

- **llms.txt** — A concise index of pages with titles and descriptions
- **llms-full.txt** — Complete markdown content of all pages

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

### How to Export from Screaming Frog

1. Open Screaming Frog SEO Spider
2. Enter your website URL and click **Start**
3. Wait for the crawl to complete
4. Go to **File > Export > Internal > All**
5. Save as CSV and upload to the app

### Expected CSV Columns

The parser reads these columns (case-insensitive):

| Column | Used For |
|--------|----------|
| `Address` | The page URL |
| `Status Code` | Filters to 200 responses |
| `Content Type` | Filters to `text/html` |
| `Indexability` | Skips non-indexable pages |
| `Title 1` | Page title |
| `Meta Description 1` | Page description |
| `H1-1` | Fallback title |

## Output Format

### llms.txt

```
# https://example.com llms.txt

- [Page Title](https://example.com/page): Brief description of the page content
```

### llms-full.txt

```
# https://example.com llms-full.txt

<|page-1-llmstxt|>
## Page Title
Full markdown content of the page...
```

## CLI Usage

The original CLI script is also available:

```bash
# Firecrawl mode
python generate-llmstxt.py https://example.com

# Screaming Frog CSV mode
python generate-llmstxt.py https://example.com --csv internal_all.csv

# CSV without AI (no API keys needed)
python generate-llmstxt.py https://example.com --csv internal_all.csv --no-ai
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
