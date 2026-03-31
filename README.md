# LLMs.txt Generator

A Python script that generates `llms.txt` and `llms-full.txt` files for any website using **Firecrawl API** and/or **Screaming Frog CSV exports**, with **OpenAI** for AI-generated summaries.

## What is llms.txt?

`llms.txt` is a standardized format for making website content more accessible to Large Language Models (LLMs). It provides:

- **llms.txt**: A concise index of all pages with titles and descriptions
- **llms-full.txt**: Complete content of all pages for comprehensive access

## Features

- **Dual Input Modes**:
  - **Firecrawl API**: Automatically discovers and scrapes website URLs
  - **Screaming Frog CSV**: Import an "Internal All" CSV export for URL discovery
- **AI Summaries**: Uses OpenAI GPT-4o-mini to generate concise titles and descriptions
- **CSV Metadata Fallback**: Use existing titles and meta descriptions from Screaming Frog without any API calls (`--no-ai`)
- **Hybrid Mode**: Combine Screaming Frog URLs with Firecrawl scraping (`--csv` + `--scrape`)
- **Parallel Processing**: Processes multiple URLs concurrently for faster generation
- **Configurable Limits**: Set maximum number of URLs to process
- **Flexible Output**: Choose to generate both files or just `llms.txt`

## Prerequisites

- Python 3.7+
- Firecrawl API key ([Get one here](https://firecrawl.dev)) — required for Firecrawl mode or `--scrape`
- OpenAI API key ([Get one here](https://platform.openai.com)) — optional with `--no-ai` flag

## Installation

1. Clone the repository:

```bash
git clone https://github.com/rsen-pattern/llms.txt.git
cd llms.txt
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up API keys (choose one method):

**Option A: Using .env file (recommended)**

```bash
cp .env.example .env
# Edit .env and add your API keys
```

**Option B: Using environment variables**

```bash
export FIRECRAWL_API_KEY="your-firecrawl-api-key"
export OPENAI_API_KEY="your-openai-api-key"
```

**Option C: Using command line arguments** (see usage examples below)

## Usage

### Mode 1: Firecrawl API (automatic discovery)

Generate `llms.txt` and `llms-full.txt` by automatically crawling a website:

```bash
python generate-llmstxt.py https://example.com
```

### Mode 2: Screaming Frog CSV Import

Use a Screaming Frog "Internal All" CSV export for URL discovery:

```bash
# With AI-generated titles/descriptions
python generate-llmstxt.py https://example.com --csv internal_all.csv

# Without AI — uses existing CSV metadata (no API keys needed)
python generate-llmstxt.py https://example.com --csv internal_all.csv --no-ai

# Hybrid: CSV for URLs + Firecrawl for full content scraping
python generate-llmstxt.py https://example.com --csv internal_all.csv --scrape
```

### Additional Options

```bash
# Limit to 50 URLs
python generate-llmstxt.py https://example.com --max-urls 50

# Save to specific directory
python generate-llmstxt.py https://example.com --output-dir ./output

# Only generate llms.txt (skip full text)
python generate-llmstxt.py https://example.com --no-full-text

# Enable verbose logging
python generate-llmstxt.py https://example.com --verbose

# Specify API keys via command line
python generate-llmstxt.py https://example.com \
  --firecrawl-api-key "fc-..." \
  --openai-api-key "sk-..."
```

## Command Line Options

| Option | Description |
|---|---|
| `url` (required) | The website URL to process |
| `--csv` | Path to Screaming Frog "Internal All" CSV export |
| `--scrape` | When using `--csv`, also scrape URLs via Firecrawl for full markdown content |
| `--no-ai` | When using `--csv`, skip AI generation and use CSV metadata directly |
| `--max-urls` | Maximum number of URLs to process (default: 20, 0 for unlimited) |
| `--output-dir` | Directory to save output files (default: current directory) |
| `--firecrawl-api-key` | Firecrawl API key (defaults to env var) |
| `--openai-api-key` | OpenAI API key (defaults to env var) |
| `--no-full-text` | Only generate `llms.txt`, skip `llms-full.txt` |
| `--verbose` | Enable verbose logging |

## Screaming Frog CSV Format

The script expects a Screaming Frog "Internal All" CSV export. It reads these columns (case-insensitive):

| Column | Used For |
|---|---|
| Address | The page URL |
| Status Code | Filters to 200 responses only |
| Content Type | Filters to `text/html` pages only |
| Indexability | Skips non-indexable pages |
| Title 1 | Page title (used with `--no-ai`) |
| Meta Description 1 | Page description (used with `--no-ai`) |
| H1-1 | Fallback title if Title 1 is empty |

### How to Export from Screaming Frog

1. Open Screaming Frog SEO Spider
2. Enter your website URL and click "Start"
3. Wait for the crawl to complete
4. Go to **File > Export > Internal > All** (or use the "Internal" tab)
5. Save as CSV
6. Use the exported CSV with `--csv path/to/export.csv`

## Output Format

### llms.txt

```
# https://example.com llms.txt

- [Page Title](https://example.com/page1): Brief description of the page content here
- [Another Page](https://example.com/page2): Another concise description of page content
```

### llms-full.txt

```
# https://example.com llms-full.txt

<|page-1-llmstxt|>
## Page Title
Full markdown content of the page...

<|page-2-llmstxt|>
## Another Page
Full markdown content of another page...
```

## How It Works

### Firecrawl Mode (default)

1. **Website Mapping**: Uses Firecrawl's `/map` endpoint to discover all URLs
2. **Batch Processing**: Processes URLs in batches of 10 concurrently
3. **Content Extraction**: Scrapes each URL to extract markdown content
4. **AI Summarization**: GPT-4o-mini generates a 3-4 word title and 9-10 word description
5. **File Generation**: Creates formatted `llms.txt` and `llms-full.txt`

### Screaming Frog CSV Mode (`--csv`)

1. **CSV Parsing**: Reads the Screaming Frog export, filtering to indexable HTML pages with 200 status
2. **Metadata Extraction**: Pulls titles and descriptions from CSV columns
3. **Optional AI Refinement**: Uses OpenAI to generate concise summaries (skip with `--no-ai`)
4. **Optional Scraping**: Uses Firecrawl to get full page content (enable with `--scrape`)
5. **File Generation**: Creates `llms.txt` (and `llms-full.txt` if content was scraped)

## Configuration Priority

API keys are checked in this order:

1. Command line arguments (`--firecrawl-api-key`, `--openai-api-key`)
2. `.env` file in the current directory
3. Environment variables (`FIRECRAWL_API_KEY`, `OPENAI_API_KEY`)

## Examples

### Quick Index from Screaming Frog (No API Keys Needed)

```bash
python generate-llmstxt.py https://example.com --csv internal_all.csv --no-ai --no-full-text
```

### Full Generation with Screaming Frog + Firecrawl

```bash
python generate-llmstxt.py https://docs.example.com \
  --csv internal_all.csv \
  --scrape \
  --max-urls 100 \
  --verbose
```

### Firecrawl-Only for a Small Site

```bash
python generate-llmstxt.py https://small-blog.com --max-urls 20
```
