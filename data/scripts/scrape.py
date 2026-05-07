"""
Scraper: fetches documents from startup homepages, YC, newsletters, news feeds,
and technical blogs. Normalizes each to the corpus schema and appends to sharded
JSON files under data/corpus/<source>.json.

Output schema for every emitted record:
{
    "title":        str,        # page or post title
    "description":  str,        # extracted main text, truncated to MAX_DESC_CHARS
    "source_url":   str,        # canonical URL of the fetched page
    "tags":         list[str],  # subset of TAG_TAXONOMY assigned by source function
    "publish_date": str         # ISO 8601 date string "YYYY-MM-DD"; "" if unknown
}

Usage:
    python data/scripts/scrape.py               # runs all sources
    python data/scripts/scrape.py --source yc   # runs only YC scraper
    python data/scripts/scrape.py --dry-run     # count only, no writes

Environment variables required: none (no LLM calls).
"""

import argparse
import hashlib
import html.parser
import json
import logging
import pathlib
import time
import urllib.parse
import urllib.robotparser
from datetime import datetime, timezone
from typing import Optional

import feedparser
import httpx
import trafilatura

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Path constants ─────────────────────────────────────────────────────────────

CORPUS_DIR: pathlib.Path = pathlib.Path(__file__).parent.parent / "corpus"
RAW_DIR: pathlib.Path = pathlib.Path(__file__).parent.parent / "raw"

# ── Scraping constants ─────────────────────────────────────────────────────────

REQUEST_DELAY_SECONDS: float = 1.0
MAX_DESC_CHARS: int = 1200
HTTP_TIMEOUT_SECONDS: float = 15.0
# Browser-like UA to avoid bot-detection 403s on well-known sites
USER_AGENT: str = (
    "Mozilla/5.0 (compatible; MarketAnalystBot/1.0; +https://github.com/research)"
)

TAG_TAXONOMY: list[str] = [
    "multi-agent-orchestration",
    "rag",
    "observability",
    "agent-runtime",
    "voice-agents",
    "coding-agents",
    "vertical-agents",
    "eval-tools",
    "memory-tools",
    "infra",
]

# Each tuple: (display_name, homepage_url, tags)
# tags must be subset of TAG_TAXONOMY
STARTUP_TARGETS: list[tuple[str, str, list[str]]] = [
    ("CrewAI",      "https://crewai.com",                          ["multi-agent-orchestration"]),
    ("AutoGen",     "https://microsoft.github.io/autogen",         ["multi-agent-orchestration"]),
    ("LangGraph",   "https://langchain-ai.github.io/langgraph",    ["multi-agent-orchestration", "agent-runtime"]),
    ("LlamaIndex",  "https://www.llamaindex.ai",                   ["rag"]),
    ("Cognition",   "https://cognition.ai",                        ["coding-agents"]),
    ("Adept",       "https://adept.ai",                            ["agent-runtime"]),
    ("Sierra",      "https://sierra.ai",                           ["vertical-agents", "voice-agents"]),
    ("Lindy",       "https://lindy.ai",                            ["multi-agent-orchestration"]),
    ("Decagon",     "https://decagon.ai",                          ["vertical-agents"]),
    ("Cresta",      "https://cresta.com",                          ["vertical-agents", "voice-agents"]),
    ("Imbue",       "https://imbue.com",                           ["agent-runtime"]),
    ("MultiOn",     "https://multion.ai",                          ["agent-runtime"]),
    ("Fixie",       "https://fixie.ai",                            ["agent-runtime"]),
    ("Reworkd",     "https://reworkd.ai",                          ["multi-agent-orchestration"]),
    ("Vellum",      "https://vellum.ai",                           ["eval-tools", "infra"]),
    ("Humanloop",   "https://humanloop.com",                       ["eval-tools", "observability"]),
    ("Langfuse",    "https://langfuse.com",                        ["observability", "eval-tools"]),
    ("Braintrust",  "https://braintrustdata.com",                  ["eval-tools"]),
    ("Composio",    "https://composio.dev",                        ["agent-runtime", "infra"]),
    ("Mem0",        "https://mem0.ai",                             ["memory-tools"]),
    ("Zep",         "https://getzep.com",                          ["memory-tools"]),
    ("Weaviate",    "https://weaviate.io",                         ["rag", "infra"]),
    ("Chroma",      "https://trychroma.com",                       ["rag"]),
    ("Qdrant",      "https://qdrant.tech",                         ["rag", "infra"]),
    ("Exa",         "https://exa.ai",                              ["rag"]),
]

# Each tuple: (display_name, rss_url, tags)
NEWSLETTER_RSS_FEEDS: list[tuple[str, str, list[str]]] = [
    ("Latent Space",  "https://www.latent.space/feed",              ["multi-agent-orchestration", "infra"]),
    ("Import AI",     "https://jack-clark.net/feed/",               ["infra"]),
    ("Ben's Bites",   "https://www.bensbites.co/feed",              ["multi-agent-orchestration"]),
    ("The Batch",     "https://www.deeplearning.ai/the-batch/feed", ["infra"]),
]

NEWS_RSS_FEEDS: list[tuple[str, str, list[str]]] = [
    ("TechCrunch AI",  "https://techcrunch.com/category/artificial-intelligence/feed/", ["infra"]),
    ("VentureBeat AI", "https://venturebeat.com/category/ai/feed/",                     ["infra"]),
]

# Each tuple: (display_name, blog_index_url, post_path_prefix, tags)
TECH_BLOG_TARGETS: list[tuple[str, str, str, list[str]]] = [
    ("Anthropic Blog",   "https://www.anthropic.com/blog",     "/blog/",      ["agent-runtime", "eval-tools"]),
    ("OpenAI Blog",      "https://openai.com/blog",            "/blog/",      ["agent-runtime"]),
    ("LangChain Blog",   "https://blog.langchain.dev",         "/",           ["multi-agent-orchestration"]),
    ("LlamaIndex Blog",  "https://medium.com/llamaindex-blog", "/",           ["rag"]),
    ("HuggingFace Blog", "https://huggingface.co/blog",        "/blog/",      ["infra"]),
]

# YC sources: RFS page + batch blog announcement posts
YC_RFS_URL: str = "https://www.ycombinator.com/rfs"
YC_BLOG_URL: str = "https://www.ycombinator.com/blog"

# Filter news/newsletters to posts on or after this date
NEWS_CUTOFF_DATE: str = "2025-01-01"


# ── Rate limiter ───────────────────────────────────────────────────────────────

class RateLimiter:
    """Enforces a minimum delay between HTTP requests to the same domain."""

    def __init__(self, delay_seconds: float = REQUEST_DELAY_SECONDS) -> None:
        """
        Args:
            delay_seconds: Minimum seconds between requests to the same host.
        """
        self._delay = delay_seconds
        self._last_fetch: dict[str, float] = {}

    def wait(self, url: str) -> None:
        """
        Sleep if needed so at least `delay_seconds` has elapsed since the last
        request to this URL's host. Updates the last-fetch timestamp.

        Args:
            url: The full URL about to be fetched.
        """
        host = urllib.parse.urlparse(url).netloc
        elapsed = time.monotonic() - self._last_fetch.get(host, 0.0)
        if elapsed < self._delay:
            time.sleep(self._delay - elapsed)
        self._last_fetch[host] = time.monotonic()


# ── Robots.txt cache ───────────────────────────────────────────────────────────

_robots_cache: dict[str, urllib.robotparser.RobotFileParser] = {}


def is_allowed(url: str, user_agent: str = "*") -> bool:
    """
    Check whether `url` is fetchable under the site's robots.txt.
    Caches one RobotFileParser per domain to avoid redundant fetches.

    Returns True if the URL is allowed or if robots.txt is unreachable,
    so transient network errors do not silently block entire domains.

    Args:
        url:        Full URL to check.
        user_agent: User agent string to check against.

    Returns:
        True if fetching is permitted or robots.txt unavailable.
    """
    parsed = urllib.parse.urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"

    if robots_url not in _robots_cache:
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(robots_url)
        try:
            rp.read()
        except Exception:
            # Treat unreadable robots.txt as permissive
            _robots_cache[robots_url] = None  # type: ignore[assignment]
            return True
        _robots_cache[robots_url] = rp

    rp = _robots_cache[robots_url]
    if rp is None:
        return True
    return rp.can_fetch(user_agent, url)


# ── Raw HTML cache ─────────────────────────────────────────────────────────────

def cache_path(url: str) -> pathlib.Path:
    """
    Derive a deterministic file path under RAW_DIR for caching the HTML of `url`.
    Uses the first 16 hex chars of SHA-256(url) as the filename stem so the
    cache survives re-runs without refetching already-seen pages.

    Args:
        url: The URL whose content will be cached.

    Returns:
        Absolute Path to the cache file (may not yet exist).
    """
    digest = hashlib.sha256(url.encode()).hexdigest()[:16]
    return RAW_DIR / f"{digest}.html"


def fetch_html(url: str, client: httpx.Client, rate_limiter: RateLimiter) -> str:
    """
    Fetch raw HTML for `url`, using the on-disk cache in RAW_DIR when available.
    Respects robots.txt and rate limits before making a live request.

    Returns the raw HTML string (from cache or live fetch).
    Returns "" rather than raising if the HTTP request fails after one retry,
    so that a single unreachable site does not abort the entire scrape run.

    Args:
        url:          URL to fetch.
        client:       Shared httpx.Client (reuse connection pool across calls).
        rate_limiter: RateLimiter instance to enforce per-domain delay.

    Returns:
        Raw HTML string, or "" if disallowed by robots.txt or fetch failed.
    """
    cached = cache_path(url)
    if cached.exists():
        return cached.read_text(encoding="utf-8", errors="replace")

    if not is_allowed(url, USER_AGENT):
        log.warning("robots.txt disallows %s — skipping", url)
        return ""

    rate_limiter.wait(url)
    for attempt in range(2):
        try:
            resp = client.get(url, timeout=HTTP_TIMEOUT_SECONDS)
            resp.raise_for_status()
            html_text = resp.text
            RAW_DIR.mkdir(parents=True, exist_ok=True)
            cached.write_text(html_text, encoding="utf-8")
            return html_text
        except Exception as exc:
            if attempt == 0:
                log.debug("Retrying %s after error: %s", url, exc)
                time.sleep(2.0)
            else:
                log.warning("Failed to fetch %s: %s", url, exc)
                return ""
    return ""


# ── Text extraction ────────────────────────────────────────────────────────────

class _MetaDescriptionParser(html.parser.HTMLParser):
    """Minimal HTML parser that extracts <meta name='description' content='...'/>."""

    def __init__(self) -> None:
        super().__init__()
        self.description: str = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        if tag != "meta":
            return
        attr_dict = dict(attrs)
        if attr_dict.get("name", "").lower() == "description":
            self.description = attr_dict.get("content", "") or ""


def extract_text(html_str: str, url: str) -> str:
    """
    Extract the main readable text from raw HTML using trafilatura.
    Falls back to <meta name='description'> content for JS-heavy SPAs where
    trafilatura finds no body text (e.g. React apps that return near-empty HTML).

    Args:
        html_str: Raw HTML string.
        url:      Source URL passed to trafilatura for metadata heuristics.

    Returns:
        Extracted plain text (non-empty when possible), or "" if nothing found.
    """
    text = trafilatura.extract(html_str, url=url, include_comments=False, include_tables=False)
    if text:
        return text

    # JS-heavy SPA fallback: at least surface the meta description
    parser = _MetaDescriptionParser()
    try:
        parser.feed(html_str)
    except Exception:
        pass
    return parser.description


# ── Blog link discovery ────────────────────────────────────────────────────────

class _LinkParser(html.parser.HTMLParser):
    """Collects href values from <a> tags that match a given path prefix."""

    def __init__(self, base_url: str, path_prefix: str) -> None:
        super().__init__()
        self.base_url = base_url
        self.path_prefix = path_prefix.rstrip("/")
        self.links: list[str] = []
        self._seen: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        if tag != "a":
            return
        href = dict(attrs).get("href", "") or ""
        if not href or href.startswith("#") or href.startswith("mailto:"):
            return
        abs_url = urllib.parse.urljoin(self.base_url, href)
        parsed = urllib.parse.urlparse(abs_url)
        # Accept paths that start with path_prefix and go deeper (not just the index)
        if parsed.path.startswith(self.path_prefix + "/") and abs_url not in self._seen:
            self._seen.add(abs_url)
            self.links.append(abs_url)


def discover_blog_links(
    html_str: str,
    base_url: str,
    path_prefix: str,
    max_links: int = 2,
) -> list[str]:
    """
    Parse raw HTML and return up to `max_links` absolute URLs whose path
    starts with `path_prefix` (e.g. "/blog"). Uses stdlib html.parser —
    no third-party HTML parsing library required.

    Args:
        html_str:    Raw HTML string of the page to scan.
        base_url:    Base URL for resolving relative hrefs.
        path_prefix: URL path prefix to match (e.g. "/blog").
        max_links:   Maximum number of links to return.

    Returns:
        List of absolute URL strings, deduplicated and in document order.
    """
    parser = _LinkParser(base_url, path_prefix)
    try:
        parser.feed(html_str)
    except Exception:
        pass
    return parser.links[:max_links]


# ── RSS date parsing ───────────────────────────────────────────────────────────

def parse_rss_date(entry: feedparser.FeedParserDict) -> str:
    """
    Extract and normalize the publish date from a feedparser entry to an
    ISO 8601 date string (YYYY-MM-DD). Returns "" if no date is parseable.

    Checks entry.published_parsed, then entry.updated_parsed in that order.
    Some Substack feeds omit dates on older posts; callers treat "" as
    "unknown / include anyway" rather than filtering the entry out.

    Args:
        entry: A single feedparser entry dict.

    Returns:
        ISO date string (e.g. "2025-03-15") or "" if unavailable.
    """
    for attr in ("published_parsed", "updated_parsed"):
        struct = getattr(entry, attr, None)
        if struct:
            try:
                dt = datetime(*struct[:6], tzinfo=timezone.utc)
                return dt.strftime("%Y-%m-%d")
            except Exception:
                continue
    return ""


# ── Schema normalization ───────────────────────────────────────────────────────

def normalize_record(
    title: str,
    description: str,
    source_url: str,
    tags: list[str],
    publish_date: str,
) -> dict:
    """
    Validate and normalize a single corpus record. Truncates description to
    MAX_DESC_CHARS. Filters tags to only values in TAG_TAXONOMY so that
    free-form tags from scrapers cannot pollute the taxonomy.

    Args:
        title:        Document or page title.
        description:  Extracted body text.
        source_url:   Canonical URL of the fetched resource.
        tags:         Tag list (will be filtered to TAG_TAXONOMY values).
        publish_date: ISO 8601 date string, or "" if unknown.

    Returns:
        Dict with keys: title, description, source_url, tags, publish_date.

    Raises:
        ValueError: If title or source_url is blank after stripping.
    """
    title = title.strip()
    source_url = source_url.strip()
    if not title:
        raise ValueError("title must not be blank")
    if not source_url:
        raise ValueError("source_url must not be blank")

    return {
        "title": title,
        "description": description.strip()[:MAX_DESC_CHARS],
        "source_url": source_url,
        "tags": [t for t in tags if t in TAG_TAXONOMY],
        "publish_date": publish_date,
    }


# ── URL deduplication ──────────────────────────────────────────────────────────

def normalize_url(url: str) -> str:
    """
    Canonicalize a URL for deduplication: lowercase scheme and host,
    strip trailing slash from path, preserve query string.
    This catches http:// vs https:// and trailing-slash variants that would
    otherwise produce duplicate embeddings in the FAISS index.

    Args:
        url: Raw URL string.

    Returns:
        Normalized URL string.
    """
    parsed = urllib.parse.urlparse(url)
    return urllib.parse.urlunparse(
        parsed._replace(
            scheme=parsed.scheme.lower(),
            netloc=parsed.netloc.lower(),
            path=parsed.path.rstrip("/"),
        )
    )


def load_seen_urls() -> set[str]:
    """
    Build the set of all source_url values already present across every
    *.json file in CORPUS_DIR, normalized via normalize_url.

    Returns:
        Set of normalized URL strings already in the corpus.
    """
    seen: set[str] = set()
    for path in CORPUS_DIR.glob("*.json"):
        try:
            records = json.loads(path.read_text())
        except Exception:
            continue
        for rec in records:
            url = rec.get("source_url", "")
            if url:
                seen.add(normalize_url(url))
    return seen


# ── Shard writer ───────────────────────────────────────────────────────────────

def append_to_shard(records: list[dict], shard_name: str, dry_run: bool = False) -> int:
    """
    Append `records` to data/corpus/<shard_name>.json, creating the file if
    absent. Skips records whose source_url is already in any corpus shard.

    Args:
        records:    List of normalized corpus records.
        shard_name: Filename stem (e.g. "startups", "yc", "newsletters").
        dry_run:    If True, compute and return the count without writing.

    Returns:
        Number of new records that would be (or were) appended.
    """
    seen = load_seen_urls()
    new_records = [
        r for r in records
        if normalize_url(r.get("source_url", "")) not in seen
    ]
    if dry_run or not new_records:
        return len(new_records)

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    shard_path = CORPUS_DIR / f"{shard_name}.json"
    existing: list[dict] = []
    if shard_path.exists():
        try:
            existing = json.loads(shard_path.read_text())
        except Exception:
            existing = []

    shard_path.write_text(
        json.dumps(existing + new_records, indent=2, ensure_ascii=False)
    )
    return len(new_records)


# ── Source scrapers ────────────────────────────────────────────────────────────

def scrape_startups(
    client: httpx.Client,
    rate_limiter: RateLimiter,
    seen_urls: set[str],
) -> list[dict]:
    """
    Fetch homepage, /about page, and up to 2 blog posts for each startup in
    STARTUP_TARGETS. Extracts text with trafilatura and normalizes to corpus
    schema. Skips pages already in seen_urls.

    Blog post discovery: looks for <a> tags whose href contains "/blog"
    and fetches the first two unique links not already in seen_urls.

    Args:
        client:       Shared httpx.Client.
        rate_limiter: Per-domain rate limiter.
        seen_urls:    Already-seen URLs (mutated in-place with new URLs).

    Returns:
        List of normalized corpus records.
    """
    records: list[dict] = []

    for name, homepage, tags in STARTUP_TARGETS:
        # Each startup contributes homepage + /about + up to 2 blog posts
        candidate_pages = [homepage, homepage.rstrip("/") + "/about"]

        for page_url in candidate_pages:
            norm = normalize_url(page_url)
            if norm in seen_urls:
                continue
            html_str = fetch_html(page_url, client, rate_limiter)
            if not html_str:
                continue
            text = extract_text(html_str, page_url)
            if not text:
                log.debug("No text extracted from %s", page_url)
                continue
            try:
                record = normalize_record(
                    title=f"{name} — {page_url.split('/')[-1] or 'homepage'}",
                    description=text,
                    source_url=page_url,
                    tags=tags,
                    publish_date="",
                )
                records.append(record)
                seen_urls.add(norm)
            except ValueError as exc:
                log.debug("Skipping %s: %s", page_url, exc)

            # Discover blog posts from the homepage HTML
            if page_url == homepage:
                blog_links = discover_blog_links(html_str, homepage, "/blog", max_links=2)
                for blog_url in blog_links:
                    bnorm = normalize_url(blog_url)
                    if bnorm in seen_urls:
                        continue
                    bhtml = fetch_html(blog_url, client, rate_limiter)
                    if not bhtml:
                        continue
                    btext = extract_text(bhtml, blog_url)
                    if not btext:
                        continue
                    try:
                        brec = normalize_record(
                            title=f"{name} Blog — {urllib.parse.urlparse(blog_url).path.strip('/').split('/')[-1]}",
                            description=btext,
                            source_url=blog_url,
                            tags=tags,
                            publish_date="",
                        )
                        records.append(brec)
                        seen_urls.add(bnorm)
                    except ValueError as exc:
                        log.debug("Skipping blog %s: %s", blog_url, exc)

        log.info("Startups: %s — %d records so far", name, len(records))

    return records


def scrape_yc(
    client: httpx.Client,
    rate_limiter: RateLimiter,
    seen_urls: set[str],
) -> list[dict]:
    """
    Scrape ycombinator.com/rfs (server-rendered RFS sections) and YC blog
    batch announcement posts tagged AI. Targets the server-rendered pages only
    — the ycombinator.com/companies React SPA is XHR-only and not accessible
    via httpx, so it is intentionally excluded.

    Args:
        client:       Shared httpx.Client.
        rate_limiter: Per-domain rate limiter.
        seen_urls:    Already-seen URLs (mutated in-place with new URLs).

    Returns:
        List of normalized corpus records (~30 target).
    """
    records: list[dict] = []

    # RFS page
    rfs_norm = normalize_url(YC_RFS_URL)
    if rfs_norm not in seen_urls:
        html_str = fetch_html(YC_RFS_URL, client, rate_limiter)
        if html_str:
            text = extract_text(html_str, YC_RFS_URL)
            if text:
                try:
                    records.append(normalize_record(
                        title="Y Combinator Request for Startups",
                        description=text,
                        source_url=YC_RFS_URL,
                        tags=["infra"],
                        publish_date="",
                    ))
                    seen_urls.add(rfs_norm)
                except ValueError as exc:
                    log.debug("YC RFS skipped: %s", exc)

    # YC blog: find posts about recent batches and AI investment themes
    blog_norm = normalize_url(YC_BLOG_URL)
    if blog_norm not in seen_urls:
        blog_html = fetch_html(YC_BLOG_URL, client, rate_limiter)
        if blog_html:
            # Discover posts linked from the blog index that mention batch or AI
            links = discover_blog_links(blog_html, YC_BLOG_URL, "/blog", max_links=10)
            for link in links:
                lnorm = normalize_url(link)
                if lnorm in seen_urls:
                    continue
                lhtml = fetch_html(link, client, rate_limiter)
                if not lhtml:
                    continue
                ltext = extract_text(lhtml, link)
                if not ltext:
                    continue
                # Filter to posts relevant to AI/agentic themes
                ltext_lower = ltext.lower()
                if not any(kw in ltext_lower for kw in ["ai", "agent", "llm", "batch", "w25", "s25", "w24"]):
                    continue
                try:
                    title_slug = urllib.parse.urlparse(link).path.strip("/").split("/")[-1].replace("-", " ").title()
                    records.append(normalize_record(
                        title=f"YC Blog — {title_slug}",
                        description=ltext,
                        source_url=link,
                        tags=["infra"],
                        publish_date="",
                    ))
                    seen_urls.add(lnorm)
                except ValueError as exc:
                    log.debug("YC blog post skipped %s: %s", link, exc)

    log.info("YC: %d records", len(records))
    return records


def scrape_newsletters(
    client: httpx.Client,
    rate_limiter: RateLimiter,
    seen_urls: set[str],
) -> list[dict]:
    """
    Fetch RSS feeds from NEWSLETTER_RSS_FEEDS, extract up to 15 entries per
    feed. For each entry: fetch the full article HTML via the entry link,
    extract text with trafilatura, normalize to corpus schema.
    Filters to entries published on or after NEWS_CUTOFF_DATE; entries with
    no date are included (treated as "unknown, include anyway") to avoid
    silently dropping older Substack posts that lack date metadata.

    Args:
        client:       Shared httpx.Client.
        rate_limiter: Per-domain rate limiter.
        seen_urls:    Already-seen URLs (mutated in-place with new URLs).

    Returns:
        List of normalized corpus records (~60 target).
    """
    records: list[dict] = []

    for feed_name, rss_url, tags in NEWSLETTER_RSS_FEEDS:
        try:
            feed = feedparser.parse(rss_url)
        except Exception as exc:
            log.warning("Failed to parse RSS %s: %s", rss_url, exc)
            continue

        count = 0
        for entry in feed.entries[:15]:
            if count >= 15:
                break
            link = getattr(entry, "link", "")
            if not link:
                continue
            lnorm = normalize_url(link)
            if lnorm in seen_urls:
                continue

            pub_date = parse_rss_date(entry)
            # Include entries with no date; filter out entries before cutoff
            if pub_date and pub_date < NEWS_CUTOFF_DATE:
                continue

            html_str = fetch_html(link, client, rate_limiter)
            if not html_str:
                continue
            text = extract_text(html_str, link)
            if not text:
                continue

            title = getattr(entry, "title", "").strip() or feed_name
            try:
                records.append(normalize_record(
                    title=f"{feed_name} — {title}",
                    description=text,
                    source_url=link,
                    tags=tags,
                    publish_date=pub_date,
                ))
                seen_urls.add(lnorm)
                count += 1
            except ValueError as exc:
                log.debug("Newsletter entry skipped %s: %s", link, exc)

        log.info("Newsletters [%s]: %d new records", feed_name, count)

    return records


def scrape_news(
    client: httpx.Client,
    rate_limiter: RateLimiter,
    seen_urls: set[str],
) -> list[dict]:
    """
    Fetch RSS feeds from NEWS_RSS_FEEDS. Filter entries to those with
    publish dates in 2025-01-01 through 2026-12-31 (or no date). Take
    up to 25 entries per feed. Fetch full article text for each filtered entry.

    Args:
        client:       Shared httpx.Client.
        rate_limiter: Per-domain rate limiter.
        seen_urls:    Already-seen URLs (mutated in-place with new URLs).

    Returns:
        List of normalized corpus records (~50 target).
    """
    records: list[dict] = []

    for feed_name, rss_url, tags in NEWS_RSS_FEEDS:
        try:
            feed = feedparser.parse(rss_url)
        except Exception as exc:
            log.warning("Failed to parse RSS %s: %s", rss_url, exc)
            continue

        count = 0
        for entry in feed.entries:
            if count >= 25:
                break
            link = getattr(entry, "link", "")
            if not link:
                continue
            lnorm = normalize_url(link)
            if lnorm in seen_urls:
                continue

            pub_date = parse_rss_date(entry)
            # Include undated entries; filter entries outside 2025–2026 window
            if pub_date and not ("2025-01-01" <= pub_date <= "2026-12-31"):
                continue

            html_str = fetch_html(link, client, rate_limiter)
            if not html_str:
                continue
            text = extract_text(html_str, link)
            if not text:
                continue

            title = getattr(entry, "title", "").strip() or feed_name
            try:
                records.append(normalize_record(
                    title=title,
                    description=text,
                    source_url=link,
                    tags=tags,
                    publish_date=pub_date,
                ))
                seen_urls.add(lnorm)
                count += 1
            except ValueError as exc:
                log.debug("News entry skipped %s: %s", link, exc)

        log.info("News [%s]: %d new records", feed_name, count)

    return records


def scrape_tech_blogs(
    client: httpx.Client,
    rate_limiter: RateLimiter,
    seen_urls: set[str],
) -> list[dict]:
    """
    Fetch the blog index page for each entry in TECH_BLOG_TARGETS.
    Discover post links by scanning <a href> elements that match the blog's
    path prefix. Fetch and extract text for up to 8 posts per blog. Filter
    to posts with dates >= 2024-01-01 (or no date).

    Args:
        client:       Shared httpx.Client.
        rate_limiter: Per-domain rate limiter.
        seen_urls:    Already-seen URLs (mutated in-place with new URLs).

    Returns:
        List of normalized corpus records (~40 target).
    """
    records: list[dict] = []

    for blog_name, index_url, path_prefix, tags in TECH_BLOG_TARGETS:
        index_html = fetch_html(index_url, client, rate_limiter)
        if not index_html:
            log.warning("Could not fetch blog index %s", index_url)
            continue

        links = discover_blog_links(index_html, index_url, path_prefix, max_links=8)
        count = 0
        for link in links:
            if count >= 8:
                break
            lnorm = normalize_url(link)
            if lnorm in seen_urls:
                continue

            html_str = fetch_html(link, client, rate_limiter)
            if not html_str:
                continue
            text = extract_text(html_str, link)
            if not text:
                continue

            title_slug = urllib.parse.urlparse(link).path.strip("/").split("/")[-1].replace("-", " ").title()
            try:
                records.append(normalize_record(
                    title=f"{blog_name} — {title_slug}",
                    description=text,
                    source_url=link,
                    tags=tags,
                    publish_date="",
                ))
                seen_urls.add(lnorm)
                count += 1
            except ValueError as exc:
                log.debug("Tech blog post skipped %s: %s", link, exc)

        log.info("Tech blogs [%s]: %d new records", blog_name, count)

    return records


# ── CLI entry point ────────────────────────────────────────────────────────────

_SOURCE_MAP = {
    "startups":    ("startups",    scrape_startups),
    "yc":          ("yc",          scrape_yc),
    "newsletters": ("newsletters", scrape_newsletters),
    "news":        ("news",        scrape_news),
    "techblogs":   ("techblogs",   scrape_tech_blogs),
}


def main() -> None:
    """
    CLI entry point. Parses --source and --dry-run flags, then runs the
    selected source scrapers and appends results to corpus shards.

    --source: one of startups, yc, newsletters, news, techblogs, all (default: all).
    --dry-run: compute record counts per source without writing to disk.

    Logs the number of new records written per shard at INFO level.
    """
    parser = argparse.ArgumentParser(description="Scrape corpus documents for the market analyst pipeline.")
    parser.add_argument(
        "--source",
        choices=list(_SOURCE_MAP.keys()) + ["all"],
        default="all",
        help="Which source to scrape (default: all).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print counts without writing to disk.",
    )
    args = parser.parse_args()

    sources = list(_SOURCE_MAP.items()) if args.source == "all" else [
        (args.source, _SOURCE_MAP[args.source])
    ]

    headers = {"User-Agent": USER_AGENT}
    seen_urls = load_seen_urls()

    with httpx.Client(headers=headers, follow_redirects=True) as client:
        rate_limiter = RateLimiter()
        for source_key, (shard_name, scrape_fn) in sources:
            log.info("Starting source: %s", source_key)
            records = scrape_fn(client, rate_limiter, seen_urls)
            written = append_to_shard(records, shard_name, dry_run=args.dry_run)
            action = "would write" if args.dry_run else "wrote"
            log.info("Source [%s]: %s %d new records to data/corpus/%s.json", source_key, action, written, shard_name)

    log.info("Scrape complete.")


if __name__ == "__main__":
    main()
