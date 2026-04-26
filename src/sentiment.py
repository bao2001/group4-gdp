"""
FinBERT Financial News Sentiment Analyzer
==========================================
Fetches financial headlines from Alpha Vantage and scores them using
ProsusAI/finbert via HuggingFace Transformers.

Usage:
    python finbert_sentiment.py

    Or call from another script:
        from finbert_sentiment import SentimentAnalyzer

        analyzer = SentimentAnalyzer(av_api_key=os.getenv("ALPHA_VANTAGE_API_KEY"))
        result   = analyzer.get_net_sentiment(time_from="20260101T0000")
        net      = result["net_sentiment"]   # single float for GDP model input

Alpha Vantage free-tier limits:
    - 25 requests / day
    - 1 request / second burst limit
    Get a free key: https://www.alphavantage.co/support/#api-key
"""
#################################################################################
# how to call this file
# from sentiment import SentimentAnalyzer

# import os

# # call our sentiment script and give it a start date for pulling data
# analyzer = SentimentAnalyzer(av_api_key=os.getenv("ALPHA_VANTAGE_API_KEY"))
# result = analyzer.get_net_sentiment(time_from="20260101T0000")

# # pulling data directly from the result
# print(result["net_sentiment"])    # e.g. 0.02  — single scalar for the model
# print(result["article_count"])    # e.g. 50
# print(result["series"])           # net sentiment broken down by quarter
# print(result["df"].head())        # full scored article table

# # an efficient way to index specific values should we need them
# df = result['df']

# sentiment = {
#     "positive": (df["finbert_label"] == "positive").mean(),
#     "neutral":  (df["finbert_label"] == "neutral").mean(),
#     "negative": (df["finbert_label"] == "negative").mean(),
#     "net":      result["net_sentiment"],
# }

# ALPHA_VANTAGE_API_KEY=TWVGV64WICIFEQZL
#################################################################################



import os
import time
import logging

from dataclasses import dataclass, field
from typing import Optional

import requests
import pandas as pd

from dotenv import load_dotenv
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data folders
RAW_DATA_DIR       = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

# Output folders
FIGURES_DIR        = os.path.join(BASE_DIR, "outputs", "figures")
TABLES_DIR         = os.path.join(BASE_DIR, "outputs", "tables")

ALPHA_VANTAGE_BASE = "https://www.alphavantage.co/query"
FINBERT_MODEL_ID   = "ProsusAI/finbert"

# Topics relevant to GDP / macroeconomic analysis.
# Full list: https://www.alphavantage.co/documentation/#news-sentiment
GDP_RELEVANT_TOPICS = [
    "economy_fiscal",
    "economy_macro",
    "economy_monetary",
    "finance",
    "manufacturing",
    "real_estate",
    "retail_wholesale",
    "energy_transportation",
]


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class AlphaVantageRateLimitError(RuntimeError):
    """Raised when Alpha Vantage returns a rate-limit or daily-cap message."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Article:
    title: str
    summary: str
    source: str
    published_at: str
    url: str
    topics: list[str] = field(default_factory=list)
    av_overall_sentiment: Optional[str]  = None
    av_overall_score: Optional[float]    = None


@dataclass
class ScoredArticle:
    article: Article
    finbert_label: str   # "positive" | "negative" | "neutral"
    finbert_score: float # model confidence (0–1)

    @property
    def sentiment_value(self) -> float:
        """Numeric encoding: positive=+1, neutral=0, negative=-1."""
        return {"positive": 1.0, "neutral": 0.0, "negative": -1.0}[self.finbert_label]


# ---------------------------------------------------------------------------
# News fetcher
# ---------------------------------------------------------------------------

class NewsFetcher:
    """Pulls financial news from Alpha Vantage NEWS_SENTIMENT endpoint."""

    _REQUEST_DELAY = 1.2  # seconds — respects 1 req/sec free-tier burst limit

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Alpha Vantage API key is required.")
        self.api_key = api_key

    def fetch(
        self,
        topics: Optional[list[str]] = None,
        limit: int = 50,
        sort: str = "LATEST",
        time_from: Optional[str] = None,
        tickers: Optional[list[str]] = None,
    ) -> list[Article]:
        """
        Fetch articles from Alpha Vantage.

        Args:
            topics:    Topic filters. Pass None or [] for no filter (broadest query).
            limit:     Max articles (1–1000; free tier caps at 50).
            sort:      "LATEST" | "EARLIEST" | "RELEVANCE".
            time_from: Optional start datetime string (YYYYMMDDTHHMM).
            tickers:   Optional ticker symbols to filter by.

        Raises:
            AlphaVantageRateLimitError: daily cap (25/day) or per-second limit hit.
            RuntimeError: unexpected API response format.
        """
        params: dict = {
            "function": "NEWS_SENTIMENT",
            "limit":    limit,
            "sort":     sort,
            "apikey":   self.api_key,
        }
        if topics:
            params["topics"] = ",".join(topics)
        if time_from:
            params["time_from"] = time_from
        if tickers:
            params["tickers"] = ",".join(tickers)

        topic_label = ",".join(topics) if topics else "(none)"
        log.info("Fetching up to %d articles  topics=%s …", limit, topic_label)

        # Respect the 1 req/sec free-tier burst limit before every call.
        time.sleep(self._REQUEST_DELAY)

        response = requests.get(ALPHA_VANTAGE_BASE, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        # Alpha Vantage signals rate limits via "Information" or "Note" keys.
        if "Information" in data:
            raise AlphaVantageRateLimitError(data["Information"])
        if "Note" in data:
            raise AlphaVantageRateLimitError(data["Note"])
        if "feed" not in data:
            raise RuntimeError(
                f"Unexpected Alpha Vantage response (keys: {list(data.keys())}): {data}"
            )

        articles = [self._parse(item) for item in data["feed"]]
        log.info("Fetched %d articles.", len(articles))

        if len(articles) == 0:
            log.warning(
                "Alpha Vantage returned an empty feed.\n"
                "  Possible causes:\n"
                "    1. Topic filter too narrow — try fewer topics or topics=None\n"
                "    2. 'time_from' window has no coverage\n"
                "  Response keys: %s",
                list(data.keys()),
            )

        return articles

    @staticmethod
    def _parse(item: dict) -> Article:
        return Article(
            title=item.get("title", ""),
            summary=item.get("summary", ""),
            source=item.get("source", ""),
            published_at=item.get("time_published", ""),
            url=item.get("url", ""),
            topics=[t["topic"] for t in item.get("topics", [])],
            av_overall_sentiment=item.get("overall_sentiment_label"),
            av_overall_score=item.get("overall_sentiment_score"),
        )


# ---------------------------------------------------------------------------
# FinBERT scorer
# ---------------------------------------------------------------------------

class FinBERTScorer:
    """
    Wraps ProsusAI/finbert for batch inference.

    The model is downloaded once (~400 MB) and cached locally by HuggingFace.
    CPU: ~2–4 s for 50 headlines. Set device=0 for CUDA GPU acceleration.
    """

    def __init__(self, device: int = -1):
        log.info("Loading %s …", FINBERT_MODEL_ID)
        tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL_ID)
        model     = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL_ID)
        self.pipe = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=device,
        )
        log.info("Model loaded.")

    def score(self, texts: list[str], batch_size: int = 16) -> list[dict]:
        """
        Score a list of text strings.

        Returns:
            [{"label": "positive"|"negative"|"neutral", "score": float}, …]
        """
        if not texts:
            return []
        truncated = [t[:512] for t in texts]
        log.info("Scoring %d texts (batch_size=%d) …", len(truncated), batch_size)
        results = self.pipe(truncated, batch_size=batch_size, truncation=True)
        return [{"label": r["label"].lower(), "score": r["score"]} for r in results]


# ---------------------------------------------------------------------------
# High-level analyzer
# ---------------------------------------------------------------------------

class SentimentAnalyzer:
    """
    Orchestrates news fetching + FinBERT scoring.

    Instantiate once per session — FinBERT takes a few seconds to load
    and should not be re-constructed on every call.

    Typical usage from another script:
        analyzer = SentimentAnalyzer(av_api_key=os.getenv("ALPHA_VANTAGE_API_KEY"))
        result   = analyzer.get_net_sentiment(time_from="20260101T0000")
        feature  = result["net_sentiment"]   # float for GDP model
    """

    def __init__(self, av_api_key: str, device: int = -1):
        self.fetcher = NewsFetcher(av_api_key)
        self.scorer  = FinBERTScorer(device=device)

    # ------------------------------------------------------------------
    # Primary interface — use this when calling from another script
    # ------------------------------------------------------------------

    def get_net_sentiment(
        self,
        time_from: Optional[str] = None,
        topics: Optional[list[str]] = None,
        limit: int = 50,
        text_field: str = "title",
        freq: str = "Q",
    ) -> dict:
        """
        Fetch, score, and aggregate in one call.

        Args:
            time_from:  Start of window (YYYYMMDDTHHMM). None = latest available.
            topics:     Topic filters. Defaults to GDP_RELEVANT_TOPICS.
            limit:      Max articles to fetch (free tier: 50 per request).
            text_field: "title" | "summary" | "both"
            freq:       Aggregation period — "Q" quarterly, "W" weekly, "M" monthly.

        Returns:
            {
                "net_sentiment":  float | None,   # mean net sentiment over the window
                                                  # — plug this into your GDP model
                "article_count":  int,            # total articles scored
                "series":         pd.Series,      # net_sentiment indexed by period
                                                  # — inspect trend if needed
                "df":             pd.DataFrame,   # full scored article table
                                                  # — use for debugging / saving
            }
        """
        if topics is None:
            topics = GDP_RELEVANT_TOPICS

        df = self.run(topics=topics, limit=limit, text_field=text_field, time_from=time_from)

        # Fallback 1: topic filter too narrow — retry with no topic filter
        if df.empty and topics:
            log.warning(
                "Topic-filtered fetch returned 0 articles — retrying without topic filter."
            )
            df = self.run(topics=None, limit=limit, text_field=text_field, time_from=time_from)

        # Fallback 2: time_from window too restrictive — retry with no date filter
        if df.empty and time_from:
            log.warning(
                "Fetch with time_from=%s returned 0 articles — retrying without date filter.",
                time_from,
            )
            df = self.run(topics=None, limit=limit, text_field=text_field)

        if df.empty:
            log.warning("get_net_sentiment: no articles returned after all fallbacks — returning None.")
            return {
                "net_sentiment": None,
                "article_count": 0,
                "series":        pd.Series(dtype=float),
                "df":            df,
            }

        # Period-level net sentiment series
        df_copy = df.copy()
        df_copy["period"] = df_copy["published_at"].dt.to_period(freq)

        def _net(g):
            total = len(g)
            pos   = (g["finbert_label"] == "positive").sum()
            neg   = (g["finbert_label"] == "negative").sum()
            return round((pos - neg) / total, 4)

        series = df_copy.groupby("period").apply(_net).rename("net_sentiment")

        return {
            "net_sentiment": round(float(df["sentiment_value"].mean()), 4),
            "article_count": len(df),
            "series":        series,
            "df":            df,
        }

    # ------------------------------------------------------------------
    # Lower-level interface — useful in Jupyter for exploration
    # ------------------------------------------------------------------

    def run(
        self,
        topics: Optional[list[str]] = None,
        limit: int = 50,
        text_field: str = "title",
        **fetch_kwargs,
    ) -> pd.DataFrame:
        """
        Fetch articles and return a fully scored DataFrame.

        Args:
            topics:         Topic filters (None = no filter, broadest query).
            limit:          Max articles to fetch.
            text_field:     "title" | "summary" | "both"
            **fetch_kwargs: Forwarded to NewsFetcher.fetch() — e.g. time_from, tickers.

        Returns:
            DataFrame columns:
                published_at, source, title, url, topics,
                finbert_label, finbert_score, sentiment_value,
                av_overall_sentiment, av_overall_score
        """
        articles = self.fetcher.fetch(topics=topics, limit=limit, **fetch_kwargs)

        texts = []
        for a in articles:
            if text_field == "title":
                texts.append(a.title)
            elif text_field == "summary":
                texts.append(a.summary)
            else:
                texts.append(f"{a.title}. {a.summary}")

        scores = self.scorer.score(texts)
        scored = [
            ScoredArticle(article=a, finbert_label=s["label"], finbert_score=s["score"])
            for a, s in zip(articles, scores)
        ]
        return self._to_dataframe(scored)

    @staticmethod
    def _to_dataframe(scored: list[ScoredArticle]) -> pd.DataFrame:
        COLUMNS = [
            "published_at", "source", "title", "url", "topics",
            "finbert_label", "finbert_score", "sentiment_value",
            "av_overall_sentiment", "av_overall_score",
        ]
        if not scored:
            return pd.DataFrame(columns=COLUMNS)

        rows = []
        for s in scored:
            a = s.article
            rows.append({
                "published_at":         a.published_at,
                "source":               a.source,
                "title":                a.title,
                "url":                  a.url,
                "topics":               ", ".join(a.topics),
                "finbert_label":        s.finbert_label,
                "finbert_score":        round(s.finbert_score, 4),
                "sentiment_value":      s.sentiment_value,
                "av_overall_sentiment": a.av_overall_sentiment,
                "av_overall_score":     a.av_overall_score,
            })

        df = pd.DataFrame(rows)
        df["published_at"] = pd.to_datetime(
            df["published_at"], format="%Y%m%dT%H%M%S", errors="coerce"
        )
        return df.sort_values("published_at", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Aggregate helpers
# ---------------------------------------------------------------------------

def compute_daily_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Roll per-article scores up to a daily feature table.

    Columns:
        mean_sentiment   - mean sentiment_value (−1 to +1)
        net_sentiment    - (positive − negative) / total
        positive_share   - fraction labelled positive
        negative_share   - fraction labelled negative
        neutral_share    - fraction labelled neutral
        article_count    - total articles that day
        mean_confidence  - mean FinBERT confidence score
    """
    return _aggregate(df, "D")


def compute_weekly_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Same as compute_daily_sentiment but aggregated by week."""
    return _aggregate(df, "W")


def compute_quarterly_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Same as compute_daily_sentiment but aggregated by quarter."""
    return _aggregate(df, "Q")


def _aggregate(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    if df.empty or "finbert_label" not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    df["period"] = df["published_at"].dt.to_period(freq)

    def agg(g):
        total = len(g)
        pos   = (g["finbert_label"] == "positive").sum()
        neg   = (g["finbert_label"] == "negative").sum()
        neu   = (g["finbert_label"] == "neutral").sum()
        return pd.Series({
            "mean_sentiment":  round(g["sentiment_value"].mean(), 4),
            "net_sentiment":   round((pos - neg) / total, 4),
            "positive_share":  round(pos / total, 4),
            "negative_share":  round(neg / total, 4),
            "neutral_share":   round(neu / total, 4),
            "article_count":   int(total),
            "mean_confidence": round(g["finbert_score"].mean(), 4),
        })

    return df.groupby("period").apply(agg).sort_index()


def print_summary(df: pd.DataFrame, agg_df: Optional[pd.DataFrame] = None) -> None:
    """Pretty-print a run summary to stdout."""
    total = len(df)

    print("\n" + "=" * 60)
    print("  FinBERT Sentiment Summary")
    print("=" * 60)
    print(f"  Articles analysed : {total}")

    if total == 0:
        print("  No articles to summarise — check the log above.")
        print("=" * 60 + "\n")
        return

    pos_pct = (df["finbert_label"] == "positive").mean() * 100
    neg_pct = (df["finbert_label"] == "negative").mean() * 100
    neu_pct = (df["finbert_label"] == "neutral").mean() * 100
    net     = df["sentiment_value"].mean()

    print(f"  Positive          : {pos_pct:.1f}%")
    print(f"  Neutral           : {neu_pct:.1f}%")
    print(f"  Negative          : {neg_pct:.1f}%")
    print(f"  Net sentiment     : {net:+.3f}  (range −1 to +1)")
    print("=" * 60)

    print("\nTop 5 most positive headlines:")
    for _, row in df[df["finbert_label"] == "positive"].nlargest(5, "finbert_score").iterrows():
        print(f"  [{row['finbert_score']:.2f}] {row['title'][:90]}")

    print("\nTop 5 most negative headlines:")
    for _, row in df[df["finbert_label"] == "negative"].nlargest(5, "finbert_score").iterrows():
        print(f"  [{row['finbert_score']:.2f}] {row['title'][:90]}")

    if agg_df is not None and not agg_df.empty:
        print("\nAggregate sentiment (most recent 7 periods):")
        print(agg_df.tail(7).to_string())

    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    av_key = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    if not av_key:
        raise SystemExit(
            "Set ALPHA_VANTAGE_API_KEY in your environment or a .env file.\n"
            "Get a free key at: https://www.alphavantage.co/support/#api-key"
        )

    # Instantiate once — reuse for all calls in this session
    analyzer = SentimentAnalyzer(av_api_key=av_key, device=-1)

    try:
        result = analyzer.get_net_sentiment(
            time_from="20260101T0000",
            freq="Q",
        )
    except AlphaVantageRateLimitError as e:
        log.error(
            "Alpha Vantage daily request limit reached (25 req/day on free tier).\n"
            "  Options:\n"
            "    1. Wait until tomorrow — the limit resets at midnight UTC.\n"
            "    2. Get a second free key at https://www.alphavantage.co/support/#api-key\n"
            "    3. Upgrade: https://www.alphavantage.co/premium/\n"
            "  Message from API: %s", e
        )
        raise SystemExit(1)

    df    = result["df"]
    daily = compute_daily_sentiment(df)
    print_summary(df, daily)

    log.info("Net sentiment (GDP model feature): %s", result["net_sentiment"])
    log.info("Articles scored: %d", result["article_count"])

    # Save outputs
    df.to_csv(f"{RAW_DATA_DIR}news_sentiment_raw.csv", index=False)
    if not daily.empty:
        daily.to_csv(f"{PROCESSED_DATA_DIR}news_sentiment_daily.csv")
    log.info(
        "Saved news_sentiment_raw.csv%s",
        " and news_sentiment_daily.csv" if not daily.empty else "",
    )
