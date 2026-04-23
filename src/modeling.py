
from sentiment import SentimentAnalyzer

import os

# call our sentiment script and give it a start date for pulling data
analyzer = SentimentAnalyzer(av_api_key=os.getenv("ALPHA_VANTAGE_API_KEY"))
result = analyzer.get_net_sentiment(time_from="20260101T0000")

# pulling data directly from the result
print(result["net_sentiment"])    # e.g. 0.02  — single scalar for the model
print(result["article_count"])    # e.g. 50
print(result["series"])           # net sentiment broken down by quarter
print(result["df"].head())        # full scored article table

# an efficient way to index specific values should we need them
df = result['df']

sentiment = {
    "positive": (df["finbert_label"] == "positive").mean(),
    "neutral":  (df["finbert_label"] == "neutral").mean(),
    "negative": (df["finbert_label"] == "negative").mean(),
    "net":      result["net_sentiment"],
}


