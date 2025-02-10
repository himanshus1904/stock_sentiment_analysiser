CONFIG = {
    "reddit": {
        "subreddit": ["IndianStockMarket", "IndiaInvestments", "IndianStreetBets", "StockMarketIndia", "DalalStreetTalks"],
        "time_filter": "month",
        "post_limit": 50,
        "comment_limit": 30
    },
    "model": {
        "name": "ProsusAI/finbert",
        "batch_size": 16,
        "confidence_threshold": 0.6
    },
    "sentiment_labels": ["Negative", "Neutral", "Positive"],
    "cache_file": "reddit_sentiment_cache.csv"
}
