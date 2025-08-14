import requests
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import asyncio
import aiohttp
from datetime import datetime
import logging


class CryptoSentimentAnalyzer:
    def __init__(self):
        # Initialize VADER for general sentiment
        self.vader = SentimentIntensityAnalyzer()

        # Initialize crypto-specific BERT model (optional - requires more memory)
        self.crypto_bert = None
        try:
            self.crypto_bert = pipeline(
                "sentiment-analysis",
                model="ElKulako/cryptobert",
                tokenizer="ElKulako/cryptobert",
                device=-1,  # Use CPU
            )
            print("CryptoBERT loaded successfully")
        except Exception as e:
            print(f"Could not load CryptoBERT: {e}. Using VADER only.")

    def preprocess_crypto_text(self, text):
        """Handle crypto-specific vocabulary and slang"""
        if not isinstance(text, str):
            return ""

        crypto_replacements = {
            "HODL": "hold",
            "FOMO": "fear of missing out",
            "FUD": "fear uncertainty doubt",
            "mooning": "rising rapidly",
            "diamond hands": "strong holder",
            "paper hands": "weak holder",
            "to the moon": "price increase",
            "rekt": "significant loss",
            "DYOR": "do your own research",
            "whale": "large investor",
            "pump": "price increase",
            "dump": "price decrease",
            "bear market": "declining market",
            "bull market": "rising market",
        }

        text_lower = text.lower()
        for crypto_term, replacement in crypto_replacements.items():
            text_lower = text_lower.replace(crypto_term.lower(), replacement)

        return text_lower

    def analyze_text_sentiment(self, text, use_bert=False):
        """Analyze sentiment of a single text"""
        if not text:
            return {"compound": 0, "confidence": 0, "method": "empty"}

        processed_text = self.preprocess_crypto_text(text)

        # VADER analysis
        vader_scores = self.vader.polarity_scores(processed_text)

        # If BERT is available and requested
        if use_bert and self.crypto_bert:
            try:
                bert_result = self.crypto_bert(
                    processed_text[:512]
                )  # Limit text length

                # Convert BERT labels to numerical scores
                if bert_result[0]["label"] == "POSITIVE":
                    bert_score = bert_result[0]["score"]
                elif bert_result[0]["label"] == "NEGATIVE":
                    bert_score = -bert_result[0]["score"]
                else:
                    bert_score = 0

                # Combine VADER and BERT (weighted average)
                combined_score = 0.6 * vader_scores["compound"] + 0.4 * bert_score

                return {
                    "compound": combined_score,
                    "confidence": (
                        abs(vader_scores["compound"]) + bert_result[0]["score"]
                    )
                    / 2,
                    "method": "vader+bert",
                    "vader_score": vader_scores["compound"],
                    "bert_score": bert_score,
                }

            except Exception as e:
                logging.warning(f"BERT analysis failed: {e}, falling back to VADER")

        return {
            "compound": vader_scores["compound"],
            "confidence": abs(vader_scores["compound"]),
            "method": "vader",
            "positive": vader_scores["pos"],
            "negative": vader_scores["neg"],
            "neutral": vader_scores["neu"],
        }

    def get_fear_greed_index(self):
        """Get Fear and Greed Index from Alternative.me"""
        try:
            url = "https://api.alternative.me/fng/"
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                latest_data = data["data"][0]

                return {
                    "fear_greed_value": int(latest_data["value"]),
                    "fear_greed_text": latest_data["value_classification"],
                    "timestamp": datetime.fromtimestamp(int(latest_data["timestamp"])),
                    "source": "alternative.me",
                }

        except Exception as e:
            logging.error(f"Error fetching Fear & Greed Index: {e}")

        return {
            "fear_greed_value": 50,
            "fear_greed_text": "Neutral",
            "source": "default",
        }

    def get_crypto_news_sentiment(self, symbol="bitcoin", limit=10):
        """Get news sentiment for a cryptocurrency"""
        try:
            # Using CoinGecko news (free endpoint)
            url = f"https://api.coingecko.com/api/v3/coins/{symbol}/news"
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                news_data = response.json()

                sentiments = []
                for article in news_data.get("data", [])[:limit]:
                    title = article.get("title", "")
                    description = article.get("description", "")

                    full_text = f"{title}. {description}"
                    sentiment = self.analyze_text_sentiment(full_text)

                    sentiments.append(
                        {
                            "text": full_text[:200] + "..."
                            if len(full_text) > 200
                            else full_text,
                            "sentiment": sentiment["compound"],
                            "confidence": sentiment["confidence"],
                            "timestamp": datetime.now(),
                            "source": "coingecko_news",
                        }
                    )

                # Calculate average sentiment
                if sentiments:
                    avg_sentiment = sum(s["sentiment"] for s in sentiments) / len(
                        sentiments
                    )
                    avg_confidence = sum(s["confidence"] for s in sentiments) / len(
                        sentiments
                    )

                    return {
                        "average_sentiment": avg_sentiment,
                        "confidence": avg_confidence,
                        "article_count": len(sentiments),
                        "articles": sentiments,
                    }

        except Exception as e:
            logging.error(f"Error fetching news sentiment: {e}")

        return {
            "average_sentiment": 0,
            "confidence": 0,
            "article_count": 0,
            "articles": [],
        }

    def aggregate_sentiment_data(self, symbol="BTC"):
        """Aggregate sentiment from multiple sources"""
        results = {"symbol": symbol, "timestamp": datetime.now(), "sources": {}}

        # Get Fear & Greed Index
        fg_data = self.get_fear_greed_index()
        results["sources"]["fear_greed"] = fg_data

        # Get news sentiment
        coin_mapping = {"BTC": "bitcoin", "ETH": "ethereum", "BNB": "binancecoin"}
        coin_id = coin_mapping.get(symbol, "bitcoin")

        news_data = self.get_crypto_news_sentiment(coin_id)
        results["sources"]["news"] = news_data

        # Calculate weighted aggregate sentiment
        weights = {"fear_greed": 0.4, "news": 0.6}

        fg_normalized = (
            fg_data["fear_greed_value"] - 50
        ) / 50  # Convert to -1 to 1 scale
        news_sentiment = news_data["average_sentiment"]

        aggregate_sentiment = (
            fg_normalized * weights["fear_greed"] + news_sentiment * weights["news"]
        )

        results["aggregate_sentiment"] = aggregate_sentiment
        results["confidence"] = (abs(fg_normalized) + news_data["confidence"]) / 2

        return results


# Test the sentiment analyzer
def test_sentiment_analyzer():
    analyzer = CryptoSentimentAnalyzer()

    # Test individual text analysis
    test_texts = [
        "Bitcoin is mooning! HODL to the moon! 🚀",
        "This is terrible FUD. Market is crashing, everyone is rekt!",
        "Moderate growth expected, good fundamentals, DYOR",
        "Whale activity suggests pump incoming, diamond hands needed",
    ]

    print("Testing text sentiment analysis:")
    for text in test_texts:
        sentiment = analyzer.analyze_text_sentiment(text)
        print(f"Text: {text}")
        print(
            f"Sentiment: {sentiment['compound']:.3f}, Confidence: {sentiment['confidence']:.3f}"
        )
        print()

    # Test Fear & Greed Index
    print("Testing Fear & Greed Index:")
    fg_data = analyzer.get_fear_greed_index()
    print(f"Fear & Greed: {fg_data['fear_greed_value']} ({fg_data['fear_greed_text']})")
    print()

    # Test aggregated sentiment
    print("Testing aggregated sentiment for BTC:")
    aggregate = analyzer.aggregate_sentiment_data("BTC")
    print(f"Aggregate sentiment: {aggregate['aggregate_sentiment']:.3f}")
    print(f"Confidence: {aggregate['confidence']:.3f}")


if __name__ == "__main__":
    test_sentiment_analyzer()
