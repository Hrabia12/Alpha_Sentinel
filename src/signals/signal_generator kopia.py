import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Dict, List, Optional
from enum import Enum


class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class TradingSignalGenerator:
    def __init__(self):
        self.min_confidence = 0.6  # Minimum confidence for signal generation
        self.rsi_oversold = 30
        self.rsi_overbought = 70

    def analyze_technical_indicators(self, indicators: Dict) -> Dict:
        """Analyze technical indicators and generate technical score"""
        technical_signals = []
        reasons = []

        # RSI Analysis
        rsi = indicators.get("rsi")
        if rsi is not None:
            if rsi < self.rsi_oversold:
                technical_signals.append(0.8)  # Strong buy signal
                reasons.append(f"RSI oversold ({rsi:.1f})")
            elif rsi > self.rsi_overbought:
                technical_signals.append(-0.8)  # Strong sell signal
                reasons.append(f"RSI overbought ({rsi:.1f})")
            elif rsi < 40:
                technical_signals.append(0.4)  # Weak buy signal
                reasons.append(f"RSI below 40 ({rsi:.1f})")
            elif rsi > 60:
                technical_signals.append(-0.4)  # Weak sell signal
                reasons.append(f"RSI above 60 ({rsi:.1f})")

        # MACD Analysis
        macd = indicators.get("macd")
        macd_signal = indicators.get("macd_signal")
        if macd is not None and macd_signal is not None:
            if macd > macd_signal and macd > 0:
                technical_signals.append(0.6)  # Buy signal
                reasons.append("MACD bullish crossover")
            elif macd < macd_signal and macd < 0:
                technical_signals.append(-0.6)  # Sell signal
                reasons.append("MACD bearish crossover")

        # Bollinger Bands Analysis
        bb_upper = indicators.get("bb_upper")
        bb_lower = indicators.get("bb_lower")
        current_price = indicators.get("close")

        if all(x is not None for x in [bb_upper, bb_lower, current_price]):
            bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)

            if bb_position < 0.1:
                technical_signals.append(0.7)  # Strong buy - price near lower band
                reasons.append("Price near Bollinger lower band")
            elif bb_position > 0.9:
                technical_signals.append(-0.7)  # Strong sell - price near upper band
                reasons.append("Price near Bollinger upper band")

        # Stochastic Analysis
        stoch_k = indicators.get("stoch_k")
        stoch_d = indicators.get("stoch_d")
        if stoch_k is not None and stoch_d is not None:
            if stoch_k < 20 and stoch_d < 20:
                technical_signals.append(0.5)  # Buy signal
                reasons.append("Stochastic oversold")
            elif stoch_k > 80 and stoch_d > 80:
                technical_signals.append(-0.5)  # Sell signal
                reasons.append("Stochastic overbought")

        # Calculate average technical score
        technical_score = np.mean(technical_signals) if technical_signals else 0

        return {
            "technical_score": technical_score,
            "signals_count": len(technical_signals),
            "reasons": reasons,
            "individual_signals": technical_signals,
        }

    def analyze_ml_prediction(self, prediction_data: Dict) -> Dict:
        """Analyze ML model prediction"""
        if not prediction_data:
            return {
                "ml_score": 0,
                "confidence": 0,
                "reason": "No ML prediction available",
            }

        predicted_price = prediction_data.get("predicted_price")
        current_price = prediction_data.get("current_price")
        confidence = prediction_data.get("confidence", 0.5)

        if predicted_price is None or current_price is None:
            return {"ml_score": 0, "confidence": 0, "reason": "Invalid prediction data"}

        # Calculate price change percentage
        price_change_pct = (predicted_price - current_price) / current_price

        # Convert to signal strength
        ml_score = np.tanh(price_change_pct * 10)  # Scale and normalize to [-1, 1]

        reason = f"ML predicts {price_change_pct * 100:.1f}% price change"

        return {
            "ml_score": ml_score,
            "confidence": confidence,
            "reason": reason,
            "predicted_price": predicted_price,
            "price_change_pct": price_change_pct,
        }

    def analyze_sentiment(self, sentiment_data: Dict) -> Dict:
        """Analyze sentiment data"""
        if not sentiment_data:
            return {
                "sentiment_score": 0,
                "confidence": 0,
                "reason": "No sentiment data",
            }

        aggregate_sentiment = sentiment_data.get("aggregate_sentiment", 0)
        confidence = sentiment_data.get("confidence", 0)

        # Sentiment already normalized to [-1, 1], but we can adjust its impact
        sentiment_score = aggregate_sentiment * 0.7  # Reduce sentiment impact slightly

        reason = f"Market sentiment: {sentiment_score:.2f}"
        if aggregate_sentiment > 0.3:
            reason += " (Very Positive)"
        elif aggregate_sentiment > 0.1:
            reason += " (Positive)"
        elif aggregate_sentiment < -0.3:
            reason += " (Very Negative)"
        elif aggregate_sentiment < -0.1:
            reason += " (Negative)"
        else:
            reason += " (Neutral)"

        return {
            "sentiment_score": sentiment_score,
            "confidence": confidence,
            "reason": reason,
        }

    def generate_trading_signal(
        self,
        symbol: str,
        current_price: float,
        technical_indicators: Dict,
        ml_prediction: Optional[Dict] = None,
        sentiment_data: Optional[Dict] = None,
    ) -> Dict:
        """Generate comprehensive trading signal"""

        # Analyze each component
        technical_analysis = self.analyze_technical_indicators(technical_indicators)
        ml_analysis = self.analyze_ml_prediction(ml_prediction)
        sentiment_analysis = self.analyze_sentiment(sentiment_data)

        # Weight the different components
        weights = {"technical": 0.5, "ml": 0.3, "sentiment": 0.2}

        # Calculate weighted signal
        weighted_signal = (
            technical_analysis["technical_score"] * weights["technical"]
            + ml_analysis["ml_score"] * weights["ml"]
            + sentiment_analysis["sentiment_score"] * weights["sentiment"]
        )

        # Calculate overall confidence
        confidence_scores = []
        if technical_analysis["signals_count"] > 0:
            confidence_scores.append(min(1.0, technical_analysis["signals_count"] / 5))
        if ml_analysis["confidence"] > 0:
            confidence_scores.append(ml_analysis["confidence"])
        if sentiment_analysis["confidence"] > 0:
            confidence_scores.append(sentiment_analysis["confidence"])

        overall_confidence = np.mean(confidence_scores) if confidence_scores else 0

        # Determine signal type
        if weighted_signal > 0.2 and overall_confidence > self.min_confidence:
            signal_type = SignalType.BUY
        elif weighted_signal < -0.2 and overall_confidence > self.min_confidence:
            signal_type = SignalType.SELL
        else:
            signal_type = SignalType.HOLD

        # Calculate stop loss and take profit levels
        atr = technical_indicators.get(
            "atr", current_price * 0.02
        )  # Default 2% if no ATR

        if signal_type == SignalType.BUY:
            stop_loss = current_price - (atr * 2)
            take_profit = current_price + (atr * 3)  # 1.5:1 risk-reward ratio
        elif signal_type == SignalType.SELL:
            stop_loss = current_price + (atr * 2)
            take_profit = current_price - (atr * 3)
        else:
            stop_loss = None
            take_profit = None

        # Compile all reasons
        all_reasons = []
        all_reasons.extend(technical_analysis.get("reasons", []))
        if ml_analysis.get("reason"):
            all_reasons.append(ml_analysis["reason"])
        if sentiment_analysis.get("reason"):
            all_reasons.append(sentiment_analysis["reason"])

        return {
            "timestamp": datetime.now(),
            "symbol": symbol,
            "signal_type": signal_type.value,
            "confidence": overall_confidence,
            "signal_strength": weighted_signal,
            "current_price": current_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "reasons": all_reasons,
            "components": {
                "technical": technical_analysis,
                "ml": ml_analysis,
                "sentiment": sentiment_analysis,
            },
            "risk_reward_ratio": 1.5 if stop_loss and take_profit else None,
        }


# Test the signal generator
def test_signal_generator():
    generator = TradingSignalGenerator()

    # Sample technical indicators
    indicators = {
        "close": 45000,
        "rsi": 25,  # Oversold
        "macd": 150,
        "macd_signal": 120,  # Bullish crossover
        "bb_upper": 46000,
        "bb_lower": 44000,  # Price near lower band
        "stoch_k": 18,  # Oversold
        "stoch_d": 20,
        "atr": 800,
    }

    # Sample ML prediction
    ml_prediction = {
        "predicted_price": 46500,
        "current_price": 45000,
        "confidence": 0.75,
    }

    # Sample sentiment data
    sentiment_data = {
        "aggregate_sentiment": 0.3,  # Positive
        "confidence": 0.6,
    }

    # Generate signal
    signal = generator.generate_trading_signal(
        symbol="BTC/USDT",
        current_price=45000,
        technical_indicators=indicators,
        ml_prediction=ml_prediction,
        sentiment_data=sentiment_data,
    )

    print("Generated Trading Signal:")
    print(f"Symbol: {signal['symbol']}")
    print(f"Signal: {signal['signal_type']}")
    print(f"Confidence: {signal['confidence']:.3f}")
    print(f"Signal Strength: {signal['signal_strength']:.3f}")
    print(f"Current Price: ${signal['current_price']:,.2f}")

    if signal["stop_loss"]:
        print(f"Stop Loss: ${signal['stop_loss']:,.2f}")
    if signal["take_profit"]:
        print(f"Take Profit: ${signal['take_profit']:,.2f}")

    print(f"Risk/Reward Ratio: {signal.get('risk_reward_ratio', 'N/A')}")
    print("\nReasons:")
    for reason in signal["reasons"]:
        print(f"  - {reason}")


if __name__ == "__main__":
    test_signal_generator()
