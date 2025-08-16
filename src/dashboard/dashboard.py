import pytz
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import asyncio
import sys
import os
import time  # Add missing import


def parse_timestamp_safely(timestamp_str):
    """Safely parse timestamp strings with various formats"""
    if pd.isna(timestamp_str) or timestamp_str is None:
        return pd.NaT
        
    try:
        # Try ISO8601 format first (handles most cases)
        return pd.to_datetime(timestamp_str, format='ISO8601')
    except:
        try:
            # Fallback to mixed format parsing
            return pd.to_datetime(timestamp_str, format='mixed', dayfirst=False)
        except:
            try:
                # Last resort: try to parse with default settings
                return pd.to_datetime(timestamp_str)
            except:
                # If all else fails, log the problematic timestamp and return NaT
                st.warning(f"Could not parse timestamp: {timestamp_str}")
                return pd.NaT


def safe_datetime_filter(df, timestamp_column, hours_back=24):
    """Safely filter datetime data handling timezone issues"""
    if df.empty or timestamp_column not in df.columns:
        return df

    try:
        # Convert timestamp column to datetime if it's not already
        df_copy = df.copy()
        
        # Handle various timestamp formats including microseconds and timezone info
        if df_copy[timestamp_column].dtype == 'object':
            df_copy[timestamp_column] = df_copy[timestamp_column].apply(parse_timestamp_safely)

        # Get current time
        now = datetime.now()
        cutoff_time = now - timedelta(hours=hours_back)

        # Check if the timestamp column has timezone info
        if df_copy[timestamp_column].dt.tz is not None:
            # Column is timezone-aware, make cutoff_time timezone-aware
            if cutoff_time.tzinfo is None:
                # Assume UTC if no timezone specified
                cutoff_time = pytz.UTC.localize(cutoff_time)
        else:
            # Column is timezone-naive, ensure cutoff_time is also naive
            if hasattr(cutoff_time, "tzinfo") and cutoff_time.tzinfo is not None:
                cutoff_time = cutoff_time.replace(tzinfo=None)

        return df_copy[df_copy[timestamp_column] > cutoff_time]

    except Exception as e:
        st.error(f"Error filtering datetime data: {e}")
        # Return original dataframe if filtering fails
        return df


# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.database import DatabaseManager
from data_pipeline.exchange_collector import ExchangeDataCollector
from data_pipeline.indicators import TechnicalIndicatorCalculator
from sentiment.sentiment_analyzer import CryptoSentimentAnalyzer
from signals.signal_generator import TradingSignalGenerator

# Configure page
st.set_page_config(page_title="Alpha Sentinel Dashboard", page_icon="🤖", layout="wide")

# Initialize session state
if "db_manager" not in st.session_state:
    st.session_state.db_manager = DatabaseManager()
if "last_update" not in st.session_state:
    st.session_state.last_update = None


def load_performance_data():
    """Load bot performance data from database"""
    try:
        # Get recent predictions
        result = (
            st.session_state.db_manager.client.table("ml_predictions")
            .select("*")
            .order("timestamp", desc=True)
            .limit(1000)
            .execute()
        )

        if result.data:
            df = pd.DataFrame(result.data)
            # Use our safe timestamp parser
            df["timestamp"] = df["timestamp"].apply(parse_timestamp_safely)
            return df

    except Exception as e:
        st.error(f"Error loading performance data: {e}")

    return pd.DataFrame()


def load_trading_signals():
    """Load recent trading signals"""
    try:
        result = (
            st.session_state.db_manager.client.table("trading_signals")
            .select("*")
            .order("timestamp", desc=True)
            .execute()
        )

        if result.data:
            df = pd.DataFrame(result.data)
            # Use our safe timestamp parser
            df["timestamp"] = df["timestamp"].apply(parse_timestamp_safely)
            return df

    except Exception as e:
        st.error(f"Error loading trading signals: {e}")

    return pd.DataFrame()


def load_market_data(symbol="BTC/USDT", days=7):
    """Load recent market data"""
    try:
        since_date = datetime.now() - timedelta(days=days)

        result = (
            st.session_state.db_manager.client.table("market_data")
            .select("*")
            .eq("symbol", symbol)
            .gte("timestamp", since_date.isoformat())
            .order("timestamp", desc=False)
            .execute()
        )

        if result.data:
            df = pd.DataFrame(result.data)
            # Use our safe timestamp parser
            df["timestamp"] = df["timestamp"].apply(parse_timestamp_safely)
            return df

    except Exception as e:
        st.error(f"Error loading market data: {e}")

    return pd.DataFrame()


def calculate_accuracy_metrics(predictions_df):
    """Calculate prediction accuracy metrics"""
    if predictions_df.empty:
        return {}

    # Filter predictions that have actual values
    completed_predictions = predictions_df.dropna(subset=["actual_value"])

    if completed_predictions.empty:
        return {
            "total_predictions": len(predictions_df),
            "completed_predictions": 0,
            "accuracy_rate": 0,
            "avg_confidence": predictions_df["confidence_score"].mean()
            if "confidence_score" in predictions_df.columns
            else 0,
        }

    # Calculate accuracy (within 5% threshold)
    price_errors = abs(
        completed_predictions["prediction_value"]
        - completed_predictions["actual_value"]
    )
    relative_errors = price_errors / completed_predictions["actual_value"]
    accurate_predictions = (relative_errors <= 0.05).sum()  # Within 5%

    return {
        "total_predictions": len(predictions_df),
        "completed_predictions": len(completed_predictions),
        "correct_predictions": accurate_predictions,
        "accuracy_rate": accurate_predictions / len(completed_predictions)
        if len(completed_predictions) > 0
        else 0,
        "avg_confidence": predictions_df["confidence_score"].mean()
        if "confidence_score" in predictions_df.columns
        else 0,
        "avg_error_pct": relative_errors.mean() * 100
        if len(completed_predictions) > 0
        else 0,
    }


def main():
    st.title("🤖 Alpha Sentinel Trading Bot Dashboard")
    st.markdown("---")

    # Sidebar
    st.sidebar.title("Dashboard Controls")

    # Symbol selection
    symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "DOGE/USDT"]
    selected_symbol = st.sidebar.selectbox("Select Symbol", symbols)

    # Time range selection
    time_ranges = {"1 Day": 1, "3 Days": 3, "1 Week": 7, "2 Weeks": 14, "1 Month": 30}
    selected_range = st.sidebar.selectbox(
        "Time Range", list(time_ranges.keys()), index=2
    )
    days = time_ranges[selected_range]

    # Auto refresh
    auto_refresh = st.sidebar.checkbox("Auto Refresh (30s)")
    if auto_refresh:
        time.sleep(30)
        st.rerun()

    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    # Main dashboard content
    col1, col2, col3, col4 = st.columns(4)

    # Load data
    predictions_df = load_performance_data()
    signals_df = load_trading_signals()
    market_df = load_market_data(selected_symbol, days)

    # Calculate metrics
    metrics = calculate_accuracy_metrics(predictions_df)

    # Key metrics cards
    with col1:
        st.metric(label="Total Predictions", value=metrics.get("total_predictions", 0))

    with col2:
        accuracy = metrics.get("accuracy_rate", 0)
        st.metric(
            label="Accuracy Rate",
            value=f"{accuracy:.1%}",
            delta=f"{accuracy - 0.5:.1%} vs random" if accuracy > 0 else None,
        )

    with col3:
        st.metric(
            label="Avg Confidence", value=f"{metrics.get('avg_confidence', 0):.3f}"
        )

    with col4:
        # Fix: Use safe datetime filtering
        try:
            recent_signals_df = safe_datetime_filter(signals_df, "timestamp", 24)
            recent_signals = len(recent_signals_df)
        except Exception as e:
            st.error(f"Error calculating recent signals: {e}")
            recent_signals = 0
        st.metric(label="Signals (24h)", value=recent_signals)

    # Charts section
    st.markdown("## 📈 Market Analysis")

    if not market_df.empty:
        # Price chart with indicators
        fig = make_subplots(
            rows=3,
            cols=1,
            subplot_titles=("Price & Signals", "Volume", "RSI"),
            vertical_spacing=0.1,
            row_heights=[0.6, 0.2, 0.2],
        )

        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=market_df["timestamp"],
                open=market_df["open"],
                high=market_df["high"],
                low=market_df["low"],
                close=market_df["close"],
                name="Price",
            ),
            row=1,
            col=1,
        )

        # Add trading signals if available
        if not signals_df.empty and "symbol" in signals_df.columns:
            symbol_signals = signals_df[signals_df["symbol"] == selected_symbol]

            # Buy signals
            if not symbol_signals.empty:
                buy_signals = symbol_signals[symbol_signals["signal_type"] == "BUY"]
                if not buy_signals.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=buy_signals["timestamp"],
                            y=buy_signals["price_at_signal"],
                            mode="markers",
                            marker=dict(color="green", size=10, symbol="triangle-up"),
                            name="Buy Signals",
                        ),
                        row=1,
                        col=1,
                    )

                # Sell signals
                sell_signals = symbol_signals[symbol_signals["signal_type"] == "SELL"]
                if not sell_signals.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=sell_signals["timestamp"],
                            y=sell_signals["price_at_signal"],
                            mode="markers",
                            marker=dict(color="red", size=10, symbol="triangle-down"),
                            name="Sell Signals",
                        ),
                        row=1,
                        col=1,
                    )

        # Volume
        if "volume" in market_df.columns:
            fig.add_trace(
                go.Bar(
                    x=market_df["timestamp"],
                    y=market_df["volume"],
                    name="Volume",
                    marker_color="lightblue",
                ),
                row=2,
                col=1,
            )

        # RSI (if available in technical indicators)
        # For now, we'll create a placeholder RSI
        if "rsi" in market_df.columns or len(market_df) > 14:
            # Calculate simple RSI if not available
            if "rsi" not in market_df.columns and "close" in market_df.columns:
                delta = market_df["close"].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                market_df["rsi"] = rsi

            if "rsi" in market_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=market_df["timestamp"],
                        y=market_df["rsi"],
                        name="RSI",
                        line=dict(color="purple"),
                    ),
                    row=3,
                    col=1,
                )

                # RSI levels
                fig.add_hline(
                    y=70, line_dash="dash", line_color="red", opacity=0.5, row=3, col=1
                )
                fig.add_hline(
                    y=30,
                    line_dash="dash",
                    line_color="green",
                    opacity=0.5,
                    row=3,
                    col=1,
                )

        fig.update_layout(
            title=f"{selected_symbol} - {selected_range}", height=800, showlegend=True
        )
        fig.update_xaxes(rangeslider_visible=False)

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info(f"No market data available for {selected_symbol}")

    # Performance Analysis Section
    st.markdown("## 🎯 Prediction Performance")

    col1, col2 = st.columns(2)

    with col1:
        if not predictions_df.empty:
            # Accuracy over time
            predictions_df["date"] = predictions_df["timestamp"].dt.date
            daily_accuracy = (
                predictions_df.groupby("date")
                .apply(lambda x: calculate_accuracy_metrics(x)["accuracy_rate"])
                .reset_index()
            )
            daily_accuracy.columns = ["date", "accuracy"]

            fig_acc = px.line(
                daily_accuracy,
                x="date",
                y="accuracy",
                title="Daily Prediction Accuracy",
                labels={"accuracy": "Accuracy Rate", "date": "Date"},
            )
            fig_acc.add_hline(
                y=0.5,
                line_dash="dash",
                line_color="red",
                annotation_text="Random (50%)",
            )
            fig_acc.update_yaxis(tickformat=".1%")

            st.plotly_chart(fig_acc, use_container_width=True)
        else:
            st.info("No prediction data available")

    with col2:
        if not predictions_df.empty and "confidence_score" in predictions_df.columns:
            # Confidence distribution
            fig_conf = px.histogram(
                predictions_df,
                x="confidence_score",
                title="Prediction Confidence Distribution",
                labels={"confidence_score": "Confidence Score", "count": "Count"},
            )

            st.plotly_chart(fig_conf, use_container_width=True)
        else:
            st.info("No confidence data available")

    # Recent Signals Table
    st.markdown("## 📊 Recent Trading Signals")

    if not signals_df.empty:
        # Display recent signals
        recent_signals = signals_df.head(10).copy()
        if "timestamp" in recent_signals.columns:
            recent_signals["timestamp"] = recent_signals["timestamp"].dt.strftime(
                "%Y-%m-%d %H:%M"
            )

        # Style the dataframe
        def color_signal_type(val):
            if val == "BUY":
                return "background-color: #90EE90"
            elif val == "SELL":
                return "background-color: #FFB6C1"
            else:
                return "background-color: #FFFFE0"

        # Check which columns exist before displaying
        available_columns = []
        desired_columns = [
            "timestamp",
            "symbol",
            "signal_type",
            "confidence",
            "price_at_signal",
            "status",
        ]

        for col in desired_columns:
            if col in recent_signals.columns:
                available_columns.append(col)

        if available_columns:
            styled_signals = recent_signals[available_columns]
            if "signal_type" in available_columns:
                styled_signals = styled_signals.style.applymap(
                    color_signal_type, subset=["signal_type"]
                )
            st.dataframe(styled_signals, use_container_width=True)
        else:
            st.dataframe(recent_signals, use_container_width=True)
    else:
        st.info("No trading signals available")

    # Bot Statistics
    st.markdown("## 📈 Bot Statistics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Prediction Stats")
        if metrics:
            st.write(f"**Total Predictions:** {metrics.get('total_predictions', 0)}")
            st.write(f"**Completed:** {metrics.get('completed_predictions', 0)}")
            st.write(f"**Correct:** {metrics.get('correct_predictions', 0)}")
            st.write(f"**Accuracy:** {metrics.get('accuracy_rate', 0):.1%}")
            st.write(f"**Avg Error:** {metrics.get('avg_error_pct', 0):.1f}%")

    with col2:
        st.subheader("Signal Stats")
        if not signals_df.empty:
            if "signal_type" in signals_df.columns:
                signal_counts = signals_df["signal_type"].value_counts()
                st.write(f"**Total Signals:** {len(signals_df)}")
                for signal_type, count in signal_counts.items():
                    st.write(f"**{signal_type}:** {count}")

            if "confidence" in signals_df.columns:
                avg_confidence = signals_df["confidence"].mean()
                st.write(f"**Avg Confidence:** {avg_confidence:.3f}")
        else:
            st.write("**Total Signals:** 0")

    with col3:
        st.subheader("System Status")
        st.write(f"**Last Update:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        st.write(f"**Selected Symbol:** {selected_symbol}")
        st.write(f"**Time Range:** {selected_range}")

        # System health indicators
        if metrics.get("total_predictions", 0) > 0:
            st.success("🟢 Predictions Active")
        else:
            st.warning("🟡 No Recent Predictions")

        if len(signals_df) > 0:
            st.success("🟢 Signals Active")
        else:
            st.warning("🟡 No Recent Signals")

    # Footer
    st.markdown("---")
    st.markdown("**Alpha Sentinel Dashboard** - AI-Powered Cryptocurrency Trading Bot")
    st.markdown(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")


if __name__ == "__main__":
    main()
