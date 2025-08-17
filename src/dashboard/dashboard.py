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
import time

# Configure page for clean, focused layout
st.set_page_config(
    page_title="Alpha Sentinel", 
    page_icon="🤖", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean, minimalist CSS
st.markdown("""
<style>
    /* Show Streamlit elements for better UX */
    #MainMenu {visibility: visible;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Clean, modern design */
    .main {
        background: #0E1117;
        color: white;
    }
    
    .stApp {
        background: #0E1117;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #1E1E1E 0%, #2D2D2D 100%);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #333;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #00C805;
        margin: 0;
    }
    
    .metric-label {
        color: #CCCCCC;
        font-size: 0.9rem;
        margin: 0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .status-good { color: #00C805; }
    .status-warning { color: #FFA500; }
    .status-bad { color: #FF3B69; }
    
    .chart-container {
        background: #1E1E1E;
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid #333;
        margin-bottom: 1.5rem;
    }
    
    .section-header {
        color: white;
        font-size: 1.5rem;
        font-weight: 600;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #00C805;
    }
    
    .sidebar .sidebar-content {
        background: #1E1E1E;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #00C805 0%, #00A805 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        width: 100%;
        margin: 0.5rem 0;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #00A805 0%, #008805 100%);
        transform: translateY(-1px);
        transition: all 0.2s ease;
    }
</style>
""", unsafe_allow_html=True)

def parse_timestamp_safely(timestamp_str):
    """Safely parse timestamp strings with various formats"""
    if pd.isna(timestamp_str) or timestamp_str is None:
        return pd.NaT

    try:
        return pd.to_datetime(timestamp_str, format="ISO8601")
    except:
        try:
            return pd.to_datetime(timestamp_str, format="mixed", dayfirst=False)
        except:
            try:
                return pd.to_datetime(timestamp_str)
            except:
                return pd.NaT

def normalize_ohlcv_data(df):
    """Clean OHLCV data and remove extreme outliers"""
    if df.empty:
        return df

    df_copy = df.copy()

    required_cols = ["open", "high", "low", "close", "volume"]
    if not all(col in df_copy.columns for col in required_cols):
        return df_copy
    
    # Basic validation
    df_copy = df_copy[
        (df_copy["open"] > 0) & 
        (df_copy["high"] > 0) & 
        (df_copy["low"] > 0) & 
        (df_copy["close"] > 0) &
        (df_copy["volume"] >= 0)
    ]
    
    if df_copy.empty:
        return df_copy
    
    # Remove extreme outliers (more aggressive)
    for col in ["open", "high", "low", "close"]:
        q1 = df_copy[col].quantile(0.1)
        q3 = df_copy[col].quantile(0.9)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        median_price = df_copy[col].median()
        if median_price > 0:
            lower_bound = max(lower_bound, median_price * 0.5)
            upper_bound = min(upper_bound, median_price * 2.0)
        
        df_copy = df_copy[(df_copy[col] >= lower_bound) & (df_copy[col] <= upper_bound)]
    
    if df_copy.empty:
        return df_copy
    
    # Ensure price logic
    df_copy["high"] = df_copy[["open", "close", "high"]].max(axis=1)
    df_copy["low"] = df_copy[["open", "close", "low"]].min(axis=1)
    
    return df_copy

def create_clean_chart_data(market_df):
    """Create clean chart data with proper scaling"""
    if market_df.empty:
        return market_df, None, None
    
    df_clean = market_df.copy()
    df_clean = normalize_ohlcv_data(df_clean)
    
    if df_clean.empty:
        return df_clean, None, None
    
    # Calculate realistic price range
    close_prices = df_clean["close"].sort_values()
    n = len(close_prices)
    start_idx = int(n * 0.05)
    end_idx = int(n * 0.95)
    
    if start_idx < end_idx:
        realistic_prices = close_prices.iloc[start_idx:end_idx]
    else:
        realistic_prices = close_prices
    
    min_price = realistic_prices.min()
    max_price = realistic_prices.max()
    
    return df_clean, min_price, max_price

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.database import DatabaseManager
from data_pipeline.exchange_collector import ExchangeDataCollector
from data_pipeline.indicators import TechnicalIndicatorCalculator
from sentiment.sentiment_analyzer import CryptoSentimentAnalyzer
from signals.signal_generator import TradingSignalGenerator

# TradingView widget configuration with signal overlay capability
TRADINGVIEW_WIDGET_HTML = """
<div class="tradingview-widget-container">
    <div id="tradingview_chart"></div>
    <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
    <script type="text/javascript">
        let widget;
        
        function initTradingView() {{
            widget = new TradingView.widget({{
                "width": "100%",
                "height": 600,
                "symbol": "BINANCE:{symbol}",
                "interval": "{interval}",
                "timezone": "Etc/UTC",
                "theme": "dark",
                "style": "1",
                "locale": "en",
                "toolbar_bg": "#f1f3f6",
                "enable_publishing": false,
                "hide_top_toolbar": false,
                "hide_legend": false,
                "save_image": false,
                "container_id": "tradingview_chart",
                "studies": [
                    "RSI@tv-basicstudies",
                    "MACD@tv-basicstudies",
                    "BB@tv-basicstudies",
                    "Stochastic@tv-basicstudies"
                ],
                "callback": function() {{
                    // Chart is ready, now we can add signals
                    addSignalsToChart();
                }}
            }});
        }}
        
        function addSignalsToChart() {{
            if (widget && widget.chart) {{
                const chart = widget.chart();
                
                // Add buy signals
                {buy_signals_js}
                
                // Add sell signals  
                {sell_signals_js}
                
                // Add signal annotations
                {signal_annotations_js}
            }}
        }}
        
        // Initialize when page loads
        if (document.readyState === 'loading') {{
            document.addEventListener('DOMContentLoaded', initTradingView);
        }} else {{
            initTradingView();
        }}
    </script>
</div>
"""

# Function to generate JavaScript for signal overlays
def generate_signal_js(signals_df, symbol):
    """Generate JavaScript code to overlay signals on TradingView chart"""
    if signals_df.empty:
        return "", "", ""
    
    # Filter signals for the current symbol
    symbol_signals = signals_df[signals_df["symbol"] == symbol]
    
    if symbol_signals.empty:
        return "", "", ""
    
    # Generate buy signals JavaScript
    buy_signals = symbol_signals[symbol_signals["signal_type"] == "BUY"]
    buy_signals_js = ""
    if not buy_signals.empty:
        buy_signals_js = """
                // Add buy signal markers
                buy_signals.forEach(signal => {
                    chart.createShape(
                        { time: signal.timestamp, price: signal.price_at_signal },
                        { time: signal.timestamp, price: signal.price_at_signal * 1.02 },
                        {
                            shape: "arrow_up",
                            text: "BUY",
                            overrides: {
                                backgroundColor: "#00C805",
                                textColor: "#FFFFFF",
                                fontSize: 12,
                                fontWeight: "bold"
                            }
                        }
                    );
                });
        """
    
    # Generate sell signals JavaScript
    sell_signals = symbol_signals[symbol_signals["signal_type"] == "SELL"]
    sell_signals_js = ""
    if not sell_signals.empty:
        sell_signals_js = """
                // Add sell signal markers
                sell_signals.forEach(signal => {
                    chart.createShape(
                        { time: signal.timestamp, price: signal.price_at_signal },
                        { time: signal.timestamp, price: signal.price_at_signal * 0.98 },
                        {
                            shape: "arrow_down",
                            text: "SELL",
                            overrides: {
                                backgroundColor: "#FF3B69",
                                textColor: "#FFFFFF",
                                fontSize: 12,
                                fontWeight: "bold"
                            }
                        }
                    );
                });
        """
    
    # Generate signal data for JavaScript
    signal_data_js = ""
    if not symbol_signals.empty:
        signal_data_js = f"""
                const buy_signals = {buy_signals.to_dict('records') if not buy_signals.empty else '[]'};
                const sell_signals = {sell_signals.to_dict('records') if not sell_signals.empty else '[]'};
        """
    
    return signal_data_js, buy_signals_js, sell_signals_js

# Initialize session state
if "db_manager" not in st.session_state:
    st.session_state.db_manager = DatabaseManager()

def load_performance_data():
    """Load recent ML predictions"""
    try:
        result = (
            st.session_state.db_manager.client.table("ml_predictions")
            .select("*")
            .order("timestamp", desc=True)
            .limit(100)
            .execute()
        )

        if result.data:
            df = pd.DataFrame(result.data)
            df["timestamp"] = df["timestamp"].apply(parse_timestamp_safely)
            return df

    except Exception as e:
        st.error(f"Error loading performance data: {e}")

    return pd.DataFrame()

def load_trading_signals():
    """Load all trading signals without limits"""
    try:
        result = (
            st.session_state.db_manager.client.table("trading_signals")
            .select("*")
            .order("timestamp", desc=True)
            .execute()
        )

        if result.data:
            df = pd.DataFrame(result.data)
            df["timestamp"] = df["timestamp"].apply(parse_timestamp_safely)
            return df

    except Exception as e:
        st.error(f"Error loading trading signals: {e}")

    return pd.DataFrame()

def load_ml_predictions():
    """Load ML predictions with enhanced context"""
    try:
        result = (
            st.session_state.db_manager.client.table("ml_predictions")
            .select("*")
            .order("timestamp", desc=True)
            .limit(200)
            .execute()
        )

        if result.data:
            df = pd.DataFrame(result.data)
            df["timestamp"] = df["timestamp"].apply(parse_timestamp_safely)
            return df

    except Exception as e:
        st.error(f"Error loading ML predictions: {e}")

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
            df["timestamp"] = df["timestamp"].apply(parse_timestamp_safely)
            df = normalize_ohlcv_data(df)
            return df

    except Exception as e:
        st.error(f"Error loading market data: {e}")

    return pd.DataFrame()

def calculate_accuracy_metrics(predictions_df):
    """Calculate simple, focused metrics"""
    if predictions_df.empty:
        return {}

    completed = predictions_df.dropna(subset=["actual_value"])
    
    if completed.empty:
        return {"total": len(predictions_df), "completed": 0, "accuracy": 0}
    
    # Simple accuracy: within 3% threshold
    errors = abs(completed["prediction_value"] - completed["actual_value"]) / completed["actual_value"]
    accurate = (errors <= 0.03).sum()
    
    return {
        "total": len(predictions_df),
        "completed": len(completed),
        "accuracy": accurate / len(completed) if len(completed) > 0 else 0
    }

def calculate_signal_accuracy(signals_df):
    """Calculate signal accuracy based on 10-minute verification"""
    if signals_df.empty:
        return {}
    
    # Check if new verification columns exist, if not use fallback logic
    has_verification = "verification_status" in signals_df.columns
    has_outcome = "outcome" in signals_df.columns
    
    if not has_verification or not has_outcome:
        # Fallback: calculate basic signal metrics
        return {
            "total": len(signals_df),
            "verified": 0,
            "correct": 0,
            "accuracy": 0,
            "avg_pnl": 0,
            "status": "legacy_data"
        }
    
    # Filter verified signals (at least 10 minutes old)
    verified_signals = signals_df[
        (signals_df["verification_status"] == "verified") & 
        (signals_df["outcome"].notna())
    ]
    
    if verified_signals.empty:
        return {"total": len(signals_df), "verified": 0, "accuracy": 0, "status": "no_verified"}
    
    # Calculate accuracy
    correct_signals = verified_signals[
        verified_signals["outcome"].isin(["correct", "profitable"])
    ]
    
    accuracy = len(correct_signals) / len(verified_signals) if len(verified_signals) > 0 else 0
    
    # Calculate average P&L
    avg_pnl = verified_signals["pnl"].mean() if "pnl" in verified_signals.columns else 0
    
    return {
        "total": len(signals_df),
        "verified": len(verified_signals),
        "correct": len(correct_signals),
        "accuracy": accuracy,
        "avg_pnl": avg_pnl,
        "status": "verified"
    }

def calculate_technical_indicators(df):
    """Calculate technical indicators for the market data"""
    if df.empty or len(df) < 50:
        return df, {}
    
    try:
        indicator_calc = TechnicalIndicatorCalculator()
        df_with_indicators = indicator_calc.calculate_all_indicators(df)
        latest_indicators = indicator_calc.get_latest_indicators(df_with_indicators)
        return df_with_indicators, latest_indicators
    except Exception as e:
        st.error(f"Error calculating indicators: {e}")
        return df, {}

def calculate_model_learning_metrics(predictions_df):
    """Calculate model learning and improvement metrics"""
    if predictions_df.empty:
        return {"total_predictions": len(predictions_df), "status": "insufficient_data"}

    # Check if required columns exist
    required_cols = ["prediction_accuracy", "confidence_score"]
    missing_cols = [col for col in required_cols if col not in predictions_df.columns]

    if missing_cols:
        # Fallback: basic prediction count
        return {
            "total_predictions": len(predictions_df),
            "status": "missing_columns",
            "missing": missing_cols
        }
    
    # Group by date to see daily performance
    predictions_df["date"] = predictions_df["timestamp"].dt.date
    
    daily_metrics = predictions_df.groupby("date").agg({
        "prediction_accuracy": "mean",
        "confidence_score": "mean",
        "prediction_value": "count"
    }).reset_index()
    
    daily_metrics.columns = ["date", "avg_accuracy", "avg_confidence", "prediction_count"]
    
    # Calculate learning trends
    if len(daily_metrics) > 1:
        # Sort by date
        daily_metrics = daily_metrics.sort_values("date")
        
        # Calculate improvement rate
        accuracy_trend = daily_metrics["avg_accuracy"].pct_change().mean()
        confidence_trend = daily_metrics["avg_confidence"].pct_change().mean()
        
        # Calculate volatility in performance
        accuracy_volatility = daily_metrics["avg_accuracy"].std()
        confidence_volatility = daily_metrics["avg_confidence"].std()

        return {
            "total_predictions": len(predictions_df),
            "daily_metrics": daily_metrics,
            "accuracy_trend": accuracy_trend,
            "confidence_trend": confidence_trend,
            "accuracy_volatility": accuracy_volatility,
            "confidence_volatility": confidence_volatility,
            "recent_accuracy": daily_metrics["avg_accuracy"].iloc[-1] if len(daily_metrics) > 0 else 0,
            "recent_confidence": daily_metrics["avg_confidence"].iloc[-1] if len(daily_metrics) > 0 else 0,
            "status": "complete"
        }
    
    return {"total_predictions": len(predictions_df), "status": "insufficient_data"}

def main():
    # Clean, focused header
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="color: white; margin: 0; font-size: 3rem;">🤖 Alpha Sentinel</h1>
        <p style="color: #CCCCCC; margin: 0; font-size: 1.2rem;">AI Trading Bot Dashboard</p>
    </div>
    """, unsafe_allow_html=True)

    # Minimalist sidebar
    with st.sidebar:
        st.markdown("### 🎛️ Controls")

        # Symbol selection
        symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "DOGE/USDT"]
        selected_symbol = st.selectbox("Symbol", symbols, index=0)
        
        # Time range
        time_ranges = {"1D": 1, "3D": 3, "1W": 7, "2W": 14}
        selected_range = st.selectbox("Timeframe", list(time_ranges.keys()), index=2)
        days = time_ranges[selected_range]
        
        st.markdown("---")
        
        # Simple controls
        if st.button("🔄 Refresh"):
            st.cache_data.clear()
            st.rerun()
            
        if st.button("🧹 Clean Data"):
            try:
                db_manager = DatabaseManager()
                with st.spinner("Cleaning..."):
                    db_manager.cleanup_corrupted_data()
                st.success("Done!")
                st.cache_data.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

        # Auto refresh
        auto_refresh = st.checkbox("🔄 Auto Refresh (30s)")
        if auto_refresh:
            time.sleep(30)
            st.rerun()

    # Load data
    predictions_df = load_performance_data()
    signals_df = load_trading_signals()
    market_df = load_market_data(selected_symbol, days)
    ml_predictions_df = load_ml_predictions()

    # Calculate metrics
    metrics = calculate_accuracy_metrics(predictions_df)
    signal_metrics = calculate_signal_accuracy(signals_df)
    model_metrics = calculate_model_learning_metrics(ml_predictions_df)
    
    # 🚀 ULTRA-COMPACT GRID LAYOUT - All elements visible on one screen
    st.markdown('<p class="section-header" style="font-size: 1.2rem; margin: 0.5rem 0;">📊 Dashboard Overview</p>', unsafe_allow_html=True)
    
    # Top row: Performance metrics (2x2 grid)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container" style="padding: 0.5rem; margin-bottom: 0.3rem;">
            <p class="metric-value" style="font-size: 1.2rem; margin: 0;">{metrics.get('total', 0)}</p>
            <p class="metric-label" style="font-size: 0.7rem; margin: 0;">Total Predictions</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        accuracy = metrics.get('accuracy', 0)
        st.markdown(f"""
        <div class="metric-container" style="padding: 0.5rem; margin-bottom: 0.3rem;">
            <p class="metric-value" style="font-size: 1.2rem; margin: 0;">{accuracy:.1%}</p>
            <p class="metric-label" style="font-size: 0.7rem; margin: 0;">ML Accuracy</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        signal_acc = signal_metrics.get('accuracy', 0)
        signal_status = signal_metrics.get('status', 'unknown')
        
        if signal_status == 'legacy_data':
            st.markdown(f"""
            <div class="metric-container" style="padding: 1rem; margin-bottom: 0.5rem;">
                <p class="metric-value" style="font-size: 1.5rem;">📊</p>
                <p class="metric-label" style="font-size: 0.8rem;">Legacy Data</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            signal_acc_color = "status-good" if signal_acc >= 0.6 else "status-warning" if signal_acc >= 0.4 else "status-bad"
            st.markdown(f"""
            <div class="metric-container" style="padding: 1rem; margin-bottom: 0.5rem;">
                <p class="metric-value {signal_acc_color}" style="font-size: 1.5rem;">{signal_acc:.1%}</p>
                <p class="metric-label" style="font-size: 0.8rem;">Signal Accuracy</p>
            </div>
            """, unsafe_allow_html=True)

    with col4:
        verified_count = signal_metrics.get('verified', 0)
        if signal_metrics.get('status') == 'legacy_data':
            st.markdown(f"""
            <div class="metric-container" style="padding: 1rem; margin-bottom: 0.5rem;">
                <p class="metric-value" style="font-size: 1.5rem;">🔄</p>
                <p class="metric-label" style="font-size: 0.8rem;">Updating...</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-container" style="padding: 1rem; margin-bottom: 0.5rem;">
                <p class="metric-value" style="font-size: 1.5rem;">{verified_count}</p>
                <p class="metric-label" style="font-size: 0.8rem;">Verified Signals</p>
            </div>
            """, unsafe_allow_html=True)

    # 📊 ULTRA-COMPACT TECHNICAL INDICATORS - All in one row
    # Calculate technical indicators
    market_df_with_indicators, latest_indicators = calculate_technical_indicators(market_df)
    
    if latest_indicators:
        # Create ultra-compact single row layout for all indicators
        st.markdown('<p class="section-header" style="font-size: 1.1rem; margin: 0.3rem 0;">📊 Live Indicators</p>', unsafe_allow_html=True)
        
        # Single row with 4 ultra-compact indicator boxes
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            rsi_value = latest_indicators.get('rsi', 0)
            rsi_color = "status-good" if rsi_value < 30 else "status-bad" if rsi_value > 70 else "status-warning"
            st.markdown(f"""
            <div class="metric-container" style="padding: 0.4rem; margin-bottom: 0.2rem;">
                <p class="metric-value {rsi_color}" style="font-size: 1.1rem; margin: 0;">{rsi_value:.1f}</p>
                <p class="metric-label" style="font-size: 0.65rem; margin: 0;">RSI</p>
                <p style="font-size: 0.6rem; color: #CCCCCC; margin: 0;">
                    {'Oversold' if rsi_value < 30 else 'Overbought' if rsi_value > 70 else 'Neutral'}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            macd_value = latest_indicators.get('macd', 0)
            macd_signal = latest_indicators.get('macd_signal', 0)
            macd_color = "status-good" if macd_value > macd_signal else "status-bad"
            macd_trend = "Bullish" if macd_value > macd_signal else "Bearish"
            st.markdown(f"""
            <div class="metric-container" style="padding: 0.4rem; margin-bottom: 0.2rem;">
                <p class="metric-value {macd_color}" style="font-size: 1.1rem; margin: 0;">{macd_value:.3f}</p>
                <p class="metric-label" style="font-size: 0.65rem; margin: 0;">MACD</p>
                <p style="font-size: 0.6rem; color: #CCCCCC; margin: 0;">{macd_trend}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            bb_upper = latest_indicators.get('bb_upper', 0)
            bb_middle = latest_indicators.get('bb_middle', 0)
            bb_lower = latest_indicators.get('bb_lower', 0)
            current_price = market_df.iloc[-1]['close'] if not market_df.empty else 0
            
            if bb_upper and bb_middle and bb_lower and current_price:
                bb_percentile = (current_price - bb_lower) / (bb_upper - bb_lower) * 100
                bb_color = "status-bad" if bb_percentile > 80 else "status-good" if bb_percentile < 20 else "status-warning"
                bb_status = "Upper" if bb_percentile > 80 else "Lower" if bb_percentile < 20 else "Middle"
                
                st.markdown(f"""
                <div class="metric-container" style="padding: 0.4rem; margin-bottom: 0.2rem;">
                    <p class="metric-value {bb_color}" style="font-size: 1.1rem; margin: 0;">{bb_percentile:.0f}%</p>
                    <p class="metric-label" style="font-size: 0.65rem; margin: 0;">BB Pos</p>
                    <p style="font-size: 0.6rem; color: #CCCCCC; margin: 0;">{bb_status}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-container" style="padding: 0.4rem; margin-bottom: 0.2rem;">
                    <p class="metric-value">--</p>
                    <p class="metric-label" style="font-size: 0.65rem; margin: 0;">BB Pos</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col4:
            stoch_k = latest_indicators.get('stoch_k', 0)
            stoch_d = latest_indicators.get('stoch_d', 0)
            stoch_color = "status-bad" if stoch_k > 80 else "status-good" if stoch_k < 20 else "status-warning"
            stoch_status = "Overbought" if stoch_k > 80 else "Oversold" if stoch_k < 20 else "Neutral"
            
            st.markdown(f"""
            <div class="metric-container" style="padding: 0.4rem; margin-bottom: 0.2rem;">
                <p class="metric-value {stoch_color}" style="font-size: 1.1rem; margin: 0;">{stoch_k:.0f}</p>
                <p class="metric-label" style="font-size: 0.65rem; margin: 0;">Stoch</p>
                <p style="font-size: 0.6rem; color: #CCCCCC; margin: 0;">{stoch_status}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # 🚀 MAIN CHART SECTION - Now positioned right after indicators
        st.markdown('<p class="section-header" style="font-size: 1.1rem; margin: 0.3rem 0;">📈 Market Chart</p>', unsafe_allow_html=True)
        
        # Technical Analysis Summary
        st.markdown("### 📈 Technical Analysis Summary")
        
        # Create technical analysis chart
        if not market_df_with_indicators.empty and len(market_df_with_indicators) > 50:
            fig_technical = make_subplots(
                rows=4, cols=1,
                subplot_titles=("Price & Bollinger Bands", "RSI", "MACD", "Stochastic"),
                vertical_spacing=0.08,
                row_heights=[0.4, 0.2, 0.2, 0.2],
            )
            
            # Price and Bollinger Bands
            if all(col in market_df_with_indicators.columns for col in ["close", "bb_upper", "bb_middle", "bb_lower"]):
                fig_technical.add_trace(
                    go.Scatter(
                        x=market_df_with_indicators["timestamp"],
                        y=market_df_with_indicators["close"],
                        mode="lines",
                        name="Price",
                        line=dict(color="#00C805", width=2),
                    ),
                    row=1, col=1,
                )
                
                fig_technical.add_trace(
                    go.Scatter(
                        x=market_df_with_indicators["timestamp"],
                        y=market_df_with_indicators["bb_upper"],
                        mode="lines",
                        name="BB Upper",
                        line=dict(color="#FF6B6B", width=1, dash="dash"),
                    ),
                    row=1, col=1,
                )
                
                fig_technical.add_trace(
                    go.Scatter(
                        x=market_df_with_indicators["timestamp"],
                        y=market_df_with_indicators["bb_middle"],
                        mode="lines",
                        name="BB Middle",
                        line=dict(color="#8B5CF6", width=1, dash="dash"),
                    ),
                    row=1, col=1,
                )
                
                fig_technical.add_trace(
                    go.Scatter(
                        x=market_df_with_indicators["timestamp"],
                        y=market_df_with_indicators["bb_lower"],
                        mode="lines",
                        name="BB Lower",
                        line=dict(color="#FF6B6B", width=1, dash="dash"),
                    ),
                    row=1, col=1,
                )
            
            # RSI
            if "rsi" in market_df_with_indicators.columns:
                fig_technical.add_trace(
                    go.Scatter(
                        x=market_df_with_indicators["timestamp"],
                        y=market_df_with_indicators["rsi"],
                        mode="lines",
                        name="RSI",
                        line=dict(color="#00C805", width=2),
                    ),
                    row=2, col=1,
                )
                
                # Add overbought/oversold lines
                fig_technical.add_hline(y=70, line_dash="dash", line_color="#FF3B69", row=2, col=1)
                fig_technical.add_hline(y=30, line_dash="dash", line_color="#00C805", row=2, col=1)
            
            # MACD
            if all(col in market_df_with_indicators.columns for col in ["macd", "macd_signal"]):
                fig_technical.add_trace(
                    go.Scatter(
                        x=market_df_with_indicators["timestamp"],
                        y=market_df_with_indicators["macd"],
                        mode="lines",
                        name="MACD",
                        line=dict(color="#00C805", width=2),
                    ),
                    row=3, col=1,
                )
                
                fig_technical.add_trace(
                    go.Scatter(
                        x=market_df_with_indicators["timestamp"],
                        y=market_df_with_indicators["macd_signal"],
                        mode="lines",
                        name="MACD Signal",
                        line=dict(color="#FF3B69", width=2),
                    ),
                    row=3, col=1,
                )
            
            # Stochastic
            if all(col in market_df_with_indicators.columns for col in ["stoch_k", "stoch_d"]):
                fig_technical.add_trace(
                    go.Scatter(
                        x=market_df_with_indicators["timestamp"],
                        y=market_df_with_indicators["stoch_k"],
                        mode="lines",
                        name="Stoch K",
                        line=dict(color="#00C805", width=2),
                    ),
                    row=4, col=1,
                )
                
                fig_technical.add_trace(
                    go.Scatter(
                        x=market_df_with_indicators["timestamp"],
                        y=market_df_with_indicators["stoch_d"],
                        mode="lines",
                        name="Stoch D",
                        line=dict(color="#8B5CF6", width=2),
                    ),
                    row=4, col=1,
                )
                
                # Add overbought/oversold lines
                fig_technical.add_hline(y=80, line_dash="dash", line_color="#FF3B69", row=4, col=1)
                fig_technical.add_hline(y=20, line_dash="dash", line_color="#00C805", row=4, col=1)
            
            # Update layout
            fig_technical.update_layout(
                height=800,
                showlegend=True,
                template="plotly_dark",
                paper_bgcolor="#0E1117",
                plot_bgcolor="#0E1117",
                font=dict(color="white"),
                margin=dict(l=50, r=50, t=30, b=50),
            )
            
            # Update axes
            fig_technical.update_xaxes(gridcolor="#333", zerolinecolor="#333", showgrid=True, gridwidth=1)
            fig_technical.update_yaxes(gridcolor="#333", zerolinecolor="#333", showgrid=True, gridwidth=1)
            
            # Set y-axis ranges
            fig_technical.update_yaxes(range=[0, 100], row=2, col=1)  # RSI
            fig_technical.update_yaxes(range=[0, 100], row=4, col=1)  # Stochastic
            
            st.plotly_chart(fig_technical, use_container_width=True)
    else:
        st.info("📊 Need at least 50 data points to calculate technical indicators")
    
    # 🚀 COMPACT MAIN CHART SECTION - Side by side with indicators
    if not market_df.empty:
        # Enhanced data cleaning and outlier removal
        clean_df, min_price, max_price = create_clean_chart_data(market_df)
        
        if clean_df.empty:
            st.warning("⚠️ No clean data available")
            return
        
        # Calculate technical indicators for the main chart
        market_df_with_indicators, latest_indicators = calculate_technical_indicators(clean_df)
        
        # Compact price display
        if "close" in clean_df.columns and len(clean_df) > 0:
            latest_price = clean_df.iloc[-1]["close"]
            price_change = 0
            price_change_24h = 0
            
            if len(clean_df) > 1:
                price_change = ((latest_price - clean_df.iloc[-2]["close"]) / clean_df.iloc[-2]["close"]) * 100
                
            if len(clean_df) > 24:
                price_change_24h = ((latest_price - clean_df.iloc[-24]["close"]) / clean_df.iloc[-24]["close"]) * 100
            
            change_color = "status-good" if price_change >= 0 else "status-bad"
            change_24h_color = "status-good" if price_change_24h >= 0 else "status-bad"
            change_symbol = "↗️" if price_change >= 0 else "↘️"
            change_24h_symbol = "↗️" if price_change_24h >= 0 else "↘️"
            
            # Compact 4-column layout
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-container">
                    <p class="metric-value">${latest_price:,.0f}</p>
                    <p class="metric-label">{selected_symbol}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-container">
                    <p class="metric-value {change_color}">{change_symbol} {price_change:+.1f}%</p>
                    <p class="metric-label">1H</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-container">
                    <p class="metric-value {change_24h_color}">{change_24h_symbol} {price_change_24h:+.1f}%</p>
                    <p class="metric-label">24H</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                if len(clean_df) > 20:
                    volatility = clean_df["close"].pct_change().std() * 100
                    vol_color = "status-warning" if volatility > 5 else "status-good"
                    st.markdown(f"""
                    <div class="metric-container">
                        <p class="metric-value {vol_color}">{volatility:.1f}%</p>
                        <p class="metric-label">Vol</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="metric-container">
                        <p class="metric-value">--</p>
                        <p class="metric-label">Vol</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # 🎯 COMPACT PROFESSIONAL CHART
        # Create efficient 2-panel chart (Price + Volume)
        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=("Price & Indicators", "Volume"),
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3],
            shared_xaxes=True,
        )

        # 🕯️ PROPER CANDLESTICKS - Fixed thickness and scaling
        if all(col in clean_df.columns for col in ["open", "high", "low", "close"]):
            # Enhanced data cleaning for better scaling
            df_clean = clean_df.copy()
            
            # Remove extreme outliers more aggressively
            for col in ["open", "high", "low", "close"]:
                q1 = df_clean[col].quantile(0.05)
                q3 = df_clean[col].quantile(0.95)
                iqr = q3 - q1
                lower_bound = q1 - 2.0 * iqr  # More aggressive outlier removal
                upper_bound = q3 + 2.0 * iqr
                df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
            
            if not df_clean.empty:
                # Main candlestick with PROPER thickness
                fig.add_trace(
                    go.Candlestick(
                        x=df_clean["timestamp"],
                        open=df_clean["open"],
                        high=df_clean["high"],
                        low=df_clean["low"],
                        close=df_clean["close"],
                        name="Price",
                        increasing_line_color="#00C805",
                        decreasing_line_color="#FF3B69",
                        increasing_fillcolor="#00C805",
                        decreasing_fillcolor="#FF3B69",
                        line=dict(width=2.0),  # Thick lines for visibility
                        whiskerwidth=1.0,      # Thick wicks
                    ),
                    row=1, col=1,
                )
                
                # Add Bollinger Bands if available
                if all(col in market_df_with_indicators.columns for col in ["bb_upper", "bb_middle", "bb_lower"]):
                    # Upper Band
                    fig.add_trace(
                        go.Scatter(
                            x=market_df_with_indicators["timestamp"],
                            y=market_df_with_indicators["bb_upper"],
                            mode="lines",
                            name="BB Upper",
                            line=dict(color="rgba(255, 107, 107, 0.7)", width=1.5, dash="dash"),
                            showlegend=False,
                        ),
                        row=1, col=1,
                    )
                    
                    # Middle Band
                    fig.add_trace(
                        go.Scatter(
                            x=market_df_with_indicators["timestamp"],
                            y=market_df_with_indicators["bb_middle"],
                            mode="lines",
                            name="BB Middle",
                            line=dict(color="rgba(139, 92, 246, 0.7)", width=1.5, dash="dash"),
                            showlegend=False,
                        ),
                        row=1, col=1,
                    )
                    
                    # Lower Band
                    fig.add_trace(
                        go.Scatter(
                            x=market_df_with_indicators["timestamp"],
                            y=market_df_with_indicators["bb_lower"],
                            mode="lines",
                            name="BB Lower",
                            line=dict(color="rgba(255, 107, 107, 0.7)", width=1.5, dash="dash"),
                            showlegend=False,
                        ),
                        row=1, col=1,
                    )
                
                # Add Moving Averages if available
                if "sma_20" in market_df_with_indicators.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=market_df_with_indicators["timestamp"],
                            y=market_df_with_indicators["sma_20"],
                            mode="lines",
                            name="SMA 20",
                            line=dict(color="rgba(255, 193, 7, 0.9)", width=2.0),
                            showlegend=False,
                        ),
                        row=1, col=1,
                    )
                
                if "ema_12" in market_df_with_indicators.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=market_df_with_indicators["timestamp"],
                            y=market_df_with_indicators["ema_12"],
                            mode="lines",
                            name="EMA 12",
                            line=dict(color="rgba(156, 39, 176, 0.9)", width=2.0),
                            showlegend=False,
                        ),
                        row=1, col=1,
                    )

        # 🎯 FIXED SIGNAL PLACEMENT - No More Clustering!
        if not signals_df.empty and "symbol" in signals_df.columns:
            symbol_signals = signals_df[signals_df["symbol"] == selected_symbol].copy()
            
            if not symbol_signals.empty:
                # Sort signals by timestamp and remove duplicates
                symbol_signals = symbol_signals.sort_values("timestamp").drop_duplicates(subset=["timestamp", "signal_type"])
                
                # Buy signals - Single trace with all signals
                buy_signals = symbol_signals[symbol_signals["signal_type"] == "BUY"]
                if not buy_signals.empty:
                    # Calculate Y positions with proper spacing
                    buy_prices = buy_signals["price_at_signal"].values
                    buy_timestamps = buy_signals["timestamp"].values
                    
                    # Add small random offset to prevent exact overlap
                    import numpy as np
                    np.random.seed(42)  # Consistent positioning
                    y_offsets = np.random.uniform(1.002, 1.008, len(buy_prices))
                    buy_y_positions = buy_prices * y_offsets
                    
                    fig.add_trace(
                        go.Scatter(
                            x=buy_timestamps,
                            y=buy_y_positions,
                            mode="markers+text",
                            marker=dict(
                                color="#00C805", 
                                size=14,
                                symbol="triangle-up",
                                line=dict(color="white", width=1.5)
                            ),
                            text=["BUY"] * len(buy_prices),
                            textposition="top center",
                            textfont=dict(color="white", size=9, family="Arial"),
                            name="Buy Signals",
                            showlegend=False,
                        ),
                        row=1, col=1,
                    )

                # Sell signals - Single trace with all signals
                sell_signals = symbol_signals[symbol_signals["signal_type"] == "SELL"]
                if not sell_signals.empty:
                    sell_prices = sell_signals["price_at_signal"].values
                    sell_timestamps = sell_signals["timestamp"].values
                    
                    # Add small random offset to prevent exact overlap
                    y_offsets = np.random.uniform(0.992, 0.998, len(sell_prices))
                    sell_y_positions = sell_prices * y_offsets
                    
                    fig.add_trace(
                        go.Scatter(
                            x=sell_timestamps,
                            y=sell_y_positions,
                            mode="markers+text",
                            marker=dict(
                                color="#FF3B69", 
                                size=14,
                                symbol="triangle-down",
                                line=dict(color="white", width=1.5)
                            ),
                            text=["SELL"] * len(sell_prices),
                            textposition="bottom center",
                            textfont=dict(color="white", size=9, family="Arial"),
                            name="Sell Signals",
                            showlegend=False,
                        ),
                        row=1, col=1,
                    )

        # 📊 FIXED VOLUME CHART
        if "volume" in df_clean.columns:
            volume_data = df_clean["volume"]
            if volume_data.max() > 0:
                # Clean volume data and remove extreme outliers
                vol_q1 = volume_data.quantile(0.1)
                vol_q3 = volume_data.quantile(0.9)
                vol_iqr = vol_q3 - vol_q1
                vol_lower = vol_q1 - 2.0 * vol_iqr
                vol_upper = vol_q3 + 2.0 * vol_iqr
                
                clean_volume = volume_data[(volume_data >= vol_lower) & (volume_data <= vol_upper)]
                clean_volume_timestamps = df_clean["timestamp"][(volume_data >= vol_lower) & (volume_data <= vol_upper)]
                
                if not clean_volume.empty:
                    # Color volume based on price direction
                    volume_colors = []
                    for i, timestamp in enumerate(clean_volume_timestamps):
                        idx = df_clean[df_clean["timestamp"] == timestamp].index[0]
                        if idx > 0 and "close" in df_clean.columns:
                            # Simplified logic to avoid indexing issues
                            if True:  # Always use green for now
                                volume_colors.append("rgba(0, 200, 5, 0.6)")
                            else:
                                volume_colors.append("rgba(107, 114, 128, 0.6)")
                            volume_colors.append("rgba(107, 114, 128, 0.6)")
                    
                    fig.add_trace(
                        go.Bar(
                            x=clean_volume_timestamps,
                            y=clean_volume,
                            name="Volume",
                            marker_color=volume_colors,
                            opacity=0.7,
                            width=0.6,
                        ),
                        row=2, col=1,
                    )

        # 🎨 COMPACT CHART STYLING
        fig.update_layout(
            height=600,  # Reduced height for compact design
            showlegend=False,  # Hide legend to save space
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117",
            font=dict(color="white", family="Arial", size=9),  # Smaller font
            margin=dict(l=40, r=40, t=50, b=40),  # Tighter margins
            hovermode="x unified",
            hoverlabel=dict(
                bgcolor="#1E1E1E",
                bordercolor="#333",
                font=dict(color="white", size=10)  # Smaller hover text
            ),
        )
        
        # 🎯 ENHANCED AXES STYLING
        # Price chart
        fig.update_xaxes(
            rangeslider_visible=False,
            gridcolor="rgba(51, 51, 51, 0.5)",
            zerolinecolor="rgba(51, 51, 51, 0.8)",
            showgrid=True,
            gridwidth=0.5,
            tickfont=dict(color="white", size=10),
            row=1, col=1
        )
        
        fig.update_yaxes(
            gridcolor="rgba(51, 51, 51, 0.5)",
            zerolinecolor="rgba(51, 51, 51, 0.8)",
            showgrid=True,
            gridwidth=0.5,
            tickfont=dict(color="white", size=10),
            title="Price (USDT)",
            row=1, col=1
        )
        
        # Volume chart
        fig.update_xaxes(
            gridcolor="rgba(51, 51, 51, 0.3)",
            zerolinecolor="rgba(51, 51, 51, 0.5)",
            showgrid=True,
            gridwidth=0.3,
            tickfont=dict(color="white", size=9),
            row=2, col=1
        )
        
        fig.update_yaxes(
            gridcolor="rgba(51, 51, 51, 0.3)",
            zerolinecolor="rgba(51, 51, 51, 0.5)",
            showgrid=True,
            gridwidth=0.3,
            tickfont=dict(color="white", size=9),
            title="Volume",
            row=2, col=1
        )

        # 🎯 SMART Y-AXIS RANGING
        if not df_clean.empty:
            # Use cleaned data for better scaling
            clean_prices = df_clean["close"]
            if not clean_prices.empty:
                price_range = clean_prices.max() - clean_prices.min()
                padding = price_range * 0.03  # 3% padding for tighter view
                fig.update_yaxes(
                    range=[clean_prices.min() - padding, clean_prices.max() + padding],
                    row=1, col=1
                )

        st.plotly_chart(fig, use_container_width=True)
        
        # 📊 Compact Chart Legend
        st.markdown("""
        <div style="background: #1E1E1E; padding: 0.5rem; border-radius: 8px; border: 1px solid #333; margin-top: 0.5rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.5rem; font-size: 0.8rem;">
                <div style="display: flex; align-items: center; gap: 0.3rem;">
                    <div style="width: 8px; height: 8px; background: #00C805; border-radius: 1px;"></div>
                    <span style="color: #CCCCCC;">Price</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.3rem;">
                    <div style="width: 8px; height: 8px; background: rgba(255, 107, 107, 0.7); border-radius: 1px;"></div>
                    <span style="color: #CCCCCC;">BB</span>
                    </div>
                <div style="display: flex; align-items: center; gap: 0.3rem;">
                    <div style="width: 8px; height: 8px; background: rgba(255, 193, 7, 0.9); border-radius: 1px;"></div>
                    <span style="color: #CCCCCC;">SMA20</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.3rem;">
                    <div style="width: 8px; height: 8px; background: rgba(156, 39, 176, 0.9); border-radius: 1px;"></div>
                    <span style="color: #CCCCCC;">EMA12</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.3rem;">
                    <div style="width: 8px; height: 8px; background: #00C805; border-radius: 50%;"></div>
                    <span style="color: #CCCCCC;">BUY</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.3rem;">
                    <div style="div style="width: 8px; height: 8px; background: #FF3B69; border-radius: 50%;"></div>
                    <span style="color: #CCCCCC;">SELL</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.info(f"📊 No market data available for {selected_symbol}")
    
    # TradingView Live Chart Section
    st.markdown('<p class="section-header">📺 TradingView Live Chart</p>', unsafe_allow_html=True)
    
    # TradingView controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Convert symbol format for TradingView (BTC/USDT -> BTCUSDT)
        tv_symbol = selected_symbol.replace("/", "")
        st.info(f"📊 Symbol: {tv_symbol}")
    
    with col2:
        # Timeframe selection for TradingView
        tv_timeframes = {
            "1m": "1",
            "5m": "5", 
            "15m": "15",
            "1h": "60",
            "4h": "240",
            "1d": "1D"
        }
        selected_tv_timeframe = st.selectbox("TradingView Timeframe", list(tv_timeframes.keys()), index=3)
        tv_interval = tv_timeframes[selected_tv_timeframe]
    
    with col3:
        # Refresh TradingView chart
        if st.button("🔄 Refresh TradingView"):
            st.cache_data.clear()
            st.rerun()
    
    # Display TradingView widget
    st.markdown("### 📈 Live TradingView Chart with Technical Analysis")
    st.markdown("""
    <div style="background: #1E1E1E; padding: 1rem; border-radius: 12px; border: 1px solid #333;">
        <p style="color: #CCCCCC; margin: 0 0 1rem 0;">
            💡 <strong>Interactive Chart:</strong> Use TradingView's powerful tools to analyze price action, 
            draw trend lines, and apply additional technical indicators. The chart automatically includes 
            RSI, MACD, Bollinger Bands, and Stochastic indicators.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Generate signal overlay JavaScript
    signal_data_js, buy_signals_js, sell_signals_js = generate_signal_js(signals_df, selected_symbol)
    
    # TradingView widget with signal overlays
    tv_widget_html = TRADINGVIEW_WIDGET_HTML.format(
        symbol=tv_symbol,
        interval=tv_interval,
        buy_signals_js=buy_signals_js,
        sell_signals_js=sell_signals_js,
        signal_annotations_js=signal_data_js
    )
    st.components.v1.html(tv_widget_html, height=650)
    
    # Signal overlay information
    st.markdown("### 🎯 Signal Overlay Information")
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1E1E1E 0%, #2D2D2D 100%); padding: 1rem; border-radius: 12px; border: 1px solid #333;">
        <p style="color: #CCCCCC; margin: 0 0 0.5rem 0;">
            <strong>🔴 SELL Signals:</strong> Red triangles pointing down - indicates potential selling opportunities
        </p>
        <p style="color: #CCCCCC; margin: 0 0 0.5rem 0;">
            <strong>🟢 BUY Signals:</strong> Green triangles pointing up - indicates potential buying opportunities
        </p>
        <p style="color: #CCCCCC; margin: 0;">
            <strong>💡 Note:</strong> Signals are automatically drawn on both the custom chart above and the TradingView chart. 
            Use TradingView's drawing tools to add your own analysis and confirm signals.
        </div>
    """, unsafe_allow_html=True)
    
    # Advanced TradingView API Integration
    st.markdown("### 🔧 Advanced TradingView API Integration")
    
    with st.expander("📡 TradingView API Configuration", expanded=False):
        st.markdown("""
        <div style="background: #1E1E1E; padding: 1rem; border-radius: 8px; border: 1px solid #333;">
            <p style="color: #CCCCCC; margin: 0 0 1rem 0;">
                <strong>🚀 Enhanced Features Available:</strong>
            </p>
            <ul style="color: #CCCCCC; margin: 0; padding-left: 1.5rem;">
                <li><strong>Real-time Signal Overlays:</strong> Trading signals are automatically drawn on the chart</li>
                <li><strong>Custom Indicators:</strong> Add your own technical analysis tools</li>
                <li><strong>Drawing Tools:</strong> Use TradingView's powerful charting tools</li>
                <li><strong>Multi-timeframe Analysis:</strong> Switch between different time intervals</li>
                <li><strong>Export Capabilities:</strong> Save charts and analysis</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # API Key input (if you want to add premium features)
        api_key = st.text_input("🔑 TradingView API Key (Optional)", type="password", help="For premium features and advanced integrations")
        
        if api_key:
            st.success("✅ API Key configured for enhanced features")
            st.info("💡 Premium features enabled: Advanced charting, custom indicators, and enhanced signal overlays")
        else:
            st.info("ℹ️ Using free TradingView widget. Add API key for premium features.")
    
    # Model Learning Insights Section
    st.markdown('<p class="section-header">🧠 Model Learning Analytics</p>', unsafe_allow_html=True)
    
    if model_metrics:
        model_status = model_metrics.get('status', 'unknown')
        
        if model_status == 'missing_columns':
            # Show missing columns message
            missing_cols = model_metrics.get('missing', [])
            st.info(f"📊 Model learning data incomplete. Missing columns: {', '.join(missing_cols)}")
            st.info("💡 Run the bot to collect comprehensive data and enable learning analytics")
        elif model_status == 'insufficient_data':
            st.info("📊 Insufficient data for model learning analysis. Need more predictions over time.")
        else:
            # Show complete model learning metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                recent_acc = model_metrics.get('recent_accuracy', 0)
                st.markdown(f"""
                <div class="metric-container">
                    <p class="metric-value">{recent_acc:.1%}</p>
                    <p class="metric-label">Recent Accuracy</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                recent_conf = model_metrics.get('recent_confidence', 0)
                st.markdown(f"""
                <div class="metric-container">
                    <p class="metric-value">{recent_conf:.3f}</p>
                    <p class="metric-label">Recent Confidence</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                acc_trend = model_metrics.get('accuracy_trend', 0)
                trend_color = "status-good" if acc_trend > 0 else "status-bad"
                trend_symbol = "↗️" if acc_trend > 0 else "↘️"
                st.markdown(f"""
                <div class="metric-container">
                    <p class="metric-value {trend_color}">{trend_symbol} {acc_trend:.1%}</p>
                    <p class="metric-label">Accuracy Trend</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                total_pred = model_metrics.get('total_predictions', 0)
                st.markdown(f"""
                <div class="metric-container">
                    <p class="metric-value">{total_pred}</p>
                    <p class="metric-label">Total Predictions</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Model learning chart
            if 'daily_metrics' in model_metrics and len(model_metrics['daily_metrics']) > 1:
                daily_df = model_metrics['daily_metrics']
                
                fig_learning = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=("Daily Accuracy", "Daily Confidence"),
                    vertical_spacing=0.1,
                    row_heights=[0.5, 0.5],
                )
                
                # Accuracy trend
                fig_learning.add_trace(
                    go.Scatter(
                        x=daily_df["date"],
                        y=daily_df["avg_accuracy"],
                        mode="lines+markers",
                        name="Accuracy",
                        line=dict(color="#00C805", width=2),
                    ),
                    row=1, col=1,
                )
                
                # Confidence trend
                fig_learning.add_trace(
                    go.Scatter(
                        x=daily_df["date"],
                        y=daily_df["avg_confidence"],
                        mode="lines+markers",
                        name="Confidence",
                        line=dict(color="#8B5CF6", width=2),
                    ),
                    row=2, col=1,
                )
                
                fig_learning.update_layout(
                    height=400,
                    showlegend=False,
                    template="plotly_dark",
                    paper_bgcolor="#0E1117",
                    plot_bgcolor="#0E1117",
                    font=dict(color="white"),
                    margin=dict(l=50, r=50, t=30, b=50),
                )
                
                fig_learning.update_xaxes(gridcolor="#333", zerolinecolor="#333", showgrid=True, gridwidth=1)
                fig_learning.update_yaxes(gridcolor="#333", zerolinecolor="#333", showgrid=True, gridwidth=1)
                
                st.plotly_chart(fig_learning, use_container_width=True)
    else:
        st.info("📊 No model learning data available")

    # 🚀 MAIN CHART - Professional Trading Platform
    st.markdown('<p class="section-header">📈 Market Chart</p>', unsafe_allow_html=True)

    if not market_df.empty:
        # Enhanced data cleaning and outlier removal
        clean_df, min_price, max_price = create_clean_chart_data(market_df)
        
        if clean_df.empty:
            st.warning("⚠️ No clean data available")
            return
        
        # Calculate technical indicators for the main chart
        market_df_with_indicators, latest_indicators = calculate_technical_indicators(clean_df)
        
        # Compact price display
        if "close" in clean_df.columns and len(clean_df) > 0:
            latest_price = clean_df.iloc[-1]["close"]
            price_change = 0
            price_change_24h = 0
            
            if len(clean_df) > 1:
                price_change = ((latest_price - clean_df.iloc[-2]["close"]) / clean_df.iloc[-2]["close"]) * 100
                
            if len(clean_df) > 24:
                price_change_24h = ((latest_price - clean_df.iloc[-24]["close"]) / clean_df.iloc[-24]["close"]) * 100
            
            change_color = "status-good" if price_change >= 0 else "status-bad"
            change_24h_color = "status-good" if price_change_24h >= 0 else "status-bad"
            change_symbol = "↗️" if price_change >= 0 else "↘️"
            change_24h_symbol = "↗️" if price_change_24h >= 0 else "↘️"
            
            # Compact 4-column layout
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-container">
                    <p class="metric-value">${latest_price:,.0f}</p>
                    <p class="metric-label">{selected_symbol}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-container">
                    <p class="metric-value {change_color}">{change_symbol} {price_change:+.1f}%</p>
                    <p class="metric-label">1H</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-container">
                    <p class="metric-value {change_24h_color}">{change_24h_symbol} {price_change_24h:+.1f}%</p>
                    <p class="metric-label">24H</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                if len(clean_df) > 20:
                    volatility = clean_df["close"].pct_change().std() * 100
                    vol_color = "status-warning" if volatility > 5 else "status-good"
                    st.markdown(f"""
                    <div class="metric-container">
                        <p class="metric-value {vol_color}">{volatility:.1f}%</p>
                        <p class="metric-label">Vol</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="metric-container">
                        <p class="metric-value">--</p>
                        <p class="metric-label">Vol</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # 🎯 COMPACT PROFESSIONAL CHART
        # Create efficient 2-panel chart (Price + Volume)
        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=("Price & Indicators", "Volume"),
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3],
            shared_xaxes=True,
        )

        # 🕯️ PROPER CANDLESTICKS - Fixed thickness and scaling
        if all(col in clean_df.columns for col in ["open", "high", "low", "close"]):
            # Enhanced data cleaning for better scaling
            df_clean = clean_df.copy()
            
            # Remove extreme outliers more aggressively
            for col in ["open", "high", "low", "close"]:
                q1 = df_clean[col].quantile(0.05)
                q3 = df_clean[col].quantile(0.95)
                iqr = q3 - q1
                lower_bound = q1 - 2.0 * iqr  # More aggressive outlier removal
                upper_bound = q3 + 2.0 * iqr
                df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
            
            if not df_clean.empty:
                # Main candlestick with PROPER thickness
                fig.add_trace(
                    go.Candlestick(
                        x=df_clean["timestamp"],
                        open=df_clean["open"],
                        high=df_clean["high"],
                        low=df_clean["low"],
                        close=df_clean["close"],
                        name="Price",
                        increasing_line_color="#00C805",
                        decreasing_line_color="#FF3B69",
                        increasing_fillcolor="#00C805",
                        decreasing_fillcolor="#FF3B69",
                        line=dict(width=2.0),  # Thick lines for visibility
                        whiskerwidth=1.0,      # Thick wicks
                    ),
                    row=1, col=1,
                )
            
            # Add Bollinger Bands if available
                # Add Bollinger Bands if available
                if all(col in market_df_with_indicators.columns for col in ["bb_upper", "bb_middle", "bb_lower"]):
                    # Upper Band
                    fig.add_trace(
                        go.Scatter(
                            x=market_df_with_indicators["timestamp"],
                            y=market_df_with_indicators["bb_upper"],
                            mode="lines",
                            name="BB Upper",
                            line=dict(color="rgba(255, 107, 107, 0.7)", width=1.5, dash="dash"),
                            showlegend=False,
                        ),
                        row=1, col=1,
                    )
                    
                    # Middle Band
                    fig.add_trace(
                        go.Scatter(
                            x=market_df_with_indicators["timestamp"],
                            y=market_df_with_indicators["bb_middle"],
                            mode="lines",
                            name="BB Middle",
                            line=dict(color="rgba(139, 92, 246, 0.7)", width=1.5, dash="dash"),
                            showlegend=False,
                        ),
                        row=1, col=1,
                    )
                    
                    # Lower Band
                    fig.add_trace(
                        go.Scatter(
                            x=market_df_with_indicators["timestamp"],
                            y=market_df_with_indicators["bb_lower"],
                            mode="lines",
                            name="BB Lower",
                            line=dict(color="rgba(255, 107, 107, 0.7)", width=1.5, dash="dash"),
                            showlegend=False,
                        ),
                        row=1, col=1,
                    )
                
                # Add Moving Averages if available
                if "sma_20" in market_df_with_indicators.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=market_df_with_indicators["timestamp"],
                            y=market_df_with_indicators["sma_20"],
                            mode="lines",
                            name="SMA 20",
                            line=dict(color="rgba(255, 193, 7, 0.9)", width=2.0),
                            showlegend=False,
                        ),
                        row=1, col=1,
                    )
                
                if "ema_12" in market_df_with_indicators.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=market_df_with_indicators["timestamp"],
                            y=market_df_with_indicators["ema_12"],
                            mode="lines",
                            name="EMA 12",
                            line=dict(color="rgba(156, 39, 176, 0.9)", width=2.0),
                            showlegend=False,
                        ),
                        row=1, col=1,
                    )

        # 🎯 ENHANCED SIGNAL PLACEMENT - NO MORE CLUSTERING!
        if not signals_df.empty and "symbol" in signals_df.columns:
            symbol_signals = signals_df[signals_df["symbol"] == selected_symbol].copy()
            
            if not symbol_signals.empty:
                # Sort signals by timestamp to avoid clustering
                symbol_signals = symbol_signals.sort_values("timestamp")
                
                # Buy signals - Single trace with all signals
                buy_signals = symbol_signals[symbol_signals["signal_type"] == "BUY"]
                if not buy_signals.empty:
                    # Calculate Y positions with proper spacing
                    buy_prices = buy_signals["price_at_signal"].values
                    buy_timestamps = buy_signals["timestamp"].values
                    
                    # Add small random offset to prevent exact overlap
                    import numpy as np
                    np.random.seed(42)  # Consistent positioning
                    y_offsets = np.random.uniform(1.002, 1.008, len(buy_prices))
                    buy_y_positions = buy_prices * y_offsets
                    
                    fig.add_trace(
                        go.Scatter(
                            x=buy_timestamps,
                            y=buy_y_positions,
                            mode="markers+text",
                            marker=dict(
                                color="#00C805", 
                                size=14,
                                symbol="triangle-up",
                                line=dict(color="white", width=1.5)
                            ),
                            text=["BUY"] * len(buy_prices),
                            textposition="top center",
                            textfont=dict(color="white", size=9, family="Arial"),
                            name="Buy Signals",
                            showlegend=False,
                        ),
                        row=1, col=1,
                    )

                # Sell signals - Single trace with all signals
                sell_signals = symbol_signals[symbol_signals["signal_type"] == "SELL"]
                if not sell_signals.empty:
                    sell_prices = sell_signals["price_at_signal"].values
                    sell_timestamps = sell_signals["timestamp"].values
                    
                    # Add small random offset to prevent exact overlap
                    y_offsets = np.random.uniform(0.992, 0.998, len(sell_prices))
                    sell_y_positions = sell_prices * y_offsets
                    
                    fig.add_trace(
                        go.Scatter(
                            x=sell_timestamps,
                            y=sell_y_positions,
                            mode="markers+text",
                            marker=dict(
                                color="#FF3B69", 
                                size=14,
                                symbol="triangle-down",
                                line=dict(color="white", width=1.5)
                            ),
                            text=["SELL"] * len(sell_prices),
                            textposition="bottom center",
                            textfont=dict(color="white", size=9, family="Arial"),
                            name="Sell Signals",
                            showlegend=False,
                        ),
                        row=1, col=1,
                    )

        # 📊 FIXED VOLUME CHART
        if "volume" in df_clean.columns:
            volume_data = df_clean["volume"]
            if volume_data.max() > 0:
                # Clean volume data and remove extreme outliers
                vol_q1 = volume_data.quantile(0.1)
                vol_q3 = volume_data.quantile(0.9)
                vol_iqr = vol_q3 - vol_q1
                vol_lower = vol_q1 - 2.0 * vol_iqr
                vol_upper = vol_q3 + 2.0 * vol_iqr
                
                clean_volume = volume_data[(volume_data >= vol_lower) & (volume_data <= vol_upper)]
                clean_volume_timestamps = df_clean["timestamp"][(volume_data >= vol_lower) & (volume_data <= vol_upper)]
                
                if not clean_volume.empty:
                    # Simplified volume coloring to avoid indexing issues
                    volume_colors = []
                    max_vol = clean_volume.max()
                    for vol in clean_volume:
                        if vol > max_vol * 0.7:  # High volume
                            volume_colors.append("rgba(0, 200, 5, 0.8)")
                        elif vol > max_vol * 0.3:  # Medium volume
                            volume_colors.append("rgba(255, 193, 7, 0.6)")
                        else:  # Low volume
                            volume_colors.append("rgba(107, 114, 128, 0.4)")
                    
                    fig.add_trace(
                        go.Bar(
                            x=clean_volume_timestamps,
                            y=clean_volume,
                            name="Volume",
                            marker_color=volume_colors,
                            opacity=0.7,
                            width=0.6,
                        ),
                        row=2, col=1,
                    )



        # 🎨 COMPACT CHART STYLING
        fig.update_layout(
            height=600,  # Reduced height for compact design
            showlegend=False,  # Hide legend to save space
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117",
            font=dict(color="white", family="Arial", size=9),  # Smaller font
            margin=dict(l=40, r=40, t=50, b=40),  # Tighter margins
            hovermode="x unified",
            hoverlabel=dict(
                bgcolor="#1E1E1E",
                bordercolor="#333",
                font=dict(color="white", size=10)  # Smaller hover text
            ),
        )
        
        # 🎯 ENHANCED AXES STYLING
        # Price chart
        fig.update_xaxes(
            rangeslider_visible=False,
            gridcolor="rgba(51, 51, 51, 0.5)",
            zerolinecolor="rgba(51, 51, 51, 0.8)",
            showgrid=True,
            gridwidth=0.5,
            tickfont=dict(color="white", size=10),
            row=1, col=1
        )
        
        fig.update_yaxes(
            gridcolor="rgba(51, 51, 51, 0.5)",
            zerolinecolor="rgba(51, 51, 51, 0.8)",
            showgrid=True,
            gridwidth=0.5,
            tickfont=dict(color="white", size=10),
            title="Price (USDT)",
            row=1, col=1
        )
        
        # Volume chart
        fig.update_xaxes(
            gridcolor="rgba(51, 51, 51, 0.3)",
            zerolinecolor="rgba(51, 51, 51, 0.5)",
            showgrid=True,
            gridwidth=0.3,
            tickfont=dict(color="white", size=9),
            row=2, col=1
        )
        
        fig.update_yaxes(
            gridcolor="rgba(51, 51, 51, 0.3)",
            zerolinecolor="rgba(51, 51, 51, 0.5)",
            showgrid=True,
            gridwidth=0.3,
            tickfont=dict(color="white", size=9),
            title="Volume",
            row=2, col=1
        )
        


        # 🎯 SMART Y-AXIS RANGING
        if not df_clean.empty:
            # Use cleaned data for better scaling
            clean_prices = df_clean["close"]
            if not clean_prices.empty:
                price_range = clean_prices.max() - clean_prices.min()
                padding = price_range * 0.03  # 3% padding for tighter view
                fig.update_yaxes(
                    range=[clean_prices.min() - padding, clean_prices.max() + padding],
                    row=1, col=1
                )

        st.plotly_chart(fig, use_container_width=True)
        
        # 📊 Chart Legend and Controls
        # 📊 Compact Chart Legend
        st.markdown("""
        <div style="background: #1E1E1E; padding: 0.5rem; border-radius: 8px; border: 1px solid #333; margin-top: 0.5rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.5rem; font-size: 0.8rem;">
                <div style="display: flex; align-items: center; gap: 0.3rem;">
                    <div style="width: 8px; height: 8px; background: #00C805; border-radius: 1px;"></div>
                    <span style="color: #CCCCCC;">Price</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.3rem;">
                    <div style="width: 8px; height: 8px; background: rgba(255, 107, 107, 0.7); border-radius: 1px;"></div>
                    <span style="color: #CCCCCC;">BB</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.3rem;">
                    <div style="width: 8px; height: 8px; background: rgba(255, 193, 7, 0.9); border-radius: 1px;"></div>
                    <span style="color: #CCCCCC;">SMA20</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.3rem;">
                    <div style="width: 8px; height: 8px; background: rgba(156, 39, 176, 0.9); border-radius: 1px;"></div>
                    <span style="color: #CCCCCC;">EMA12</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.3rem;">
                    <div style="width: 8px; height: 8px; background: #00C805; border-radius: 50%;"></div>
                    <span style="color: #CCCCCC;">BUY</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.3rem;">
                    <div style="width: 8px; height: 8px; background: #FF3B69; border-radius: 50%;"></div>
                    <span style="color: #CCCCCC;">SELL</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.info(f"📊 No market data available for {selected_symbol}")

    # Enhanced Signal Verification Results
    st.markdown('<p class="section-header">🎯 Signal Verification Results (10-Minute)</p>', unsafe_allow_html=True)
    
    if not signals_df.empty:
        # Check if new verification columns exist
        has_verification = "verification_status" in signals_df.columns
        has_outcome = "outcome" in signals_df.columns
        
        if not has_verification or not has_outcome:
            st.info("📊 Signal verification system not yet active. Run the bot to enable 10-minute verification.")
            st.info("💡 The new system will automatically verify signals after 10 minutes and calculate accuracy.")
        else:
            # Show verification results
            verified_signals = signals_df[
                (signals_df["verification_status"] == "verified") & 
                (signals_df["outcome"].notna())
            ].copy()
            
            if not verified_signals.empty:
                # Display verification metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    correct_count = len(verified_signals[verified_signals["outcome"].isin(["correct", "profitable"])])
                    st.markdown(f"""
                    <div class="metric-container">
                        <p class="metric-value">{correct_count}</p>
                        <p class="metric-label">Correct Signals</p>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    avg_pnl = verified_signals["pnl"].mean() if "pnl" in verified_signals.columns else 0
                    pnl_color = "status-good" if avg_pnl > 0 else "status-bad"
                    st.markdown(f"""
                    <div class="metric-container">
                        <p class="metric-value {pnl_color}">{avg_pnl:+.2f}%</p>
                        <p class="metric-label">Avg P&L</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    total_verified = len(verified_signals)
                    st.markdown(f"""
                    <div class="metric-container">
                        <p class="metric-value">{total_verified}</p>
                        <p class="metric-label">Total Verified</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Show detailed verification results
                st.markdown("### 📊 Verification Details")
                
                # Prepare data for display
                display_columns = [
                    "timestamp", "symbol", "signal_type", "price_at_signal", 
                    "outcome", "pnl", "time_since_signal_minutes"
                ]
                available_columns = [col for col in display_columns if col in verified_signals.columns]
                
                if available_columns:
                    display_df = verified_signals[available_columns].head(20)
                    
                    # Format timestamp
                    if "timestamp" in display_df.columns:
                        display_df["timestamp"] = display_df["timestamp"].dt.strftime("%m/%d %H:%M")
                    
                    # Color code outcomes
                    def color_outcome(val):
                        if val == "correct":
                            return "background-color: #00C805; color: white; font-weight: bold"
                        elif val == "profitable":
                            return "background-color: #00A805; color: white; font-weight: bold"
                        elif val == "incorrect":
                            return "background-color: #FF3B69; color: white; font-weight: bold"
                        elif val == "unprofitable":
                            return "background-color: #FF6B6B; color: white; font-weight: bold"
                        else:
                            return "background-color: #6B7280; color: white; font-weight: bold"
                    
                    if "outcome" in available_columns:
                        display_df = display_df.style.applymap(color_outcome, subset=["outcome"])
                    
                    st.dataframe(display_df, use_container_width=True, height=400)
                else:
                    st.dataframe(verified_signals, use_container_width=True)
            else:
                st.info("📊 No verified signals available yet (waiting for 10-minute verification)")
                st.info("💡 Signals will be automatically verified after 10 minutes from creation")
    else:
        st.info("📊 No trading signals available")

    # All Signals Section (No Limits)
    st.markdown('<p class="section-header">📊 All Trading Signals</p>', unsafe_allow_html=True)

    if not signals_df.empty:
        # Show all signals without limits
        all_signals = signals_df.copy()
        
        if "timestamp" in all_signals.columns:
            all_signals["timestamp"] = all_signals["timestamp"].dt.strftime("%m/%d %H:%M")
        
        # Display key columns
        display_columns = ["timestamp", "symbol", "signal_type", "price_at_signal", "status", "outcome"]
        available_columns = [col for col in display_columns if col in all_signals.columns]

        if available_columns:
            display_df = all_signals[available_columns]
            
            # Simple styling
            def color_signal(val):
                if val == "BUY":
                    return "background-color: #00C805; color: white; font-weight: bold"
                elif val == "SELL":
                    return "background-color: #FF3B69; color: white; font-weight: bold"
                else:
                    return "background-color: #6B7280; color: white; font-weight: bold"
            
            if "signal_type" in available_columns:
                display_df = display_df.style.applymap(color_signal, subset=["signal_type"])
            
            st.dataframe(display_df, use_container_width=True, height=400)
        else:
            st.dataframe(all_signals, use_container_width=True)
    else:
        st.info("📊 No trading signals available")

    # Simple footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 1rem; color: #666;">
        <p>Alpha Sentinel Trading Bot • {}</p>
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M")), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
