-- =====================================================
-- ALPHA SENTINEL BOT - COMPLETE DATABASE SCHEMA
-- =====================================================
-- This schema supports all bot functionality:
-- - Market data collection and storage
-- - Technical indicator calculations
-- - Sentiment analysis data
-- - ML model predictions
-- - Trading signal generation
-- - Performance tracking and validation
-- =====================================================

-- Enable UUID extension for unique identifiers
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =====================================================
-- 1. MARKET DATA TABLE
-- =====================================================
-- Stores raw market data from exchanges
CREATE TABLE market_data (
    id BIGSERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    open NUMERIC NOT NULL,
    high NUMERIC NOT NULL,
    low NUMERIC NOT NULL,
    close NUMERIC NOT NULL,
    volume NUMERIC NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Composite unique constraint to prevent duplicates
    UNIQUE(symbol, timestamp)
);

-- Index for efficient querying
CREATE INDEX idx_market_data_symbol_timestamp ON market_data(symbol, timestamp DESC);
CREATE INDEX idx_market_data_timestamp ON market_data(timestamp DESC);

-- =====================================================
-- 2. TECHNICAL INDICATORS TABLE
-- =====================================================
-- Stores calculated technical indicators
CREATE TABLE technical_indicators (
    id BIGSERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    
    -- RSI (Relative Strength Index)
    rsi NUMERIC,
    
    -- MACD (Moving Average Convergence Divergence)
    macd NUMERIC,
    macd_signal NUMERIC,
    macd_histogram NUMERIC,
    
    -- Bollinger Bands
    bb_upper NUMERIC,
    bb_middle NUMERIC,
    bb_lower NUMERIC,
    bb_width NUMERIC,
    bb_percent NUMERIC,
    
    -- Stochastic Oscillator
    stoch_k NUMERIC,
    stoch_d NUMERIC,
    
    -- Moving Averages
    sma_20 NUMERIC,
    sma_50 NUMERIC,
    ema_12 NUMERIC,
    ema_26 NUMERIC,
    
    -- ATR (Average True Range)
    atr NUMERIC,
    
    -- Volume indicators
    volume_sma NUMERIC,
    volume_ratio NUMERIC,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Composite unique constraint
    UNIQUE(symbol, timestamp)
);

-- Indexes for technical indicators
CREATE INDEX idx_technical_indicators_symbol_timestamp ON technical_indicators(symbol, timestamp DESC);
CREATE INDEX idx_technical_indicators_timestamp ON technical_indicators(timestamp DESC);

-- =====================================================
-- 3. SENTIMENT DATA TABLE
-- =====================================================
-- Stores sentiment analysis results
CREATE TABLE sentiment_data (
    id BIGSERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    
    -- Fear & Greed Index
    fear_greed_index INTEGER,
    fear_greed_value TEXT,
    
    -- News sentiment
    news_sentiment_score NUMERIC,
    news_sentiment_label TEXT,
    news_count INTEGER,
    
    -- Social media sentiment
    social_sentiment_score NUMERIC,
    social_sentiment_label TEXT,
    
    -- Overall sentiment
    overall_sentiment_score NUMERIC,
    overall_sentiment_label TEXT,
    
    -- Sentiment source details
    sentiment_sources TEXT[],
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Composite unique constraint
    UNIQUE(symbol, timestamp)
);

-- Indexes for sentiment data
CREATE INDEX idx_sentiment_data_symbol_timestamp ON sentiment_data(symbol, timestamp DESC);
CREATE INDEX idx_sentiment_data_timestamp ON sentiment_data(timestamp DESC);

-- =====================================================
-- 4. ML PREDICTIONS TABLE
-- =====================================================
-- Stores machine learning model predictions
CREATE TABLE ml_predictions (
    id BIGSERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    
    -- Model information
    model_name TEXT NOT NULL,
    model_version TEXT,
    
    -- Prediction details
    prediction_value NUMERIC NOT NULL,
    actual_value NUMERIC NOT NULL,
    confidence_score NUMERIC NOT NULL,
    
    -- Prediction metadata
    prediction_type TEXT NOT NULL, -- 'price_prediction', 'trend_prediction', etc.
    timeframe TEXT NOT NULL, -- '1_step_ahead', '5_step_ahead', etc.
    features_used TEXT[],
    
    -- Status tracking
    status TEXT DEFAULT 'active', -- 'active', 'completed', 'expired'
    
    -- Performance metrics (filled after validation)
    prediction_accuracy NUMERIC,
    is_correct BOOLEAN,
    price_error_percentage NUMERIC,
    
    -- Completion tracking
    completed_at TIMESTAMP WITH TIME ZONE,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Composite unique constraint
    UNIQUE(symbol, timestamp, model_name)
);

-- Indexes for ML predictions
CREATE INDEX idx_ml_predictions_symbol_timestamp ON ml_predictions(symbol, timestamp DESC);
CREATE INDEX idx_ml_predictions_status ON ml_predictions(status);
CREATE INDEX idx_ml_predictions_model ON ml_predictions(model_name);

-- =====================================================
-- 5. TRADING SIGNALS TABLE
-- =====================================================
-- Stores generated trading signals
CREATE TABLE trading_signals (
    id BIGSERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    
    -- Signal details
    signal_type TEXT NOT NULL, -- 'BUY', 'SELL', 'HOLD'
    confidence NUMERIC NOT NULL,
    price_at_signal NUMERIC NOT NULL,
    
    -- Risk management
    stop_loss NUMERIC,
    take_profit NUMERIC,
    risk_reward_ratio NUMERIC,
    
    -- Signal reasoning
    reason TEXT,
    technical_reasons TEXT[],
    sentiment_reasons TEXT[],
    ml_reasons TEXT[],
    
    -- Status tracking
    status TEXT DEFAULT 'active', -- 'active', 'completed', 'cancelled'
    outcome TEXT DEFAULT 'pending', -- 'pending', 'correct', 'incorrect'
    
    -- Performance metrics (filled after validation)
    pnl NUMERIC, -- Profit/Loss percentage
    current_price NUMERIC,
    price_change_pct NUMERIC,
    
    -- Completion tracking
    completed_at TIMESTAMP WITH TIME ZONE,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Composite unique constraint
    UNIQUE(symbol, timestamp, signal_type)
);

-- Indexes for trading signals
CREATE INDEX idx_trading_signals_symbol_timestamp ON trading_signals(symbol, timestamp DESC);
CREATE INDEX idx_trading_signals_status ON trading_signals(status);
CREATE INDEX idx_trading_signals_type ON trading_signals(signal_type);
CREATE INDEX idx_trading_signals_outcome ON trading_signals(outcome);

-- =====================================================
-- 6. BOT PERFORMANCE TABLE
-- =====================================================
-- Stores daily performance metrics
CREATE TABLE bot_performance (
    id BIGSERIAL PRIMARY KEY,
    date DATE NOT NULL UNIQUE,
    
    -- Signal metrics
    total_signals INTEGER DEFAULT 0,
    buy_signals INTEGER DEFAULT 0,
    sell_signals INTEGER DEFAULT 0,
    hold_signals INTEGER DEFAULT 0,
    
    -- Prediction accuracy
    correct_predictions INTEGER DEFAULT 0,
    incorrect_predictions INTEGER DEFAULT 0,
    accuracy_rate NUMERIC DEFAULT 0.0,
    
    -- Financial performance
    total_pnl NUMERIC DEFAULT 0.0,
    average_pnl_per_signal NUMERIC DEFAULT 0.0,
    best_signal_pnl NUMERIC DEFAULT 0.0,
    worst_signal_pnl NUMERIC DEFAULT 0.0,
    
    -- ML model performance
    ml_predictions_count INTEGER DEFAULT 0,
    ml_accuracy_rate NUMERIC DEFAULT 0.0,
    
    -- Risk metrics
    max_drawdown NUMERIC DEFAULT 0.0,
    sharpe_ratio NUMERIC DEFAULT 0.0,
    
    -- Timestamps
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for bot performance
CREATE INDEX idx_bot_performance_date ON bot_performance(date DESC);

-- =====================================================
-- 7. MODEL TRAINING HISTORY TABLE
-- =====================================================
-- Tracks ML model training sessions
CREATE TABLE model_training_history (
    id BIGSERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    model_name TEXT NOT NULL,
    
    -- Training parameters
    training_start TIMESTAMP WITH TIME ZONE NOT NULL,
    training_end TIMESTAMP WITH TIME ZONE,
    epochs INTEGER,
    batch_size INTEGER,
    
    -- Training results
    final_loss NUMERIC,
    final_accuracy NUMERIC,
    validation_accuracy NUMERIC,
    
    -- Data used
    training_samples INTEGER,
    validation_samples INTEGER,
    features_count INTEGER,
    
    -- Model file
    model_file_path TEXT,
    model_size_mb NUMERIC,
    
    -- Status
    status TEXT DEFAULT 'training', -- 'training', 'completed', 'failed'
    error_message TEXT,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for model training
CREATE INDEX idx_model_training_symbol ON model_training_history(symbol);
CREATE INDEX idx_model_training_status ON model_training_history(status);
CREATE INDEX idx_model_training_date ON model_training_history(training_start DESC);

-- =====================================================
-- 8. SYSTEM LOGS TABLE
-- =====================================================
-- Stores system-level logs for debugging
CREATE TABLE system_logs (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    level TEXT NOT NULL, -- 'INFO', 'WARNING', 'ERROR', 'DEBUG'
    component TEXT NOT NULL, -- 'main', 'ml_models', 'data_collector', etc.
    message TEXT NOT NULL,
    details JSONB,
    
    -- Optional error tracking
    error_type TEXT,
    stack_trace TEXT
);

-- Indexes for system logs
CREATE INDEX idx_system_logs_timestamp ON system_logs(timestamp DESC);
CREATE INDEX idx_system_logs_level ON system_logs(level);
CREATE INDEX idx_system_logs_component ON system_logs(component);

-- =====================================================
-- 9. CONFIGURATION TABLE
-- =====================================================
-- Stores bot configuration settings
CREATE TABLE bot_config (
    id BIGSERIAL PRIMARY KEY,
    key TEXT NOT NULL UNIQUE,
    value TEXT NOT NULL,
    description TEXT,
    category TEXT NOT NULL, -- 'trading', 'ml', 'data_collection', 'risk_management'
    is_active BOOLEAN DEFAULT TRUE,
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Insert default configuration
INSERT INTO bot_config (key, value, description, category) VALUES
('analysis_interval_minutes', '5', 'Interval between market analysis cycles', 'trading'),
('ml_confidence_threshold', '0.6', 'Minimum confidence for ML predictions', 'ml'),
('max_position_size_usd', '1000', 'Maximum position size in USD', 'risk_management'),
('stop_loss_percentage', '2.0', 'Default stop loss percentage', 'risk_management'),
('take_profit_percentage', '4.0', 'Default take profit percentage', 'risk_management'),
('data_retention_days', '90', 'Number of days to retain historical data', 'data_collection'),
('sentiment_update_interval', '15', 'Minutes between sentiment updates', 'data_collection');

-- =====================================================
-- 10. EXCHANGE API KEYS TABLE
-- =====================================================
-- Stores encrypted API keys for exchanges
CREATE TABLE exchange_api_keys (
    id BIGSERIAL PRIMARY KEY,
    exchange_name TEXT NOT NULL,
    api_key_encrypted TEXT NOT NULL,
    api_secret_encrypted TEXT NOT NULL,
    passphrase_encrypted TEXT, -- For some exchanges like Coinbase Pro
    is_active BOOLEAN DEFAULT TRUE,
    permissions TEXT[], -- ['read', 'trade', 'withdraw']
    rate_limit_info JSONB,
    last_used TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(exchange_name)
);

-- =====================================================
-- VIEWS FOR EASY DATA ACCESS
-- =====================================================

-- View for latest market data with indicators
CREATE VIEW latest_market_analysis AS
SELECT 
    md.symbol,
    md.timestamp,
    md.open,
    md.high,
    md.low,
    md.close,
    md.volume,
    ti.rsi,
    ti.macd,
    ti.bb_upper,
    ti.bb_lower,
    ti.sma_20,
    ti.sma_50,
    sd.overall_sentiment_score,
    sd.overall_sentiment_label
FROM market_data md
LEFT JOIN technical_indicators ti ON md.symbol = ti.symbol AND md.timestamp = ti.timestamp
LEFT JOIN sentiment_data sd ON md.symbol = sd.symbol AND md.timestamp = sd.timestamp
WHERE md.timestamp >= NOW() - INTERVAL '24 hours'
ORDER BY md.symbol, md.timestamp DESC;

-- View for signal performance summary
CREATE VIEW signal_performance_summary AS
SELECT 
    symbol,
    signal_type,
    COUNT(*) as total_signals,
    COUNT(CASE WHEN outcome = 'correct' THEN 1 END) as correct_signals,
    COUNT(CASE WHEN outcome = 'incorrect' THEN 1 END) as incorrect_signals,
    ROUND(
        COUNT(CASE WHEN outcome = 'correct' THEN 1 END)::NUMERIC / COUNT(*) * 100, 2
    ) as accuracy_percentage,
    AVG(pnl) as avg_pnl,
    MAX(pnl) as best_pnl,
    MIN(pnl) as worst_pnl
FROM trading_signals
WHERE status = 'completed'
GROUP BY symbol, signal_type
ORDER BY symbol, signal_type;

-- View for ML prediction accuracy
CREATE VIEW ml_prediction_accuracy AS
SELECT 
    symbol,
    model_name,
    COUNT(*) as total_predictions,
    COUNT(CASE WHEN is_correct = TRUE THEN 1 END) as correct_predictions,
    ROUND(
        COUNT(CASE WHEN is_correct = TRUE THEN 1 END)::NUMERIC / COUNT(*) * 100, 2
    ) as accuracy_percentage,
    AVG(prediction_accuracy) as avg_accuracy,
    AVG(price_error_percentage) as avg_error_percentage
FROM ml_predictions
WHERE status = 'completed'
GROUP BY symbol, model_name
ORDER BY symbol, accuracy_percentage DESC;

-- =====================================================
-- FUNCTIONS FOR AUTOMATED CLEANUP
-- =====================================================

-- Function to clean old data based on retention policy
CREATE OR REPLACE FUNCTION cleanup_old_data()
RETURNS void AS $$
DECLARE
    retention_days INTEGER;
BEGIN
    -- Get retention period from config
    SELECT value::INTEGER INTO retention_days 
    FROM bot_config 
    WHERE key = 'data_retention_days';
    
    IF retention_days IS NULL THEN
        retention_days := 90; -- Default to 90 days
    END IF;
    
    -- Clean up old data
    DELETE FROM market_data WHERE timestamp < NOW() - INTERVAL '1 day' * retention_days;
    DELETE FROM technical_indicators WHERE timestamp < NOW() - INTERVAL '1 day' * retention_days;
    DELETE FROM sentiment_data WHERE timestamp < NOW() - INTERVAL '1 day' * retention_days;
    DELETE FROM system_logs WHERE timestamp < NOW() - INTERVAL '1 day' * retention_days;
    
    -- Keep completed predictions and signals for performance tracking
    -- Only clean up expired active ones
    DELETE FROM ml_predictions WHERE status = 'active' AND timestamp < NOW() - INTERVAL '1 day' * 7;
    DELETE FROM trading_signals WHERE status = 'active' AND timestamp < NOW() - INTERVAL '1 day' * 7;
    
    RAISE NOTICE 'Cleaned up data older than % days', retention_days;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- TRIGGERS FOR AUTOMATIC UPDATES
-- =====================================================

-- Trigger to update bot_performance when signals are completed
CREATE OR REPLACE FUNCTION update_bot_performance()
RETURNS TRIGGER AS $$
BEGIN
    -- Update performance metrics when signal outcome changes
    IF OLD.outcome = 'pending' AND NEW.outcome IN ('correct', 'incorrect') THEN
        -- Update daily performance
        INSERT INTO bot_performance (date, total_signals, correct_predictions, accuracy_rate)
        VALUES (
            DATE(NEW.timestamp),
            1,
            CASE WHEN NEW.outcome = 'correct' THEN 1 ELSE 0 END,
            CASE WHEN NEW.outcome = 'correct' THEN 100.0 ELSE 0.0 END
        )
        ON CONFLICT (date) DO UPDATE SET
            total_signals = bot_performance.total_signals + 1,
            correct_predictions = bot_performance.correct_predictions + 
                CASE WHEN NEW.outcome = 'correct' THEN 1 ELSE 0 END,
            accuracy_rate = (bot_performance.correct_predictions + 
                CASE WHEN NEW.outcome = 'correct' THEN 1 ELSE 0 END)::NUMERIC / 
                (bot_performance.total_signals + 1) * 100,
            last_updated = NOW();
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger
CREATE TRIGGER trigger_update_bot_performance
    AFTER UPDATE ON trading_signals
    FOR EACH ROW
    EXECUTE FUNCTION update_bot_performance();

-- =====================================================
-- FINAL COMMENTS AND NOTES
-- =====================================================

/*
IMPORTANT NOTES:

1. This schema supports all Alpha Sentinel bot functionality
2. All tables include proper indexing for performance
3. Composite unique constraints prevent duplicate data
4. Views provide easy access to common data combinations
5. Automated cleanup function manages data retention
6. Triggers automatically update performance metrics
7. Configuration table allows runtime parameter changes

TO USE THIS SCHEMA:

1. Run this entire SQL file in your Supabase SQL editor
2. The bot will automatically create the necessary tables
3. Default configuration values are pre-populated
4. All functionality should work immediately after creation

PERFORMANCE CONSIDERATIONS:

- Indexes are created on frequently queried columns
- Composite indexes support multi-column queries
- Data retention policies prevent unlimited growth
- Regular cleanup maintains optimal performance

SECURITY:

- API keys are stored encrypted (implement encryption in application)
- Sensitive data access should be restricted via RLS policies
- Consider implementing row-level security for multi-tenant setups
*/
