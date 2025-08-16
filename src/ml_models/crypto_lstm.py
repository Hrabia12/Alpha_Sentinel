import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import os


class CryptoLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(CryptoLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        lstm_out, _ = self.lstm(x, (h0, c0))

        # Use the last output
        lstm_out = self.dropout(lstm_out[:, -1, :])
        prediction = self.linear(lstm_out)

        return prediction


class CryptoPredictor:
    def __init__(
        self,
        sequence_length=60,
        features=["close", "volume", "rsi", "macd", "bb_position"],
    ):
        self.sequence_length = sequence_length
        self.features = features
        self.scaler = MinMaxScaler()
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def prepare_features(self, df):
        """Prepare features for training"""
        df = df.copy()

        # Calculate Bollinger Band position
        if "bb_upper" in df.columns and "bb_lower" in df.columns:
            bb_range = df["bb_upper"] - df["bb_lower"]
            df["bb_position"] = (df["close"] - df["bb_lower"]) / bb_range
            df["bb_position"] = df["bb_position"].fillna(
                0.5
            )  # Middle position if no data

        # Handle missing values
        for feature in self.features:
            if feature in df.columns:
                df[feature] = (
                    df[feature].fillna(method="forward").fillna(method="backward")
                )

        return df[self.features].values

    def create_sequences(self, data):
        """Create sequences for LSTM input"""
        X, y = [], []
        for i in range(self.sequence_length, len(data)):
            X.append(data[i - self.sequence_length : i])
            y.append(data[i, 0])  # Predict close price (first feature)

        return np.array(X), np.array(y)

    def train(self, df, epochs=100, batch_size=32, learning_rate=0.001):
        """Train the LSTM model"""
        # Prepare data
        feature_data = self.prepare_features(df)

        if len(feature_data) < self.sequence_length + 50:
            raise ValueError(
                f"Not enough data. Need at least {self.sequence_length + 50} records"
            )

        # Scale the data
        scaled_data = self.scaler.fit_transform(feature_data)

        # Create sequences
        X, y = self.create_sequences(scaled_data)

        # Split data (80% train, 20% test)
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Convert to PyTorch tensors
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)
        X_test = torch.FloatTensor(X_test).to(self.device)
        y_test = torch.FloatTensor(y_test).to(self.device)

        # Initialize model
        input_size = len(self.features)
        self.model = CryptoLSTM(
            input_size=input_size, hidden_size=64, num_layers=2, output_size=1
        ).to(self.device)

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # Training loop
        self.model.train()
        for epoch in range(epochs):
            # Forward pass
            outputs = self.model(X_train)
            loss = criterion(outputs.squeeze(), y_train)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                # Validation
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_test)
                    val_loss = criterion(val_outputs.squeeze(), y_test)
                print(
                    f"Epoch [{epoch + 1}/{epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}"
                )
                self.model.train()

        # Final evaluation
        self.model.eval()
        with torch.no_grad():
            train_pred = self.model(X_train).squeeze().cpu().numpy()
            test_pred = self.model(X_test).squeeze().cpu().numpy()

            # Inverse transform predictions
            train_pred_orig = self.inverse_transform_prediction(train_pred)
            test_pred_orig = self.inverse_transform_prediction(test_pred)
            y_train_orig = self.inverse_transform_prediction(y_train.cpu().numpy())
            y_test_orig = self.inverse_transform_prediction(y_test.cpu().numpy())

            train_rmse = np.sqrt(mean_squared_error(y_train_orig, train_pred_orig))
            test_rmse = np.sqrt(mean_squared_error(y_test_orig, test_pred_orig))

            print(f"Training RMSE: {train_rmse:.2f}")
            print(f"Testing RMSE: {test_rmse:.2f}")

        return {"train_rmse": train_rmse, "test_rmse": test_rmse}

    def inverse_transform_prediction(self, prediction):
        """Inverse transform prediction to original scale"""
        # Create dummy array with same shape as original features
        dummy = np.zeros((len(prediction), len(self.features)))
        dummy[:, 0] = prediction  # Close price is first feature

        # Inverse transform and return only close price
        return self.scaler.inverse_transform(dummy)[:, 0]

    def predict(self, df, steps_ahead=1):
        """Make predictions on new data"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        self.model.eval()

        # Prepare features
        feature_data = self.prepare_features(df)

        if len(feature_data) < self.sequence_length:
            raise ValueError(
                f"Need at least {self.sequence_length} records for prediction"
            )

        # Use last sequence for prediction
        scaled_data = self.scaler.transform(feature_data)
        last_sequence = scaled_data[-self.sequence_length :].reshape(
            1, self.sequence_length, -1
        )

        predictions = []
        current_sequence = torch.FloatTensor(last_sequence).to(self.device)

        with torch.no_grad():
            for _ in range(steps_ahead):
                pred = self.model(current_sequence)
                predictions.append(pred.item())

                # Update sequence for next prediction
                # (This is simplified - in practice, you'd need other features too)
                new_row = current_sequence[:, -1, :].clone()
                new_row[:, 0] = pred  # Update close price

                current_sequence = torch.cat(
                    [current_sequence[:, 1:, :], new_row.unsqueeze(1)], dim=1
                )

        # Inverse transform predictions
        predictions_orig = self.inverse_transform_prediction(np.array(predictions))

        return predictions_orig

    def save_model(self, filepath):
        """Save model and scaler"""
        if self.model is None:
            raise ValueError("No model to save")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Save model state dict
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "scaler": self.scaler,
                "features": self.features,
                "sequence_length": self.sequence_length,
                "model_config": {
                    "input_size": len(self.features),
                    "hidden_size": 64,
                    "num_layers": 2,
                    "output_size": 1,
                },
            },
            filepath,
        )

        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load model and scaler"""
        import torch
        import torch.serialization
        
        # First try with weights_only=False for backward compatibility
        try:
            checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        except Exception as e:
            # If that fails, try with safe globals
            try:
                torch.serialization.add_safe_globals(['sklearn.preprocessing._data.MinMaxScaler'])
                checkpoint = torch.load(filepath, map_location=self.device, weights_only=True)
            except Exception as e2:
                # If both fail, provide clear error message
                error_msg = f"Failed to load model from {filepath}. "
                error_msg += f"First attempt failed: {str(e)}. "
                error_msg += f"Fallback attempt failed: {str(e2)}"
                raise RuntimeError(error_msg)

        self.scaler = checkpoint["scaler"]
        self.features = checkpoint["features"]
        self.sequence_length = checkpoint["sequence_length"]

        config = checkpoint["model_config"]
        self.model = CryptoLSTM(**config).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        print(f"Model loaded from {filepath}")


# Test the model
def test_model():
    # Generate sample data
    dates = pd.date_range(start="2024-01-01", end="2024-03-01", freq="1H")
    np.random.seed(42)

    # More realistic crypto price simulation
    returns = np.random.normal(0.0001, 0.02, len(dates))  # Small positive drift
    prices = 50000 * np.exp(np.cumsum(returns))

    sample_data = pd.DataFrame(
        {
            "timestamp": dates,
            "close": prices,
            "volume": np.random.lognormal(10, 1, len(dates)),
            "rsi": 30 + 40 * np.random.random(len(dates)),  # RSI between 30-70
            "macd": np.random.normal(0, 100, len(dates)),
            "bb_upper": prices * 1.05,
            "bb_lower": prices * 0.95,
        }
    )

    # Train model
    predictor = CryptoPredictor()
    print("Training model...")
    results = predictor.train(sample_data, epochs=50)

    # Make predictions
    print("Making predictions...")
    predictions = predictor.predict(sample_data, steps_ahead=5)
    print(f"Next 5 predictions: {predictions}")

    # Save model
    predictor.save_model("models/crypto_lstm_test.pth")


if __name__ == "__main__":
    test_model()
