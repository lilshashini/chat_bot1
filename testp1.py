# Time Series Prediction Model for Production, Consumption, and Utilization
# This script handles data preprocessing, model training, and chatbot integration

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import joblib

# Time Series Libraries
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    print("Prophet not installed. Install with: pip install prophet")
    PROPHET_AVAILABLE = False

# Statistical Libraries
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

class ProductionPredictionModel:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.data = {}
        self.predictions = {}
        
    def load_and_preprocess_data(self, file_paths):
        """
        Load and preprocess CSV files
        
        Args:
            file_paths (dict): Dictionary with keys like 'daily_consumption', 'daily_production', etc.
                             and values as file paths
        """
        print("Loading and preprocessing data...")
        
        # Load all CSV files
        raw_data = {}
        for key, path in file_paths.items():
            try:
                df = pd.read_csv(path)
                raw_data[key] = df
                print(f"Loaded {key}: {df.shape}")
            except Exception as e:
                print(f"Error loading {key}: {e}")
                
        # Preprocess each dataset
        processed_data = {}
        
        for key, df in raw_data.items():
            # Find date column (common names)
            date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            if not date_columns:
                # If no date column found, assume first column is date
                date_columns = [df.columns[0]]
            
            date_col = date_columns[0]
            
            # Convert to datetime
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            
            # Remove rows with invalid dates
            df = df.dropna(subset=[date_col])
            
            # Sort by date
            df = df.sort_values(date_col)
            
            # Set date as index
            df.set_index(date_col, inplace=True)
            
            # Handle missing values
            df = df.fillna(method='forward').fillna(method='backward')
            
            processed_data[key] = df
            
        self.data = processed_data
        return processed_data
    
    def create_unified_dataset(self):
        """
        Merge all datasets into a unified time series dataset
        """
        print("Creating unified dataset...")
        
        # Start with an empty dataframe
        unified_df = pd.DataFrame()
        
        for key, df in self.data.items():
            # Rename columns to include dataset name
            renamed_df = df.copy()
            for col in df.columns:
                if col.lower() not in ['date', 'time']:
                    new_name = f"{key}_{col}"
                    renamed_df.rename(columns={col: new_name}, inplace=True)
            
            if unified_df.empty:
                unified_df = renamed_df
            else:
                unified_df = unified_df.join(renamed_df, how='outer')
        
        # Fill missing values
        unified_df = unified_df.fillna(method='forward').fillna(method='backward')
        
        # If we have hourly data, aggregate to daily
        if len(unified_df) > 1000:  # Assuming hourly data if more than 1000 records
            unified_df = unified_df.resample('D').mean()
        
        self.unified_data = unified_df
        return unified_df
    
    def visualize_data(self):
        """
        Create visualizations for the time series data
        """
        print("Creating visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot main metrics
        metrics = ['production', 'consumption', 'utilisation']  # Note: British spelling
        
        for i, metric in enumerate(metrics):
            # Find columns containing the metric name
            metric_cols = [col for col in self.unified_data.columns if metric in col.lower()]
            
            if metric_cols:
                ax = axes[i//2, i%2]
                for col in metric_cols:
                    ax.plot(self.unified_data.index, self.unified_data[col], label=col)
                ax.set_title(f'{metric.title()} Over Time')
                ax.set_xlabel('Date')
                ax.set_ylabel('Value')
                ax.legend()
                ax.tick_params(axis='x', rotation=45)
        
        # Correlation heatmap
        ax = axes[1, 1]
        correlation_matrix = self.unified_data.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Correlation Matrix')
        
        plt.tight_layout()
        plt.show()
    
    def prepare_features(self, target_column, lookback_days=30):
        """
        Prepare features for machine learning models
        """
        df = self.unified_data.copy()
        
        # Create lagged features
        for i in range(1, lookback_days + 1):
            df[f'{target_column}_lag_{i}'] = df[target_column].shift(i)
        
        # Create rolling statistics
        for window in [7, 14, 30]:
            df[f'{target_column}_rolling_mean_{window}'] = df[target_column].rolling(window=window).mean()
            df[f'{target_column}_rolling_std_{window}'] = df[target_column].rolling(window=window).std()
        
        # Create time-based features
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['year'] = df.index.year
        
        # Drop rows with NaN values
        df = df.dropna()
        
        return df
    
    def train_prophet_model(self, target_column):
        """
        Train Facebook Prophet model for time series forecasting
        """
        if not PROPHET_AVAILABLE:
            print("Prophet not available. Skipping Prophet model.")
            return None
            
        print(f"Training Prophet model for {target_column}...")
        
        # Prepare data for Prophet
        prophet_data = pd.DataFrame({
            'ds': self.unified_data.index,
            'y': self.unified_data[target_column]
        }).reset_index(drop=True)
        
        # Train Prophet model
        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05
        )
        model.fit(prophet_data)
        
        return model
    
    def train_ml_model(self, target_column, model_type='rf'):
        """
        Train machine learning model
        """
        print(f"Training {model_type} model for {target_column}...")
        
        # Prepare features
        feature_df = self.prepare_features(target_column)
        
        # Separate features and target
        feature_cols = [col for col in feature_df.columns if col != target_column]
        X = feature_df[feature_cols]
        y = feature_df[target_column]
        
        # Split data (80% train, 20% test)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        if model_type == 'rf':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            model = LinearRegression()
        
        model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"Model Performance for {target_column}:")
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²: {r2:.4f}")
        
        return model, scaler
    
    def train_all_models(self):
        """
        Train models for all target variables
        """
        print("Training models for all target variables...")
        
        # Identify target columns
        target_columns = []
        for col in self.unified_data.columns:
            if any(keyword in col.lower() for keyword in ['production', 'consumption', 'utilisation', 'utilization']):
                target_columns.append(col)
        
        print(f"Target columns identified: {target_columns}")
        
        for target_col in target_columns:
            print(f"\n--- Training models for {target_col} ---")
            
            # Train Prophet model
            if PROPHET_AVAILABLE:
                prophet_model = self.train_prophet_model(target_col)
                self.models[f"{target_col}_prophet"] = prophet_model
            
            # Train Random Forest model
            rf_model, rf_scaler = self.train_ml_model(target_col, 'rf')
            self.models[f"{target_col}_rf"] = rf_model
            self.scalers[f"{target_col}_rf"] = rf_scaler
    
    def predict_future(self, target_column, days_ahead=7, model_type='prophet'):
        """
        Make future predictions
        """
        model_key = f"{target_column}_{model_type}"
        
        if model_key not in self.models:
            print(f"Model {model_key} not found!")
            return None
        
        if model_type == 'prophet' and PROPHET_AVAILABLE:
            model = self.models[model_key]
            
            # Create future dataframe
            future = model.make_future_dataframe(periods=days_ahead)
            forecast = model.predict(future)
            
            # Return predictions for future dates only
            future_predictions = forecast.tail(days_ahead)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
            return future_predictions
        
        elif model_type == 'rf':
            # For ML models, we need to create features for future dates
            model = self.models[model_key]
            scaler = self.scalers[model_key]
            
            # This is a simplified approach - in practice, you'd need to 
            # generate features for future dates based on recent data
            last_values = self.unified_data[target_column].tail(30).values
            
            predictions = []
            for i in range(days_ahead):
                # Simple approach: use last known pattern
                # In practice, you'd create proper features
                pred_value = np.mean(last_values) + np.random.normal(0, np.std(last_values) * 0.1)
                predictions.append(pred_value)
            
            future_dates = pd.date_range(
                start=self.unified_data.index[-1] + timedelta(days=1),
                periods=days_ahead,
                freq='D'
            )
            
            return pd.DataFrame({
                'ds': future_dates,
                'yhat': predictions
            })
    
    def save_models(self, save_path='models/'):
        """
        Save trained models
        """
        import os
        os.makedirs(save_path, exist_ok=True)
        
        for model_name, model in self.models.items():
            if 'prophet' in model_name:
                # Save Prophet model
                with open(f"{save_path}/{model_name}.json", 'w') as f:
                    import json
                    json.dump(model.to_json(), f)
            else:
                # Save scikit-learn model
                joblib.dump(model, f"{save_path}/{model_name}.pkl")
        
        # Save scalers
        for scaler_name, scaler in self.scalers.items():
            joblib.dump(scaler, f"{save_path}/{scaler_name}.pkl")
        
        print(f"Models saved to {save_path}")
    
    def load_models(self, load_path='models/'):
        """
        Load saved models
        """
        import os
        
        for file in os.listdir(load_path):
            if file.endswith('.pkl'):
                model_name = file.replace('.pkl', '')
                if 'scaler' in model_name or model_name in self.scalers:
                    self.scalers[model_name] = joblib.load(f"{load_path}/{file}")
                else:
                    self.models[model_name] = joblib.load(f"{load_path}/{file}")
        
        print("Models loaded successfully!")

# Chatbot Integration Class
class ProductionChatbot:
    def __init__(self, prediction_model):
        self.prediction_model = prediction_model
    
    def process_query(self, user_query):
        """
        Process user queries and return predictions
        """
        query = user_query.lower()
        
        # Identify the metric
        metric = None
        if 'production' in query:
            metric = 'production'
        elif 'consumption' in query:
            metric = 'consumption'
        elif 'utilization' in query or 'utilisation' in query:
            metric = 'utilisation'
        
        # Identify time period
        days_ahead = 7  # default
        if 'next week' in query:
            days_ahead = 7
        elif 'next month' in query:
            days_ahead = 30
        elif 'tomorrow' in query:
            days_ahead = 1
        
        if metric:
            # Find the appropriate column
            target_columns = [col for col in self.prediction_model.unified_data.columns 
                            if metric in col.lower()]
            
            if target_columns:
                target_col = target_columns[0]  # Use first matching column
                
                # Get prediction
                prediction = self.prediction_model.predict_future(
                    target_col, days_ahead=days_ahead, model_type='prophet'
                )
                
                if prediction is not None:
                    avg_prediction = prediction['yhat'].mean()
                    
                    time_period = "next week" if days_ahead == 7 else "next month" if days_ahead == 30 else "tomorrow"
                    
                    return f"The predicted average {metric} for {time_period} is: {avg_prediction:.2f}"
                else:
                    return f"Sorry, I couldn't generate a prediction for {metric}."
            else:
                return f"Sorry, I don't have data for {metric}."
        else:
            return "I can help you with predictions for production, consumption, and utilization. Please ask about next week or next month forecasts."

# Example usage
def main():
    # Initialize the prediction model
    model = ProductionPredictionModel()
    
    # Define file paths (update these with your actual file paths)
    file_paths = {
        'daily_consumption': '/Users/shashinisathsaranilaksiri/Desktop/Althinect/AI_Chatty/consumption.csv',
        'daily_production': '/Users/shashinisathsaranilaksiri/Desktop/Althinect/AI_Chatty/production.csv',
        'daily_utilisation': '/Users/shashinisathsaranilaksiri/Desktop/Althinect/AI_Chatty/utilisation.csv',
        'hourly_production': '/Users/shashinisathsaranilaksiri/Desktop/Althinect/AI_Chatty/hourly_production.csv'
    }
    
    try:
        # Load and preprocess data
        processed_data = model.load_and_preprocess_data(file_paths)
        
        # Create unified dataset
        unified_data = model.create_unified_dataset()
        print(f"Unified dataset shape: {unified_data.shape}")
        print(f"Columns: {list(unified_data.columns)}")
        
        # Visualize data (optional)
        # model.visualize_data()
        
        # Train all models
        model.train_all_models()
        
        # Save models
        model.save_models()
        
        # Initialize chatbot
        chatbot = ProductionChatbot(model)
        
        # Example queries
        example_queries = [
            "What is the expected production next week?",
            "How about next month's consumption?",
            "What will be the utilization next week?",
            "Predict tomorrow's production"
        ]
        
        print("\n--- Chatbot Demo ---")
        for query in example_queries:
            response = chatbot.process_query(query)
            print(f"Query: {query}")
            print(f"Response: {response}\n")
            
    except Exception as e:
        print(f"Error in main execution: {e}")
        print("Please ensure your CSV files exist and have the correct format.")

if __name__ == "__main__":
    main()