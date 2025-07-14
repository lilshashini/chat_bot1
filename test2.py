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

class RobustProductionPredictionModel:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.data = {}
        self.predictions = {}
        self.column_info = {}
    
    def load_csv_with_proper_delimiter(self, file_path):
        """
        Try to load CSV with different delimiters and detect the correct one
        """
        delimiters = [';', ',', '\t', '|']
        
        for delimiter in delimiters:
            try:
                # First, try to read just the header to see column count
                df_test = pd.read_csv(file_path, delimiter=delimiter, nrows=1)
                
                # If we get more than 1 column and reasonable column names, this is likely correct
                if len(df_test.columns) > 1:
                    # Read the full file
                    df = pd.read_csv(file_path, delimiter=delimiter)
                    print(f"Successfully loaded with delimiter '{delimiter}': {df.shape}")
                    print(f"Columns: {list(df.columns[:5])}...")  # Show first 5 columns
                    return df
                    
            except Exception as e:
                continue
        
        # If all delimiters fail, try default comma
        try:
            df = pd.read_csv(file_path)
            print(f"Loaded with default settings: {df.shape}")
            return df
        except Exception as e:
            print(f"Failed to load {file_path}: {e}")
            return None
    
    def detect_datetime_columns(self, df):
        """
        Detect datetime columns in the dataframe
        """
        datetime_cols = []
        
        # Check column names for common datetime indicators
        datetime_keywords = ['date', 'time', 'timestamp', 'created', 'updated', 'day', 'month', 'year']
        
        for col in df.columns:
            col_lower = str(col).lower().strip()
            
            # Check if column name suggests datetime
            if any(keyword in col_lower for keyword in datetime_keywords):
                # Verify the column actually contains datetime-like data
                if self.is_datetime_column(df[col]):
                    datetime_cols.append(col)
        
        return datetime_cols
    
    def is_datetime_column(self, series):
        """
        Check if a pandas series contains datetime-like data
        """
        # Skip if all values are null
        if series.isnull().all():
            return False
            
        # Get a sample of non-null values
        sample_values = series.dropna().head(10)
        if len(sample_values) == 0:
            return False
        
        datetime_like_count = 0
        for val in sample_values:
            try:
                # Try to parse as datetime
                pd.to_datetime(val)
                datetime_like_count += 1
            except:
                pass
        
        # If more than 50% look like datetime, consider it a datetime column
        return datetime_like_count / len(sample_values) > 0.5
    
    def detect_numeric_columns(self, df, exclude_datetime_cols):
        """
        Detect numeric columns that could be used for prediction
        """
        numeric_cols = []
        
        for col in df.columns:
            if col in exclude_datetime_cols:
                continue
            
            # Skip if column name suggests it's not numeric
            col_lower = str(col).lower().strip()
            if any(keyword in col_lower for keyword in ['name', 'id', 'device', 'status', 'info']):
                continue
                
            # Check if column is numeric or can be converted to numeric
            try:
                # Try to convert to numeric
                numeric_series = pd.to_numeric(df[col], errors='coerce')
                
                # If more than 50% of values are numeric (not NaN), consider it numeric
                non_null_count = numeric_series.count()
                total_count = len(df[col])
                
                if non_null_count / total_count > 0.5:
                    numeric_cols.append(col)
                    
            except:
                pass
        
        return numeric_cols
    
    def clean_and_convert_data(self, df, datetime_cols, numeric_cols):
        """
        Clean and convert data to proper types
        """
        cleaned_df = df.copy()
        
        # Convert datetime columns
        for col in datetime_cols:
            try:
                cleaned_df[col] = pd.to_datetime(cleaned_df[col], errors='coerce')
                print(f"Converted {col} to datetime")
            except Exception as e:
                print(f"Warning: Could not convert {col} to datetime: {e}")
        
        # Convert numeric columns
        for col in numeric_cols:
            try:
                # Clean common formatting issues
                if cleaned_df[col].dtype == 'object':
                    # Remove currency symbols, commas, percentages
                    cleaned_df[col] = cleaned_df[col].astype(str).str.replace(r'[,$%]', '', regex=True)
                    # Replace empty strings with NaN
                    cleaned_df[col] = cleaned_df[col].replace('', np.nan)
                
                # Convert to numeric
                cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
                print(f"Converted {col} to numeric")
            except Exception as e:
                print(f"Warning: Could not convert {col} to numeric: {e}")
        
        return cleaned_df
    
    def handle_missing_values(self, df):
        """
        Handle missing values using modern pandas methods
        """
        # For numeric columns, use forward fill then backward fill
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].ffill().bfill()
            # If still NaN, fill with median
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        
        # For datetime columns, forward fill only
        datetime_cols = df.select_dtypes(include=['datetime64[ns]']).columns
        for col in datetime_cols:
            df[col] = df[col].ffill().bfill()
        
        return df
    
    def load_and_preprocess_data(self, file_paths):
        """
        Robust data loading and preprocessing with proper CSV delimiter detection
        """
        print("Loading and preprocessing data...")
        
        processed_data = {}
        
        for key, path in file_paths.items():
            try:
                print(f"\nProcessing {key} from {path}...")
                
                # Load CSV with proper delimiter detection
                df = self.load_csv_with_proper_delimiter(path)
                
                if df is None:
                    print(f"Failed to load {key}")
                    continue
                
                print(f"Original shape: {df.shape}")
                print(f"First few columns: {list(df.columns[:5])}")
                
                # Detect column types
                datetime_cols = self.detect_datetime_columns(df)
                numeric_cols = self.detect_numeric_columns(df, datetime_cols)
                
                print(f"Detected datetime columns: {datetime_cols}")
                print(f"Detected numeric columns: {numeric_cols}")
                
                # Store column information
                self.column_info[key] = {
                    'datetime_cols': datetime_cols,
                    'numeric_cols': numeric_cols,
                    'original_columns': list(df.columns)
                }
                
                # Clean and convert data
                df_cleaned = self.clean_and_convert_data(df, datetime_cols, numeric_cols)
                
                # Handle missing values
                df_cleaned = self.handle_missing_values(df_cleaned)
                
                # Set datetime index if possible
                if datetime_cols:
                    # Try to find the best datetime column (prefer 'date' columns)
                    date_col = None
                    for col in datetime_cols:
                        if 'date' in str(col).lower():
                            date_col = col
                            break
                    
                    if date_col is None:
                        date_col = datetime_cols[0]  # Use first datetime column
                    
                    # Remove rows where datetime is NaN
                    df_cleaned = df_cleaned.dropna(subset=[date_col])
                    
                    if len(df_cleaned) == 0:
                        print(f"No valid dates found in {key}")
                        continue
                    
                    # Sort by date
                    df_cleaned = df_cleaned.sort_values(date_col)
                    
                    # Set as index
                    df_cleaned.set_index(date_col, inplace=True)
                    
                    print(f"Set {date_col} as index")
                
                # Keep all numeric columns
                if numeric_cols:
                    df_final = df_cleaned[numeric_cols]
                else:
                    print(f"No numeric columns found in {key}")
                    continue
                
                print(f"Final shape: {df_final.shape}")
                if not df_final.empty:
                    print(f"Date range: {df_final.index.min()} to {df_final.index.max()}")
                    print(f"Final columns: {list(df_final.columns)}")
                
                processed_data[key] = df_final
                
            except Exception as e:
                print(f"Error processing {key}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        self.data = processed_data
        return processed_data
    
    def create_unified_dataset(self):
        """
        Merge all datasets into a unified time series dataset
        """
        print("\nCreating unified dataset...")
        
        if not self.data:
            print("No data loaded!")
            return None
        
        # Filter out empty datasets
        valid_datasets = {k: v for k, v in self.data.items() if not v.empty}
        
        if not valid_datasets:
            print("No valid datasets found!")
            return None
        
        print(f"Valid datasets: {list(valid_datasets.keys())}")
        
        # Start with an empty dataframe
        unified_df = pd.DataFrame()
        
        for key, df in valid_datasets.items():
            # Rename columns to include dataset name to avoid conflicts
            renamed_df = df.copy()
            new_columns = {}
            
            for col in df.columns:
                # Create descriptive column names
                new_name = f"{key}_{col}"
                new_columns[col] = new_name
            
            renamed_df = renamed_df.rename(columns=new_columns)
            
            if unified_df.empty:
                unified_df = renamed_df
            else:
                # Join on index (date)
                unified_df = unified_df.join(renamed_df, how='outer')
        
        # Handle missing values in unified dataset
        unified_df = self.handle_missing_values(unified_df)
        
        print(f"Unified dataset shape: {unified_df.shape}")
        print(f"Columns: {list(unified_df.columns)}")
        if not unified_df.empty:
            print(f"Date range: {unified_df.index.min()} to {unified_df.index.max()}")
        
        self.unified_data = unified_df
        return unified_df
    
    def get_target_columns(self):
        """
        Identify columns suitable for prediction
        """
        if not hasattr(self, 'unified_data') or self.unified_data.empty:
            print("No unified dataset available!")
            return []
        
        target_keywords = ['production', 'consumption', 'utilisation', 'utilization', 'usage', 'output', 'efficiency']
        target_columns = []
        
        for col in self.unified_data.columns:
            col_lower = str(col).lower()
            if any(keyword in col_lower for keyword in target_keywords):
                target_columns.append(col)
        
        return target_columns
    
    def train_prophet_model(self, target_column):
        """
        Train Facebook Prophet model
        """
        if not PROPHET_AVAILABLE:
            print("Prophet not available. Skipping Prophet model.")
            return None
            
        print(f"Training Prophet model for {target_column}...")
        
        try:
            # Prepare data for Prophet
            prophet_data = pd.DataFrame({
                'ds': self.unified_data.index,
                'y': self.unified_data[target_column]
            }).reset_index(drop=True)
            
            # Remove any remaining NaN values
            prophet_data = prophet_data.dropna()
            
            if len(prophet_data) < 10:
                print(f"Not enough data points for {target_column} ({len(prophet_data)} points)")
                return None
            
            # Train Prophet model
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=True,
                changepoint_prior_scale=0.05
            )
            
            model.fit(prophet_data)
            print(f"Successfully trained Prophet model for {target_column}")
            
            return model
            
        except Exception as e:
            print(f"Error training Prophet model for {target_column}: {e}")
            return None
    
    def train_all_models(self):
        """
        Train models for all suitable target variables
        """
        print("\nTraining models for all target variables...")
        
        if not hasattr(self, 'unified_data') or self.unified_data.empty:
            print("Please create unified dataset first!")
            return
        
        target_columns = self.get_target_columns()
        
        if not target_columns:
            print("No suitable target columns found!")
            print("Looking for columns containing: production, consumption, utilization, efficiency")
            print(f"Available columns: {list(self.unified_data.columns)}")
            return
        
        print(f"Target columns identified: {target_columns}")
        
        for target_col in target_columns:
            print(f"\n--- Training models for {target_col} ---")
            
            # Check if column has enough non-null values
            non_null_count = self.unified_data[target_col].count()
            if non_null_count < 10:
                print(f"Skipping {target_col}: insufficient data ({non_null_count} points)")
                continue
            
            # Train Prophet model
            prophet_model = self.train_prophet_model(target_col)
            if prophet_model:
                self.models[f"{target_col}_prophet"] = prophet_model
    
    def predict_future(self, target_column, days_ahead=7):
        """
        Make future predictions using Prophet model
        """
        model_key = f"{target_column}_prophet"
        
        if model_key not in self.models:
            print(f"Model {model_key} not found!")
            available_models = list(self.models.keys())
            print(f"Available models: {available_models}")
            return None
        
        try:
            model = self.models[model_key]
            
            # Create future dataframe
            future = model.make_future_dataframe(periods=days_ahead)
            forecast = model.predict(future)
            
            # Return predictions for future dates only
            future_predictions = forecast.tail(days_ahead)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
            return future_predictions
            
        except Exception as e:
            print(f"Error making predictions: {e}")
            return None
    
    def save_models(self, save_path='models_new/'):
        """
        Save trained models and metadata
        """
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # Save models
        for model_name, model in self.models.items():
            try:
                joblib.dump(model, f"{save_path}/{model_name}.pkl")
            except Exception as e:
                print(f"Error saving {model_name}: {e}")
        
        # Save column information
        joblib.dump(self.column_info, f"{save_path}/column_info.pkl")
        
        print(f"Models and metadata saved to {save_path}")

# Simplified Chatbot for the robust model
class RobustChatbot:
    def __init__(self, prediction_model):
        self.prediction_model = prediction_model
    
    def find_matching_column(self, metric_name):
        """
        Find column matching the requested metric
        """
        if not hasattr(self.prediction_model, 'unified_data'):
            return None
        
        metric_lower = metric_name.lower()
        
        for col in self.prediction_model.unified_data.columns:
            if metric_lower in col.lower():
                return col
        
        return None
    
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
            metric = 'utilization'
        elif 'efficiency' in query:
            metric = 'efficiency'
        
        # Identify time period
        days_ahead = 7  # default
        if 'next week' in query or '7 days' in query:
            days_ahead = 7
        elif 'next month' in query or '30 days' in query:
            days_ahead = 30
        elif 'tomorrow' in query:
            days_ahead = 1
        elif '14 days' in query or '2 weeks' in query:
            days_ahead = 14
        
        if metric:
            # Find matching column
            target_col = self.find_matching_column(metric)
            
            if target_col:
                # Get prediction
                prediction = self.prediction_model.predict_future(target_col, days_ahead=days_ahead)
                
                if prediction is not None and len(prediction) > 0:
                    avg_prediction = prediction['yhat'].mean()
                    min_prediction = prediction['yhat'].min()
                    max_prediction = prediction['yhat'].max()
                    
                    time_period = {
                        1: "tomorrow",
                        7: "next week", 
                        14: "next 2 weeks",
                        30: "next month"
                    }.get(days_ahead, f"next {days_ahead} days")
                    
                    return f"""
**{metric.title()} Prediction for {time_period}:**
• Average: {avg_prediction:.2f}
• Range: {min_prediction:.2f} to {max_prediction:.2f}
• Trend: {'↗️ Increasing' if prediction['yhat'].iloc[-1] > prediction['yhat'].iloc[0] else '↘️ Decreasing'}
                    """.strip()
                else:
                    return f"Sorry, I couldn't generate a prediction for {metric}. The model might need more training data."
            else:
                available_cols = list(self.prediction_model.unified_data.columns)
                return f"Sorry, I don't have data for {metric}. Available metrics: {', '.join(available_cols)}"
        else:
            return "I can help you with predictions for production, consumption, utilization, and efficiency. Try asking: 'What will be the production next week?'"

# Example usage with improved error handling
def robust_main():
    """
    Robust main function with comprehensive error handling
    """
    # Initialize the prediction model
    model = RobustProductionPredictionModel()
    
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
        
        if not processed_data:
            print("No data could be loaded. Please check your file paths and formats.")
            return
        
        # Create unified dataset
        unified_data = model.create_unified_dataset()
        
        if unified_data is None or unified_data.empty:
            print("Could not create unified dataset. Using individual datasets.")
            # Use the first available dataset as fallback
            if processed_data:
                first_key = list(processed_data.keys())[0]
                model.unified_data = processed_data[first_key]
                print(f"Using {first_key} as primary dataset")
            else:
                print("No datasets available.")
                return
        
        # Train all models
        model.train_all_models()
        
        if not model.models:
            print("No models were trained successfully.")
            return
        
        # Save models
        model.save_models()
        
        # Initialize chatbot
        chatbot = RobustChatbot(model)
        
        # Example queries
        example_queries = [
            "What is the expected production next week give with date range?",
            "How about next month's consumption?",
            "What will be the utilization next week?",
            "How about next month's efficiency?",
            "Predict tomorrow's production"
        ]
        
        print("\n" + "="*50)
        print("CHATBOT DEMO")
        print("="*50)
        
        for query in example_queries:
            try:
                response = chatbot.process_query(query)
                print(f"\nQuery: {query}")
                print(f"Response: {response}")
            except Exception as e:
                print(f"Error processing query '{query}': {e}")
                
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    robust_main()