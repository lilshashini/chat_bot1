import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Import your prediction model classes
# Make sure to save your model classes in a separate file called 'prediction_model.py'
# and import them here, or include them directly

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

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
    
    def get_available_metrics(self):
        """
        Get list of available metrics from the model
        """
        if not hasattr(self.prediction_model, 'unified_data'):
            return []
        
        return list(self.prediction_model.unified_data.columns)
    
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
        elif 'usage' in query:
            metric = 'usage'
        elif 'output' in query:
            metric = 'output'
        
        # Identify time period
        days_ahead = 7  # default
        if 'next week' in query or '7 days' in query or 'week' in query:
            days_ahead = 7
        elif 'next month' in query or '30 days' in query or 'month' in query:
            days_ahead = 30
        elif 'tomorrow' in query:
            days_ahead = 1
        elif '14 days' in query or '2 weeks' in query or 'two weeks' in query:
            days_ahead = 14
        elif '3 days' in query:
            days_ahead = 3
        elif '5 days' in query:
            days_ahead = 5
        elif '10 days' in query:
            days_ahead = 10
        
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
                        3: "next 3 days",
                        5: "next 5 days",
                        7: "next week", 
                        10: "next 10 days",
                        14: "next 2 weeks",
                        30: "next month"
                    }.get(days_ahead, f"next {days_ahead} days")
                    
                    trend = "↗️ Increasing" if prediction['yhat'].iloc[-1] > prediction['yhat'].iloc[0] else "↘️ Decreasing"
                    
                    return {
                        'type': 'prediction',
                        'metric': metric.title(),
                        'period': time_period,
                        'average': avg_prediction,
                        'min': min_prediction,
                        'max': max_prediction,
                        'trend': trend,
                        'data': prediction
                    }
                else:
                    return {
                        'type': 'error',
                        'message': f"Sorry, I couldn't generate a prediction for {metric}. The model might need more training data."
                    }
            else:
                available_metrics = self.get_available_metrics()
                return {
                    'type': 'error',
                    'message': f"Sorry, I don't have data for {metric}. Available metrics: {', '.join(available_metrics)}"
                }
        else:
            return {
                'type': 'help',
                'message': "I can help you with predictions for production, consumption, utilization, efficiency, usage, and output. Try asking: 'What will be the production next week?'"
            }

def load_trained_model():
    """
    Load your pre-trained model
    Modify this function to load your specific trained model
    """
    try:
        # Try to load from different possible paths
        model_paths = [
            'models/',  # Current directory models folder
            '../models/',  # Parent directory models folder
            './prediction_model.pkl',  # Direct pickle file
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                if os.path.isdir(path):
                    # Load model from directory
                    model_files = [f for f in os.listdir(path) if f.endswith('.pkl')]
                    if model_files:
                        # Load the prediction model class and its data
                        # You'll need to modify this based on how you save your model
                        return None  # Placeholder
                else:
                    # Load direct pickle file
                    return joblib.load(path)
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def create_sample_model():
    """
    Create a sample model for demonstration
    Replace this with your actual model loading
    """
    class SampleModel:
        def __init__(self):
            self.models = {
                'daily_production_production_prophet': 'mock_model',
                'daily_consumption_consumption_prophet': 'mock_model',
                'daily_utilisation_utilisation_prophet': 'mock_model'
            }
            
            # Create sample unified data
            dates = pd.date_range('2023-01-01', '2024-01-01', freq='D')
            self.unified_data = pd.DataFrame({
                'daily_production_production': np.random.uniform(80, 120, len(dates)),
                'daily_consumption_consumption': np.random.uniform(60, 100, len(dates)),
                'daily_utilisation_utilisation': np.random.uniform(70, 90, len(dates)),
            }, index=dates)
        
        def predict_future(self, target_column, days_ahead=7):
            # Mock prediction
            future_dates = pd.date_range(
                start=self.unified_data.index.max() + pd.Timedelta(days=1),
                periods=days_ahead,
                freq='D'
            )
            
            # Generate mock predictions
            base_value = self.unified_data[target_column].mean()
            predictions = np.random.uniform(base_value*0.9, base_value*1.1, days_ahead)
            
            return pd.DataFrame({
                'ds': future_dates,
                'yhat': predictions,
                'yhat_lower': predictions * 0.95,
                'yhat_upper': predictions * 1.05
            })
    
    return SampleModel()

def main():
    st.set_page_config(
        page_title="Production Prediction Chatbot",
        page_icon="🔮",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .chat-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .user-message {
        background: #e3f2fd;
        padding: 0.5rem 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #2196f3;
    }
    .bot-message {
        background: #f3e5f5;
        padding: 0.5rem 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #9c27b0;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>Prediction Model</h1>
        <p>Ask questions about your production data and get AI-powered predictions!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'model' not in st.session_state:
        # Try to load your trained model, fallback to sample
        st.session_state.model = load_trained_model()
        if st.session_state.model is None:
            st.session_state.model = create_sample_model()
            
    
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = RobustChatbot(st.session_state.model)
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Main layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Chat interface
        st.subheader("💬 Ask Your Questions")
        
        # Input area
        user_query = st.text_input(
            "",
            placeholder="e.g., What will be the production next week?",
            key="user_input",
            label_visibility="collapsed"
        )
        
        col_send, col_clear = st.columns([1, 4])
        with col_send:
            send_button = st.button("Send 📤", use_container_width=True)
        with col_clear:
            if st.button("Clear Chat 🗑️", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        
        # Process query
        if send_button and user_query.strip():
            with st.spinner("🤔 Thinking..."):
                response = st.session_state.chatbot.process_query(user_query)
                
                # Add to chat history
                st.session_state.chat_history.append({
                    "user": user_query,
                    "bot": response,
                    "timestamp": datetime.now()
                })
            
            st.rerun()
        
        # Display chat history
        if st.session_state.chat_history:
            st.subheader("💭 Conversation")
            
            for i, chat in enumerate(reversed(st.session_state.chat_history)):
                st.markdown(f"""
                <div class="user-message">
                    <strong>You:</strong> {chat['user']}
                </div>
                """, unsafe_allow_html=True)
                
                # Format bot response based on type
                if isinstance(chat['bot'], dict):
                    if chat['bot']['type'] == 'prediction':
                        bot_response = chat['bot']
                        st.markdown(f"""
                        <div class="bot-message">
                            <strong>🤖 Bot:</strong><br>
                            <div class="metric-card">
                                <h4>{bot_response['metric']} Prediction for {bot_response['period']}</h4>
                                <p>📊 <strong>Average:</strong> {bot_response['average']:.2f}</p>
                                <p>📈 <strong>Range:</strong> {bot_response['min']:.2f} to {bot_response['max']:.2f}</p>
                                <p>📉 <strong>Trend:</strong> {bot_response['trend']}</p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show prediction chart
                        if 'data' in bot_response and not bot_response['data'].empty:
                            st.line_chart(bot_response['data'].set_index('ds')['yhat'])
                    else:
                        st.markdown(f"""
                        <div class="bot-message">
                            <strong>🤖 Bot:</strong> {chat['bot']['message']}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="bot-message">
                        <strong>🤖 Bot:</strong> {chat['bot']}
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown(f"<small>⏰ {chat['timestamp'].strftime('%H:%M:%S')}</small>", unsafe_allow_html=True)
                st.markdown("---")
        
    
        


if __name__ == "__main__":
    main()