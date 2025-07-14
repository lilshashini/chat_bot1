# Enhanced Industrial Prediction Chatbot with Natural Chat Interface
# Combines time-series forecasting and parameter-based predictions

import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os
import json
import re
from datetime import datetime, timedelta
from dotenv import load_dotenv
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from groq import Groq
from openai import OpenAI

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Enhanced Industrial Prediction Assistant",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

class EnhancedIndustrialChatbot:
    def __init__(self):
        self.models = {}
        self.groq_client = None
        self.openai_client = None
        self.api_provider = None
        self.setup_api_clients()
        self.load_models()
    
    def setup_api_clients(self):
        """Initialize API clients based on available keys"""
        # Try Groq first
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            try:
                self.groq_client = Groq(api_key=groq_key)
                self.api_provider = "Groq"
                return
            except Exception as e:
                st.warning(f"Groq initialization failed: {e}")
        
        # Try OpenAI
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            try:
                self.openai_client = OpenAI(api_key=openai_key)
                self.api_provider = "OpenAI"
                return
            except Exception as e:
                st.warning(f"OpenAI initialization failed: {e}")
        
        # No API available
        self.api_provider = None
    
    def load_models(self):
        """Load prediction models from multiple possible locations"""
        possible_paths = [
            'models/all_models.pkl',
            '/Users/shashinisathsaranilaksiri/Desktop/Althinect/Prediction_model/models/all_models.pkl',
            'all_models.pkl'
        ]
        
        for path in possible_paths:
            try:
                with open(path, 'rb') as f:
                    self.models = pickle.load(f)
                st.success(f"✅ Models loaded from: {path}")
                return True
            except FileNotFoundError:
                continue
            except Exception as e:
                st.error(f"Error loading from {path}: {e}")
                continue
        
        st.error("❌ No model files found. Please ensure 'all_models.pkl' is available.")
        return False
    
    def detect_query_type(self, query):
        """Detect if query is time-series or parameter-based"""
        time_keywords = ['tomorrow', 'next week', 'next month', 'forecast', 'future', 'predict', 'days ahead', 'weeks ahead', 'months ahead']
        param_keywords = ['capacity', 'demand', 'supply', 'inventory', 'workforce', 'equipment', 'if', 'with', 'given', 'when']
        
        query_lower = query.lower()
        
        time_score = sum(1 for keyword in time_keywords if keyword in query_lower)
        param_score = sum(1 for keyword in param_keywords if keyword in query_lower)
        
        # Also check for time-related patterns
        time_patterns = [r'\d+\s+(day|week|month|year)s?\s+(ahead|from now)', r'next\s+\d+\s+(day|week|month)s?']
        for pattern in time_patterns:
            if re.search(pattern, query_lower):
                time_score += 2
        
        if time_score > param_score:
            return "time_series"
        elif param_score > time_score:
            return "parameter_based"
        else:
            return "unclear"
    
    def parse_time_series_query(self, query):
        """Parse time-series related query"""
        if not self.api_provider:
            return self.simple_time_parse(query)
        
        try:
            available_categories = list(self.models.keys()) if self.models else []
            
            system_prompt = f"""
            You are an assistant that extracts time-series prediction parameters from user queries.
            Available categories: {available_categories}
            
            Extract and return ONLY a JSON object with these fields:
            - category: one of {available_categories} or best guess
            - device: device name if mentioned, or null
            - days_ahead: number (1 for tomorrow, 7 for week, 30 for month)
            - confidence: float between 0-1
            
            Example: {{"category": "production", "device": "Machine1", "days_ahead": 7, "confidence": 0.9}}
            """
            
            if self.api_provider == "Groq":
                response = self.groq_client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": query}
                    ],
                    max_tokens=150,
                    temperature=0.1
                )
            else:  # OpenAI
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": query}
                    ],
                    max_tokens=150,
                    temperature=0.1
                )
            
            content = response.choices[0].message.content
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return json.loads(content)
                
        except Exception as e:
            st.error(f"Error parsing time-series query: {e}")
            return self.simple_time_parse(query)
    
    def parse_parameter_query(self, query):
        """Parse parameter-based query"""
        if not self.api_provider:
            return self.simple_param_parse(query)
        
        try:
            system_prompt = """
            You are an AI assistant that extracts numerical parameters from natural language queries for predicting utilization, production, and consumption.
            
            Extract relevant numerical values and categorize them. Return ONLY a JSON object with the following structure:
            {
                "parameters": {
                    "feature1": value1,
                    "feature2": value2,
                    ...
                },
                "prediction_type": "utilization" | "production" | "consumption",
                "confidence": "high" | "medium" | "low"
            }
            
            Common features might include:
            - capacity, demand, supply, inventory, workforce, equipment, time_period, season, etc.
            
            If you cannot extract clear numerical values, set confidence to "low".
            
            Example:
            Input: "What will be the utilization if capacity is 1000 and demand is 800?"
            Output: {"parameters": {"capacity": 1000, "demand": 800}, "prediction_type": "utilization", "confidence": "high"}
            """
            
            if self.api_provider == "Groq":
                response = self.groq_client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": query}
                    ],
                    max_tokens=500,
                    temperature=0.1
                )
            else:  # OpenAI
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": query}
                    ],
                    max_tokens=500,
                    temperature=0.1
                )
            
            content = response.choices[0].message.content
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return json.loads(content)
                
        except Exception as e:
            st.error(f"Error parsing parameter query: {e}")
            return self.simple_param_parse(query)
    
    def simple_time_parse(self, query):
        """Fallback simple time-series parsing"""
        query_lower = query.lower()
        
        # Extract category
        category = None
        if any(word in query_lower for word in ['production', 'produce', 'manufacturing']):
            category = 'production'
        elif any(word in query_lower for word in ['consumption', 'consume', 'usage']):
            category = 'consumption'
        elif any(word in query_lower for word in ['utilization', 'efficiency']):
            category = 'utilization'
        
        # Extract days
        days_ahead = 7
        if any(word in query_lower for word in ['tomorrow', 'next day']):
            days_ahead = 1
        elif any(word in query_lower for word in ['week', 'weekly']):
            days_ahead = 7
        elif any(word in query_lower for word in ['month', 'monthly']):
            days_ahead = 30
        
        # Extract specific numbers
        number_match = re.search(r'(\d+)\s*(day|week|month)s?', query_lower)
        if number_match:
            num = int(number_match.group(1))
            unit = number_match.group(2)
            if unit == 'day':
                days_ahead = num
            elif unit == 'week':
                days_ahead = num * 7
            elif unit == 'month':
                days_ahead = num * 30
        
        return {
            'category': category,
            'device': None,
            'days_ahead': days_ahead,
            'confidence': 0.7
        }
    
    def simple_param_parse(self, query):
        """Fallback simple parameter parsing"""
        query_lower = query.lower()
        
        # Extract prediction type
        prediction_type = "production"
        if "utilization" in query_lower:
            prediction_type = "utilization"
        elif "consumption" in query_lower:
            prediction_type = "consumption"
        
        # Extract numbers
        numbers = re.findall(r'\d+(?:\.\d+)?', query_lower)
        parameters = {}
        
        if len(numbers) >= 2:
            parameters['capacity'] = float(numbers[0])
            parameters['demand'] = float(numbers[1])
            confidence = "high"
        elif len(numbers) == 1:
            parameters['value'] = float(numbers[0])
            confidence = "medium"
        else:
            confidence = "low"
        
        return {
            "parameters": parameters,
            "prediction_type": prediction_type,
            "confidence": confidence
        }
    
    def get_time_series_prediction(self, category, device, days_ahead=7):
        """Generate time-series prediction"""
        try:
            if not self.models or category not in self.models:
                return None, f"Category '{category}' not available"
            
            # Get device or use first available
            category_models = self.models[category]
            if device and device in category_models:
                model = category_models[device]
            else:
                device = list(category_models.keys())[0]
                model = category_models[device]
            
            # Generate prediction
            future = model.make_future_dataframe(periods=days_ahead, freq='D')
            forecast = model.predict(future)
            
            future_predictions = forecast.tail(days_ahead)
            
            return {
                'category': category,
                'device': device,
                'days_ahead': days_ahead,
                'predictions': future_predictions,
                'average': round(future_predictions['yhat'].mean(), 2),
                'total': round(future_predictions['yhat'].sum(), 2),
                'min': round(future_predictions['yhat'].min(), 2),
                'max': round(future_predictions['yhat'].max(), 2)
            }, None
            
        except Exception as e:
            return None, f"Time-series prediction error: {str(e)}"
    
    def get_parameter_prediction(self, parameters, prediction_type):
        """Generate parameter-based prediction"""
        try:
            if not self.models or not parameters:
                return None, "No parameters or models available"
            
            # Convert parameters to DataFrame
            input_data = pd.DataFrame([parameters])
            
            # Select appropriate model
            if prediction_type in self.models:
                model_key = prediction_type
            else:
                model_key = list(self.models.keys())[0]
            
            # Get first model from the category
            model = list(self.models[model_key].values())[0]
            
            # Make prediction
            prediction = model.predict(input_data)
            
            return {
                'prediction_type': prediction_type,
                'model_used': model_key,
                'parameters': parameters,
                'result': round(float(prediction[0]), 2)
            }, None
            
        except Exception as e:
            return None, f"Parameter prediction error: {str(e)}"
    
    def create_time_series_chart(self, prediction_data):
        """Create time-series chart"""
        df = prediction_data['predictions']
        
        fig = go.Figure()
        
        # Add prediction line
        fig.add_trace(go.Scatter(
            x=df['ds'],
            y=df['yhat'],
            mode='lines+markers',
            name='Prediction',
            line=dict(color='#1f77b4', width=3)
        ))
        
        # Add confidence interval
        fig.add_trace(go.Scatter(
            x=df['ds'],
            y=df['yhat_upper'],
            mode='lines',
            name='Upper Bound',
            line=dict(color='rgba(31, 119, 180, 0.3)', width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=df['ds'],
            y=df['yhat_lower'],
            mode='lines',
            name='Lower Bound',
            line=dict(color='rgba(31, 119, 180, 0.3)', width=0),
            fill='tonexty',
            fillcolor='rgba(31, 119, 180, 0.2)',
            showlegend=False
        ))
        
        fig.update_layout(
            title=f"{prediction_data['category'].title()} Forecast - {prediction_data['device']}",
            xaxis_title="Date",
            yaxis_title="Value",
            hovermode='x unified',
            template='plotly_white',
            height=500
        )
        
        return fig
    
    def create_parameter_chart(self, prediction_data):
        """Create parameter-based visualization"""
        parameters = prediction_data['parameters']
        result = prediction_data['result']
        
        # Create bar chart of parameters
        fig = go.Figure()
        
        param_names = list(parameters.keys())
        param_values = list(parameters.values())
        
        fig.add_trace(go.Bar(
            x=param_names,
            y=param_values,
            name='Input Parameters',
            marker_color='lightblue'
        ))
        
        fig.update_layout(
            title=f"Parameters for {prediction_data['prediction_type'].title()} Prediction",
            xaxis_title="Parameters",
            yaxis_title="Values",
            template='plotly_white',
            height=400
        )
        
        return fig
    
    def generate_natural_response(self, query, prediction_data, query_type):
        """Generate natural language response"""
        if not self.api_provider:
            return self.simple_response(prediction_data, query_type)
        
        try:
            if query_type == "time_series":
                context = f"User asked: '{query}'. Time-series prediction shows average: {prediction_data['average']}, total: {prediction_data['total']}, range: {prediction_data['min']}-{prediction_data['max']} for {prediction_data['days_ahead']} days."
            else:
                context = f"User asked: '{query}'. Parameter-based prediction with inputs {prediction_data['parameters']} resulted in {prediction_data['result']} for {prediction_data['prediction_type']}."
            
            system_prompt = "You are a helpful industrial analytics assistant. Provide a brief, natural, and insightful response about the prediction results."
            
            if self.api_provider == "Groq":
                response = self.groq_client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": context}
                    ],
                    max_tokens=200,
                    temperature=0.3
                )
            else:  # OpenAI
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": context}
                    ],
                    max_tokens=200,
                    temperature=0.3
                )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return self.simple_response(prediction_data, query_type)
    
    def simple_response(self, prediction_data, query_type):
        """Simple fallback response"""
        if query_type == "time_series":
            return f"Based on the forecast, I predict an average of {prediction_data['average']} units per day for the next {prediction_data['days_ahead']} days, with a total of {prediction_data['total']} units."
        else:
            return f"Based on your parameters, the predicted {prediction_data['prediction_type']} is {prediction_data['result']} units."
    
    def process_query(self, query):
        """Process a single query and return response"""
        if not self.models:
            return "❌ No models available. Please load your prediction models first.", None, None
        
        # Detect query type - FIXED: changed from chatbot.detect_query_type to self.detect_query_type
        query_type = self.detect_query_type(query)
        
        if query_type == "time_series":
            # Time-series prediction
            parsed = self.parse_time_series_query(query)
            
            if parsed.get('confidence', 0) > 0.5:
                prediction_data, error = self.get_time_series_prediction(
                    parsed['category'], parsed['device'], parsed['days_ahead']
                )
                
                if prediction_data:
                    # Generate natural response
                    response = self.generate_natural_response(query, prediction_data, "time_series")
                    return response, prediction_data, "time_series"
                else:
                    return f"❌ {error}", None, None
            else:
                return "🤔 I couldn't understand your time-series request. Please be more specific about what you want to predict and the time period.", None, None
        
        elif query_type == "parameter_based":
            # Parameter-based prediction
            parsed = self.parse_parameter_query(query)
            
            if parsed.get('confidence') != 'low' and parsed.get('parameters'):
                prediction_data, error = self.get_parameter_prediction(
                    parsed['parameters'], parsed['prediction_type']
                )
                
                if prediction_data:
                    # Generate natural response
                    response = self.generate_natural_response(query, prediction_data, "parameter_based")
                    return response, prediction_data, "parameter_based"
                else:
                    return f"❌ {error}", None, None
            else:
                return "🤔 I couldn't extract clear parameters from your question.", None, None
        
        else:
            return "🤔 I'm not sure if you want a time-series forecast or a parameter-based calculation. Could you please clarify? For example: 'predict next month production'?", None, None

# Initialize chatbot
@st.cache_resource
def get_chatbot():
    return EnhancedIndustrialChatbot()

def main():
    st.title("🏭 Industrial Prediction Assistant")
    st.markdown("Ask me anything about **production forecasts**, **consumption predictions**, or **utilization calculations**")
    
    # Initialize chatbot
    chatbot = get_chatbot()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ System Status")
        
        # API Status
        if chatbot.api_provider:
            st.success(f"✅ {chatbot.api_provider} API Connected")
        else:
            st.warning("⚠️ Using Simple Parsing Mode")
            st.caption("Add GROQ_API_KEY or OPENAI_API_KEY to .env file for enhanced NLP")
        
        # Model status
        st.header("📊 Model Status")
        if chatbot.models:
            st.success("✅ Models loaded successfully")
            total_models = sum(len(devices) for devices in chatbot.models.values())
            st.metric("Total Models", total_models)
            
            for category, devices in chatbot.models.items():
                with st.expander(f"{category.title()} ({len(devices)} models)"):
                    for device in devices.keys():
                        st.write(f"• {device}")
        else:
            st.error("❌ No models found")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
        
        # Example queries
        st.header("💡 Example Queries")
        st.markdown("**Time-series examples:**")
        st.code("What will be the production next month?")
        st.code("Show me consumption forecast for next week")
        st.code("Predict utilization for tomorrow")
        
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display charts if available
            if "chart" in message and message["chart"]:
                st.plotly_chart(message["chart"], use_container_width=True)
            
            # Display metrics if available
            if "metrics" in message and message["metrics"]:
                metrics = message["metrics"]
                if "average" in metrics:  # Time-series metrics
                    cols = st.columns(4)
                    with cols[0]:
                        st.metric("Average", f"{metrics['average']:.2f}")
                    with cols[1]:
                        st.metric("Total", f"{metrics['total']:.2f}")
                    with cols[2]:
                        st.metric("Min", f"{metrics['min']:.2f}")
                    with cols[3]:
                        st.metric("Max", f"{metrics['max']:.2f}")
                else:  # Parameter-based metrics
                    st.metric("Result", f"{metrics['result']:.2f}")
    
    # Chat input
    if prompt := st.chat_input("Ask me about predictions... (e.g., 'predict next month production')"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("🤖 Processing your request..."):
                response, prediction_data, query_type = chatbot.process_query(prompt)
                
                # Display response
                st.markdown(response)
                
                # Display charts and metrics if available
                chart = None
                metrics = None
                
                if prediction_data:
                    if query_type == "time_series":
                        # Display metrics
                        metrics = {
                            "average": prediction_data['average'],
                            "total": prediction_data['total'],
                            "min": prediction_data['min'],
                            "max": prediction_data['max']
                        }
                        
                        cols = st.columns(4)
                        with cols[0]:
                            st.metric("Average", f"{prediction_data['average']:.2f}")
                        with cols[1]:
                            st.metric("Total", f"{prediction_data['total']:.2f}")
                        with cols[2]:
                            st.metric("Min", f"{prediction_data['min']:.2f}")
                        with cols[3]:
                            st.metric("Max", f"{prediction_data['max']:.2f}")
                        
                        # Display chart
                        chart = chatbot.create_time_series_chart(prediction_data)
                        st.plotly_chart(chart, use_container_width=True)
                        
                    elif query_type == "parameter_based":
                        # Display result
                        metrics = {"result": prediction_data['result']}
                        st.metric("Result", f"{prediction_data['result']:.2f}")
                        
                        # Display chart
                        chart = chatbot.create_parameter_chart(prediction_data)
                        st.plotly_chart(chart, use_container_width=True)
        
        # Add assistant response to chat history
        assistant_message = {
            "role": "assistant", 
            "content": response
        }
        if chart:
            assistant_message["chart"] = chart
        if metrics:
            assistant_message["metrics"] = metrics
            
        st.session_state.messages.append(assistant_message)

if __name__ == "__main__":
    main()
    
    
    
