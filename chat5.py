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
        
        # Define comprehensive keyword patterns
        self.greeting_patterns = [
            r'\b(hi|hello|hey|good morning|good afternoon|good evening|greetings)\b',
            r'\b(how are you|what\'s up|whats up)\b',
            r'\b(thanks|thank you|bye|goodbye)\b'
        ]
        
        self.time_series_patterns = [
            r'\b(predict|forecast|future|next|tomorrow|today|yesterday)\b',
            r'\b(week|month|day|year|period|time)\b',
            r'\b(production|consumption|utilization|efficiency|output|usage)\b',
            r'\b(trend|pattern|outlook|projection)\b',
            r'\d+\s*(day|week|month|year)s?',
            r'(next|coming|following)\s+\d*\s*(day|week|month|year)s?'
        ]
        
        self.parameter_patterns = [
            r'\b(if|given|with|when|assuming)\b',
            r'\b(capacity|demand|supply|inventory|workforce|equipment)\b',
            r'\d+(\.\d+)?',  # Numbers
            r'\b(what if|scenario|calculate|compute)\b',
            r'\b(rate|level|amount|quantity)\b'
        ]
        
        # Categories mapping
        self.category_mapping = {
            'production': ['production', 'produce', 'manufacturing', 'output', 'make', 'create', 'generate'],
            'consumption': ['consumption', 'consume', 'usage', 'use', 'intake', 'demand', 'requirement'],
            'utilization': ['utilization', 'efficiency', 'performance', 'capacity', 'util', 'usage rate']
        }
    
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
    
    def is_greeting(self, query):
        """Check if query is a greeting or casual conversation"""
        query_lower = query.lower().strip()
        
        for pattern in self.greeting_patterns:
            if re.search(pattern, query_lower):
                return True
        
        # Check for very short queries that might be greetings
        if len(query_lower.split()) <= 2 and any(word in query_lower for word in ['hi', 'hello', 'hey', 'thanks', 'bye']):
            return True
            
        return False
    
    def detect_query_type(self, query):
        """Enhanced query type detection"""
        query_lower = query.lower().strip()
        
        # Check for greetings first
        if self.is_greeting(query):
            return "greeting"
        
        # Count pattern matches
        time_score = 0
        param_score = 0
        
        # Time-series pattern matching
        for pattern in self.time_series_patterns:
            matches = len(re.findall(pattern, query_lower))
            time_score += matches
        
        # Parameter pattern matching
        for pattern in self.parameter_patterns:
            matches = len(re.findall(pattern, query_lower))
            param_score += matches
        
        # Additional scoring based on context
        if any(word in query_lower for word in ['predict', 'forecast', 'future', 'next', 'tomorrow']):
            time_score += 3
        
        if any(word in query_lower for word in ['if', 'given', 'with', 'when', 'scenario']):
            param_score += 3
        
        # Check for specific time references
        if re.search(r'\d+\s*(day|week|month|year)s?', query_lower):
            time_score += 2
        
        if re.search(r'(next|coming|following)\s+\d*\s*(day|week|month|year)s?', query_lower):
            time_score += 2
        
        # Determine query type
        if time_score > param_score and time_score > 0:
            return "time_series"
        elif param_score > time_score and param_score > 0:
            return "parameter_based"
        elif time_score > 0 or param_score > 0:
            return "time_series"  # Default to time series if any prediction keywords
        else:
            return "unclear"
    
    def handle_greeting(self, query):
        """Handle greeting and casual conversation"""
        query_lower = query.lower().strip()
        
        greetings = {
            'hi': "Hello! 👋 I'm your Industrial Prediction Assistant. I can help you with production forecasts, consumption predictions, and utilization calculations. What would you like to know?",
            'hello': "Hi there! 👋 I'm here to help with your industrial predictions. You can ask me about production forecasts, consumption trends, or utilization rates. What can I predict for you today?",
            'hey': "Hey! 👋 Ready to dive into some predictions? I can forecast production, consumption, and utilization data. What would you like to explore?",
            'good morning': "Good morning! 🌅 Hope you're having a great day! I'm here to help with your industrial predictions. What would you like to forecast today?",
            'good afternoon': "Good afternoon! ☀️ I'm your prediction assistant, ready to help with production, consumption, and utilization forecasts. What can I analyze for you?",
            'good evening': "Good evening! 🌆 I'm here to help with your industrial predictions. Whether it's production forecasts or consumption analysis, I'm ready to assist!",
            'how are you': "I'm doing great and ready to help! 🤖 I specialize in industrial predictions. Ask me about production forecasts, consumption trends, or utilization calculations!",
            'thanks': "You're welcome! 😊 Feel free to ask me anything about predictions - production, consumption, or utilization forecasts. I'm here to help!",
            'thank you': "My pleasure! 😊 If you need any more predictions or forecasts, just ask. I'm always ready to help with your industrial data analysis!",
            'bye': "Goodbye! 👋 Thanks for using the Industrial Prediction Assistant. Come back anytime you need forecasts or predictions!",
            'goodbye': "See you later! 👋 Don't hesitate to return when you need more industrial predictions and forecasts!"
        }
        
        # Find the best matching greeting
        for key, response in greetings.items():
            if key in query_lower:
                return response
        
        # Default friendly response
        return "Hello! 👋 I'm your Industrial Prediction Assistant. I can help you forecast production, predict consumption, and analyze utilization rates. What would you like to know? \n\n💡 Try asking things like:\n• 'Predict next week production'\n• 'What will be consumption tomorrow?'\n• 'Show me utilization forecast for next month'"
    
    def extract_category(self, query):
        """Extract category from query"""
        query_lower = query.lower()
        
        for category, keywords in self.category_mapping.items():
            if any(keyword in query_lower for keyword in keywords):
                return category
        
        # Default fallback
        return list(self.models.keys())[0] if self.models else 'production'
    
    def extract_time_period(self, query):
        """Extract time period from query"""
        query_lower = query.lower()
        
        # Direct time references
        if 'tomorrow' in query_lower:
            return 1
        elif 'today' in query_lower:
            return 1
        elif 'next week' in query_lower or 'this week' in query_lower:
            return 7
        elif 'next month' in query_lower or 'this month' in query_lower:
            return 30
        elif 'next year' in query_lower or 'this year' in query_lower:
            return 365
        
        # Extract specific numbers
        number_patterns = [
            r'(\d+)\s*days?',
            r'(\d+)\s*weeks?',
            r'(\d+)\s*months?',
            r'(\d+)\s*years?'
        ]
        
        for i, pattern in enumerate(number_patterns):
            match = re.search(pattern, query_lower)
            if match:
                num = int(match.group(1))
                multipliers = [1, 7, 30, 365]  # days, weeks, months, years
                return num * multipliers[i]
        
        # Default to 7 days
        return 7
    
    def parse_time_series_query(self, query):
        """Enhanced time-series query parsing"""
        category = self.extract_category(query)
        days_ahead = self.extract_time_period(query)
        
        # Extract device if mentioned
        device = None
        query_lower = query.lower()
        
        # Look for device names in the query
        if self.models and category in self.models:
            for device_name in self.models[category].keys():
                if device_name.lower() in query_lower:
                    device = device_name
                    break
        
        return {
            'category': category,
            'device': device,
            'days_ahead': days_ahead,
            'confidence': 0.9  # High confidence for enhanced parsing
        }
    
    def parse_parameter_query(self, query):
        """Enhanced parameter query parsing"""
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
            return self.simple_param_parse(query)
    
    def simple_param_parse(self, query):
        """Enhanced fallback parameter parsing"""
        query_lower = query.lower()
        
        # Extract prediction type
        prediction_type = self.extract_category(query)
        
        # Extract numbers
        numbers = re.findall(r'\d+(?:\.\d+)?', query_lower)
        parameters = {}
        
        # Try to match numbers with context
        words = query_lower.split()
        for i, word in enumerate(words):
            if word.replace('.', '').isdigit():
                num = float(word)
                # Look for context words around the number
                context_start = max(0, i-2)
                context_end = min(len(words), i+3)
                context = ' '.join(words[context_start:context_end])
                
                if any(term in context for term in ['capacity', 'cap']):
                    parameters['capacity'] = num
                elif any(term in context for term in ['demand', 'need']):
                    parameters['demand'] = num
                elif any(term in context for term in ['supply', 'available']):
                    parameters['supply'] = num
                elif any(term in context for term in ['inventory', 'stock']):
                    parameters['inventory'] = num
                elif any(term in context for term in ['workforce', 'workers', 'employees']):
                    parameters['workforce'] = num
                elif any(term in context for term in ['equipment', 'machines']):
                    parameters['equipment'] = num
                else:
                    parameters[f'value_{i}'] = num
        
        # If no specific parameters found, use generic approach
        if not parameters and numbers:
            if len(numbers) >= 2:
                parameters['capacity'] = float(numbers[0])
                parameters['demand'] = float(numbers[1])
                confidence = "high"
            elif len(numbers) == 1:
                parameters['value'] = float(numbers[0])
                confidence = "medium"
            else:
                confidence = "low"
        else:
            confidence = "high" if parameters else "low"
        
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
        if query_type == "time_series":
            category = prediction_data['category']
            device = prediction_data['device']
            days_ahead = prediction_data['days_ahead']
            average = prediction_data['average']
            total = prediction_data['total']
            
            time_phrase = "tomorrow" if days_ahead == 1 else f"the next {days_ahead} days"
            
            return f"📊 Based on the forecast for **{category}** from **{device}**, I predict an average of **{average:.2f} units** per day for {time_phrase}. The total expected {category} is **{total:.2f} units**."
        
        else:  # parameter_based
            prediction_type = prediction_data['prediction_type']
            result = prediction_data['result']
            parameters = prediction_data['parameters']
            
            param_text = ", ".join([f"{k}: {v}" for k, v in parameters.items()])
            
            return f"🔢 Based on your parameters ({param_text}), the predicted **{prediction_type}** is **{result:.2f} units**."
    
    def process_query(self, query):
        """Process a single query and return response"""
        # Handle empty or very short queries
        if not query or len(query.strip()) < 2:
            return "I didn't receive a clear question. Could you please ask me something about predictions? 🤔", None, None
        
        # Detect query type
        query_type = self.detect_query_type(query)
        
        # Handle greetings
        if query_type == "greeting":
            return self.handle_greeting(query), None, None
        
        # Check if models are available
        if not self.models:
            return "❌ No models available. Please load your prediction models first.", None, None
        
        if query_type == "time_series":
            # Time-series prediction
            parsed = self.parse_time_series_query(query)
            
            prediction_data, error = self.get_time_series_prediction(
                parsed['category'], parsed['device'], parsed['days_ahead']
            )
            
            if prediction_data:
                # Generate natural response
                response = self.generate_natural_response(query, prediction_data, "time_series")
                return response, prediction_data, "time_series"
            else:
                return f"❌ {error}", None, None
        
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
                return "🤔 I couldn't extract clear parameters from your question. Please provide specific numbers for your calculation.", None, None
        
        else:
            # Try to be helpful with unclear queries
            return """🤔 I'd love to help you with predictions! Could you be more specific? Here are some examples:

**Time-series predictions:**
• "Predict next week production"
• "What will be consumption tomorrow?"
• "Show me utilization forecast for next month"

**Parameter-based calculations:**
• "What will be the utilization if capacity is 1000 and demand is 800?"
• "Calculate production with 500 workforce and 10 equipment"

What would you like to predict? 📊""", None, None

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
            st.warning("⚠️ Using Enhanced Parsing Mode")
            st.caption("Add GROQ_API_KEY or OPENAI_API_KEY to .env file for even better NLP")
        
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
        
        st.markdown("**Greetings:**")
        st.code("Hi there!")
        st.code("Hello, how are you?")
        
        st.markdown("**Time-series examples:**")
        st.code("Predict next week production")
        st.code("What will be consumption tomorrow?")
        st.code("Show me utilization for next month")
        
        st.markdown("**Parameter-based examples:**")
        st.code("What if capacity is 1000 and demand is 800?")
        st.code("Calculate production with 500 workforce")
        
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
    if prompt := st.chat_input("Ask me about predictions... (e.g., 'Hi!' or 'predict next month production')"):
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