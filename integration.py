import pandas as pd
import numpy as np
import joblib
import logging
from datetime import datetime, timedelta
import re
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class PredictionModule:
    def __init__(self, model_path_prefix='models/production_predictor'):
        """
        Initialize the Prediction Module for chatbot integration
        
        Args:
            model_path_prefix (str): Path prefix for model files
        """
        self.model_path_prefix = model_path_prefix
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = {}
        self.logger = self._setup_logging()
        self.load_models()
    
    def _setup_logging(self):
        """Setup logging"""
        return logging.getLogger(__name__)
    
    def load_models(self):
        """Load pre-trained models"""
        try:
            self.models = joblib.load(f'{self.model_path_prefix}_models.pkl')
            self.scalers = joblib.load(f'{self.model_path_prefix}_scalers.pkl')
            self.encoders = joblib.load(f'{self.model_path_prefix}_encoders.pkl')
            self.feature_columns = joblib.load(f'{self.model_path_prefix}_features.pkl')
            
            self.logger.info("Prediction models loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load models: {str(e)}")
            return False
    
    def detect_prediction_query(self, user_query):
        """
        Detect if user is asking for predictions
        
        Args:
            user_query (str): User's query
            
        Returns:
            dict: Prediction query details or None
        """
        user_query_lower = user_query.lower()
        
        # Prediction keywords
        prediction_keywords = [
            'predict', 'forecast', 'future', 'next week', 'next month', 'next day',
            'tomorrow', 'upcoming', 'expected', 'will be', 'estimate', 'projection'
        ]
        
        # Time period keywords
        time_keywords = {
            'day': ['day', 'daily', 'tomorrow'],
            'week': ['week', 'weekly', 'next week'],
            'month': ['month', 'monthly', 'next month'],
            'days': r'(\d+)\s*days?',
            'weeks': r'(\d+)\s*weeks?',
            'months': r'(\d+)\s*months?'
        }
        
        # Metric keywords
        metric_keywords = {
            'production': ['production', 'output', 'produce', 'manufactured'],
            'consumption': ['consumption', 'consume', 'usage', 'used'],
            'utilization': ['utilization', 'efficiency', 'performance']
        }
        
        # Check if it's a prediction query
        is_prediction = any(keyword in user_query_lower for keyword in prediction_keywords)
        
        if not is_prediction:
            return None
        
        # Extract time period
        time_period = 'week'  # default
        time_value = 7  # default days
        
        for period, keywords in time_keywords.items():
            if period in ['days', 'weeks', 'months']:
                match = re.search(keywords, user_query_lower)
                if match:
                    number = int(match.group(1))
                    if period == 'days':
                        time_period = 'days'
                        time_value = number
                    elif period == 'weeks':
                        time_period = 'weeks'
                        time_value = number * 7
                    elif period == 'months':
                        time_period = 'months'
                        time_value = number * 30
                    break
            else:
                if any(keyword in user_query_lower for keyword in keywords):
                    if period == 'day':
                        time_period = 'day'
                        time_value = 1
                    elif period == 'week':
                        time_period = 'week'
                        time_value = 7
                    elif period == 'month':
                        time_period = 'month'
                        time_value = 30
                    break
        
        # Extract metric
        detected_metric = None
        for metric, keywords in metric_keywords.items():
            if any(keyword in user_query_lower for keyword in keywords):
                detected_metric = metric
                break
        
        if not detected_metric:
            detected_metric = 'production'  # default
        
        # Extract machine information
        machine_name = None
        machine_keywords = ['machine', 'device', 'equipment']
        
        # Look for specific machine names or "all machines"
        if 'all machines' in user_query_lower or 'all three machines' in user_query_lower:
            machine_name = 'all'
        else:
            # Try to extract specific machine name
            words = user_query.split()
            for i, word in enumerate(words):
                if any(mk in word.lower() for mk in machine_keywords):
                    if i + 1 < len(words):
                        potential_name = words[i + 1]
                        if not potential_name.lower() in ['will', 'be', 'production', 'consumption']:
                            machine_name = potential_name
                        break
        
        return {
            'is_prediction': True,
            'metric': detected_metric,
            'time_period': time_period,
            'time_value': time_value,
            'machine_name': machine_name,
            'original_query': user_query
        }
    
    def get_available_machines(self, db):
        """
        Get list of available machines from database
        
        Args:
            db: Database connection
            
        Returns:
            list: List of machine names
        """
        try:
            # Query to get unique machine names
            query = "SELECT DISTINCT device_name FROM hourly_production ORDER BY device_name"
            result = db.run(query)
            
            # Parse the result
            machines = []
            if result and result != "[]":
                # Extract machine names from result
                import ast
                try:
                    parsed_result = ast.literal_eval(result)
                    machines = [row[0] for row in parsed_result if row[0]]
                except:
                    # Fallback parsing
                    lines = result.strip('[]').split('\n')
                    for line in lines:
                        if line.strip():
                            machine = line.strip("(), '\"")
                            if machine:
                                machines.append(machine)
            
            return machines if machines else ['Machine1', 'Machine2', 'Machine3']  # Default
            
        except Exception as e:
            self.logger.error(f"Failed to get machine names: {str(e)}")
            return ['Machine1', 'Machine2', 'Machine3']  # Default fallback
    
    def make_prediction(self, prediction_details, db):
        """
        Make predictions based on user query
        
        Args:
            prediction_details (dict): Details from detect_prediction_query
            db: Database connection
            
        Returns:
            dict: Prediction results
        """
        try:
            metric = prediction_details['metric']
            time_value = prediction_details['time_value']
            machine_name = prediction_details['machine_name']
            
            # Check if model exists
            if metric not in self.models:
                return {
                    'success': False,
                    'error': f'Prediction model for {metric} is not available. Available models: {list(self.models.keys())}'
                }
            
            # Get available machines
            available_machines = self.get_available_machines(db)
            
            # Determine which machines to predict for
            if machine_name == 'all' or machine_name is None:
                machines_to_predict = available_machines[:3]  # First 3 machines
            elif machine_name in available_machines:
                machines_to_predict = [machine_name]
            else:
                # Find closest match
                machine_name_lower = machine_name.lower()
                closest_match = None
                for machine in available_machines:
                    if machine_name_lower in machine.lower() or machine.lower() in machine_name_lower:
                        closest_match = machine
                        break
                
                machines_to_predict = [closest_match] if closest_match else available_machines[:1]
            
            # Make predictions for each machine
            all_predictions = []
            
            for machine in machines_to_predict:
                predictions = self._predict_for_machine(metric, machine, time_value)
                all_predictions.extend(predictions)
            
            return {
                'success': True,
                'predictions': all_predictions,
                'metric': metric,
                'time_period': prediction_details['time_period'],
                'machines': machines_to_predict,
                'query': prediction_details['original_query']
            }
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            return {
                'success': False,
                'error': f'Prediction failed: {str(e)}'
            }
    
    def _predict_for_machine(self, metric, machine_name, prediction_days):
        """
        Make predictions for a specific machine
        
        Args:
            metric (str): Metric to predict
            machine_name (str): Machine name
            prediction_days (int): Number of days to predict
            
        Returns:
            list: List of prediction dictionaries
        """
        model_info = self.models[metric]
        model = model_info['model']
        model_type = model_info['model_type']
        
        current_date = datetime.now()
        predictions = []
        
        # Generate hourly predictions for better granularity
        hours_to_predict = min(prediction_days * 24, 168)  # Max 1 week of hourly data
        
        for hour_offset in range(0, hours_to_predict, 24):  # Daily predictions
            future_date = current_date + timedelta(hours=hour_offset)
            
            # Create features
            features = self._create_prediction_features(future_date, machine_name, metric)
            
            if features is not None:
                # Make prediction
                if model_type == 'linear_regression':
                    feature_vector = self.scalers[metric].transform(features)
                else:
                    feature_vector = features
                
                prediction_value = model.predict(feature_vector)[0]
                prediction_value = max(0, prediction_value)  # Ensure non-negative
                
                predictions.append({
                    'date': future_date,
                    'machine': machine_name,
                    'predicted_value': prediction_value,
                    'metric': metric,
                    'confidence': model_info.get('r2_score', 0.8)
                })
        
        return predictions
    
    def _create_prediction_features(self, future_date, machine_name, metric):
        """
        Create feature vector for prediction
        
        Args:
            future_date (datetime): Date to predict for
            machine_name (str): Machine name
            metric (str): Metric being predicted
            
        Returns:
            np.array: Feature vector
        """
        try:
            # Base time features
            features = {
                'hour': future_date.hour,
                'day_of_week': future_date.weekday(),
                'day_of_month': future_date.day,
                'month': future_date.month,
                'is_weekend': 1 if future_date.weekday() >= 5 else 0,
            }
            
            # Add machine encoding
            if 'device_name' in self.encoders:
                try:
                    features['device_name'] = self.encoders['device_name'].transform([machine_name])[0]
                except:
                    features['device_name'] = 0  # Default if machine not in training data
            
            # Add shift encoding
            hour = future_date.hour
            if 0 <= hour < 8:
                shift = 'morning'
            elif 8 <= hour < 16:
                shift = 'afternoon'
            else:
                shift = 'night'
            
            if 'shift' in self.encoders:
                try:
                    features['shift'] = self.encoders['shift'].transform([shift])[0]
                except:
                    features['shift'] = 0
            
            # Create feature vector matching training features
            feature_vector = []
            required_features = self.feature_columns[metric]
            
            for feature_name in required_features:
                if feature_name in features:
                    feature_vector.append(features[feature_name])
                else:
                    # Handle lag and rolling features with default values
                    if 'lag' in feature_name or 'rolling' in feature_name:
                        # Use reasonable defaults based on metric
                        if metric == 'production':
                            feature_vector.append(100.0)  # Default production value
                        elif metric == 'consumption':
                            feature_vector.append(50.0)   # Default consumption value
                        elif metric == 'utilization':
                            feature_vector.append(0.8)    # Default efficiency
                        else:
                            feature_vector.append(0.0)
                    else:
                        feature_vector.append(0.0)
            
            return np.array(feature_vector).reshape(1, -1)
            
        except Exception as e:
            self.logger.error(f"Error creating features: {str(e)}")
            return None
    
    def format_prediction_response(self, prediction_result):
        """
        Format prediction results for chatbot response
        
        Args:
            prediction_result (dict): Results from make_prediction
            
        Returns:
            str: Formatted response text
        """
        if not prediction_result['success']:
            return f"❌ {prediction_result['error']}"
        
        predictions = prediction_result['predictions']
        metric = prediction_result['metric']
        machines = prediction_result['machines']
        time_period = prediction_result['time_period']
        
        # Group predictions by machine
        machine_predictions = {}
        for pred in predictions:
            machine = pred['machine']
            if machine not in machine_predictions:
                machine_predictions[machine] = []
            machine_predictions[machine].append(pred)
        
        response_parts = []
        response_parts.append(f"🔮 **{metric.title()} Predictions** for the next {time_period}")
        response_parts.append("")
        
        for machine, preds in machine_predictions.items():
            response_parts.append(f"**{machine}:**")
            
            # Calculate summary statistics
            values = [p['predicted_value'] for p in preds]
            avg_value = np.mean(values)
            total_value = np.sum(values)
            confidence = preds[0]['confidence'] if preds else 0.8
            
            # Format based on metric type
            if metric == 'production':
                response_parts.append(f"  • Average daily production: {avg_value:.1f} units")
                response_parts.append(f"  • Total predicted production: {total_value:.1f} units")
            elif metric == 'consumption':
                response_parts.append(f"  • Average daily consumption: {avg_value:.1f} units")
                response_parts.append(f"  • Total predicted consumption: {total_value:.1f} units")
            elif metric == 'utilization':
                response_parts.append(f"  • Average utilization: {avg_value:.1%}")
                response_parts.append(f"  • Peak utilization: {max(values):.1%}")
            
            response_parts.append(f"  • Confidence: {confidence:.1%}")
            response_parts.append("")
        
        # Add trend analysis
        if len(predictions) > 1:
            first_day = np.mean([p['predicted_value'] for p in predictions[:len(predictions)//3]])
            last_day = np.mean([p['predicted_value'] for p in predictions[-len(predictions)//3:]])
            
            if last_day > first_day * 1.05:
                trend = "📈 Increasing trend"
            elif last_day < first_day * 0.95:
                trend = "📉 Decreasing trend"
            else:
                trend = "➡️ Stable trend"
            
            response_parts.append(f"**Trend Analysis:** {trend}")
        
        return "\n".join(response_parts)
    
    def create_prediction_chart(self, prediction_result, chart_type='line'):
        """
        Create a visualization chart for predictions
        
        Args:
            prediction_result (dict): Results from make_prediction
            chart_type (str): Type of chart ('line', 'bar', 'combined')
            
        Returns:
            plotly.graph_objects.Figure: Chart figure
        """
        if not prediction_result['success']:
            return None
        
        predictions = prediction_result['predictions']
        metric = prediction_result['metric']
        
        # Convert to DataFrame for easier plotting
        df = pd.DataFrame(predictions)
        df['date'] = pd.to_datetime(df['date'])
        df['date_str'] = df['date'].dt.strftime('%Y-%m-%d %H:%M')
        
        if chart_type == 'line':
            fig = px.line(
                df, 
                x='date', 
                y='predicted_value', 
                color='machine',
                title=f'Predicted {metric.title()} Over Time',
                labels={
                    'predicted_value': f'{metric.title()} Value',
                    'date': 'Date',
                    'machine': 'Machine'
                }
            )
        elif chart_type == 'bar':
            # Aggregate by day for bar chart
            daily_df = df.groupby(['machine', df['date'].dt.date])['predicted_value'].sum().reset_index()
            daily_df['date'] = pd.to_datetime(daily_df['date'])
            
            fig = px.bar(
                daily_df,
                x='date',
                y='predicted_value',
                color='machine',
                title=f'Daily Predicted {metric.title()}',
                labels={
                    'predicted_value': f'Daily {metric.title()}',
                    'date': 'Date',
                    'machine': 'Machine'
                }
            )
        else:  # combined
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=[f'Predicted {metric.title()} Timeline', 'Machine Comparison'],
                specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
            )
            
            # Line plot
            for machine in df['machine'].unique():
                machine_data = df[df['machine'] == machine]
                fig.add_trace(
                    go.Scatter(
                        x=machine_data['date'],
                        y=machine_data['predicted_value'],
                        mode='lines+markers',
                        name=machine,
                        line=dict(width=2)
                    ),
                    row=1, col=1
                )
            
            # Bar plot for totals
            machine_totals = df.groupby('machine')['predicted_value'].sum().reset_index()
            fig.add_trace(
                go.Bar(
                    x=machine_totals['machine'],
                    y=machine_totals['predicted_value'],
                    name='Total Predicted',
                    showlegend=False
                ),
                row=2, col=1
            )
            
            fig.update_layout(
                title=f'Prediction Analysis: {metric.title()}',
                height=600
            )
        
        # Update layout
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title=f'{metric.title()} Value',
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    def get_prediction_summary(self, prediction_result):
        """
        Get a summary of prediction results
        
        Args:
            prediction_result (dict): Results from make_prediction
            
        Returns:
            dict: Summary statistics
        """
        if not prediction_result['success']:
            return {'error': prediction_result['error']}
        
        predictions = prediction_result['predictions']
        metric = prediction_result['metric']
        
        df = pd.DataFrame(predictions)
        
        summary = {
            'metric': metric,
            'total_predictions': len(predictions),
            'machines': list(df['machine'].unique()),
            'date_range': {
                'start': df['date'].min().strftime('%Y-%m-%d %H:%M'),
                'end': df['date'].max().strftime('%Y-%m-%d %H:%M')
            },
            'statistics': {
                'mean': float(df['predicted_value'].mean()),
                'median': float(df['predicted_value'].median()),
                'std': float(df['predicted_value'].std()),
                'min': float(df['predicted_value'].min()),
                'max': float(df['predicted_value'].max()),
                'total': float(df['predicted_value'].sum())
            },
            'confidence': {
                'average': float(df['confidence'].mean()),
                'min': float(df['confidence'].min()),
                'max': float(df['confidence'].max())
            }
        }
        
        # Machine-wise breakdown
        machine_summary = {}
        for machine in df['machine'].unique():
            machine_data = df[df['machine'] == machine]
            machine_summary[machine] = {
                'predictions': len(machine_data),
                'total_predicted': float(machine_data['predicted_value'].sum()),
                'average_predicted': float(machine_data['predicted_value'].mean()),
                'confidence': float(machine_data['confidence'].mean())
            }
        
        summary['machine_breakdown'] = machine_summary
        
        return summary
    
    def validate_prediction_request(self, prediction_details):
        """
        Validate prediction request parameters
        
        Args:
            prediction_details (dict): Prediction request details
            
        Returns:
            dict: Validation result
        """
        errors = []
        warnings = []
        
        # Check metric
        if prediction_details['metric'] not in self.models:
            errors.append(f"Model for metric '{prediction_details['metric']}' not available")
        
        # Check time range
        if prediction_details['time_value'] > 30:
            warnings.append("Predictions beyond 30 days may be less accurate")
        elif prediction_details['time_value'] < 1:
            errors.append("Time period must be at least 1 day")
        
        # Check if models are loaded
        if not self.models:
            errors.append("No prediction models are loaded")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }