from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
import logging
import sys
from datetime import datetime
import numpy as np
from langchain.chat_models import ChatOpenAI

# Configure logging
def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('chatbot.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def init_database(user: str, password: str, host: str, port: str, database: str) -> SQLDatabase:
    """Initialize database connection with logging"""
    try:
        db_uri = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
        logger.info(f"Attempting to connect to database: {host}:{port}/{database}")
        db = SQLDatabase.from_uri(db_uri)
        logger.info("Database connection successful")
        return db
    except Exception as e:
        logger.error(f"Database connection failed: {str(e)}")
        raise e
    

def detect_metric_and_visualization(user_query: str):
    """Enhanced detection for different metrics and visualization requirements"""
    user_query_lower = user_query.lower()
    logger.info(f"Analyzing query for metric and visualization: {user_query}")
    
    # Detect metric type
    metric_keywords = {
        'production': ['production', 'produce', 'output', 'manufactured', 'created', 'generated'],
        'consumption': ['consumption', 'consume', 'used', 'utilized', 'consumed', 'usage', 'intake'],
        'efficiency': ['efficiency', 'efficient', 'performance', 'productivity', 'yield'],
        'downtime': ['downtime', 'offline', 'stopped', 'breakdown', 'maintenance'],
        'energy': ['energy', 'power', 'electricity', 'kw', 'kwh', 'watts'],
        'temperature': ['temperature', 'temp', 'heat', 'cooling', 'celsius', 'fahrenheit'],
        'speed': ['speed', 'rpm', 'rate', 'velocity', 'pace'],
        'pulse': ['pulse', 'pulse per minute', 'pulse rate', 'length difference', 'increment']
    }
    
    detected_metric = 'production'  # default
    for metric, keywords in metric_keywords.items():
        if any(keyword in user_query_lower for keyword in keywords):
            detected_metric = metric
            break
    
    # Keywords that indicate visualization request
    viz_keywords = [
        'plot', 'chart', 'graph', 'visualize', 'show', 'display',
        'bar chart', 'line chart', 'pie chart', 'histogram', 'scatter plot',
        'bar', 'line', 'pie', 'trend', 'comparison', 'grouped bar', 'stacked bar'
    ]
    
    needs_viz = any(keyword in user_query_lower for keyword in viz_keywords)
    
    # Enhanced chart type detection
    #chart_type = "bar"  # default
    # Enhanced chart type detection - PULSE SHOULD DEFAULT TO LINE CHART
    chart_type = "line" if detected_metric == 'pulse' else "bar"  # default
    
    # Check for multi-machine/multi-category requests
    multi_machine_keywords = ['all machines', 'each machine', 'by machine', 'machines', 'three machines', 'every machine']
    multi_category_keywords = ['by day', 'each day', 'daily', 'monthly', 'by month', 'each month', 'hourly', 'by hour']
    
    is_multi_machine = any(keyword in user_query_lower for keyword in multi_machine_keywords)
    is_multi_category = any(keyword in user_query_lower for keyword in multi_category_keywords)
    
    if is_multi_machine and (is_multi_category or 'bar' in user_query_lower):
        chart_type = "multi_machine_bar"
    elif any(word in user_query_lower for word in ['line', 'trend', 'over time', 'hourly', 'daily', 'time series']):
        chart_type = "line"
    elif any(word in user_query_lower for word in ['pie', 'proportion', 'percentage', 'share', 'distribution']):
        chart_type = "pie"
    elif any(word in user_query_lower for word in ['scatter', 'relationship', 'correlation']):
        chart_type = "scatter"
    elif any(word in user_query_lower for word in ['histogram', 'distribution', 'frequency']):
        chart_type = "histogram"
    elif any(word in user_query_lower for word in ['grouped', 'stacked', 'multiple']):
        chart_type = "grouped_bar"
    
    logger.info(f"Detected metric: {detected_metric}, Visualization needed: {needs_viz}, Chart type: {chart_type}, Multi-machine: {is_multi_machine}")
    return detected_metric, needs_viz, chart_type




def calculate_pulse_per_minute(df, length_col='length', timestamp_col='timestamp', device_col='device_name'):
    """
    Calculate pulse per minute from length data
    Pulse = present length - previous length for each minute
    """
    try:
        logger.info(f"Calculating pulse per minute for data with shape: {df.shape}")
        
        # Ensure timestamp is datetime
        if df[timestamp_col].dtype != 'datetime64[ns]':
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # Sort by device and timestamp
        df = df.sort_values([device_col, timestamp_col])
        
        # Calculate pulse per minute for each device
        pulse_data = []
        
        for device in df[device_col].unique():
            device_data = df[df[device_col] == device].copy()
            
            # Calculate pulse (difference between consecutive readings)
            device_data['pulse_per_minute'] = device_data[length_col].diff()
            
            # Remove the first row (NaN) and negative values (if any)
            device_data = device_data.dropna()
            device_data = device_data[device_data['pulse_per_minute'] >= 0]
            
            pulse_data.append(device_data)
        
        # Combine all device data
        result_df = pd.concat(pulse_data, ignore_index=True) if pulse_data else pd.DataFrame()
        
        logger.info(f"Pulse calculation completed. Result shape: {result_df.shape}")
        return result_df
        
    except Exception as e:
        logger.error(f"Error calculating pulse per minute: {str(e)}")
        return pd.DataFrame()







def get_enhanced_sql_chain(db):
    """Enhanced SQL chain with dynamic metric detection and flexible column mapping"""
    template = """
    You are an expert data analyst. Based on the table schema below, write a SQL query that answers the user's question.
    
    <SCHEMA>{schema}</SCHEMA>
    
    Conversation History: {chat_history}
    
    IMPORTANT GUIDELINES:
    
    1. METRIC DETECTION - Identify what the user is asking for:
       - PRODUCTION: production_output, output, produced_quantity, manufactured_units
       - CONSUMPTION: consumption_amount, consumed_quantity, material_used, energy_consumed, power_consumption
       - EFFICIENCY: efficiency_rate, performance_ratio, yield_percentage
       - DOWNTIME: downtime_minutes, offline_duration, maintenance_time
       - ENERGY: energy_usage, power_consumption, electricity_used
       - TEMPERATURE: temperature_value, temp_reading, heat_level
       - SPEED: machine_speed, rpm_value, operation_rate
    
    2. FLEXIBLE COLUMN MAPPING - Adapt to available columns:
       - For machine identification: device_name, machine_name, machine_id, equipment_name
       - For time columns: actual_start_time, timestamp, date_time, created_at, time_stamp
       - For metric values: Look for columns that match the requested metric type
    
    3. PULSE-SPECIFIC QUERIES:
       - For pulse per minute: Select device_name, timestamp, length columns
       - Order by device_name, timestamp for proper calculation
       - Include recent data for trend analysis
       - Example: SELECT device_name, timestamp, length FROM table_name WHERE timestamp >= DATE_SUB(NOW(), INTERVAL 1 HOUR) ORDER BY device_name, timestamp
       - Calculate pulse per minute using length differences
    
    4. TIME PERIOD HANDLING:
       - April 2024: WHERE YEAR(time_column) = 2024 AND MONTH(time_column) = 4
       - Daily grouping: GROUP BY DATE(time_column)
       - Monthly grouping: GROUP BY DATE_FORMAT(time_column, '%Y-%m')
       - Hourly grouping: GROUP BY DATE_FORMAT(time_column, '%Y-%m-%d %H:00:00')
       
    5. PULSE QUERY EXAMPLES:
       - "Show pulse per minute for last hour": SELECT device_name, timestamp, length FROM length_data WHERE timestamp >= DATE_SUB(NOW(), INTERVAL 1 HOUR) ORDER BY device_name, timestamp
       - "Pulse trends for today": SELECT device_name, timestamp, length FROM length_data WHERE DATE(timestamp) = CURDATE() ORDER BY device_name, timestamp
       - "Compare pulse rates by machine": SELECT device_name, timestamp, length FROM length_data WHERE timestamp >= DATE_SUB(NOW(), INTERVAL 6 HOUR) ORDER BY device_name, timestamp
    
    
    6. MULTI-MACHINE QUERIES:
       - Always include machine identifier as separate column
       - Use appropriate aggregation: SUM(), AVG(), MAX(), MIN() based on metric
       - GROUP BY time period AND machine identifier
       - ORDER BY time period, then machine name
    
    7. ADAPTIVE QUERY STRUCTURE:
       - Consumption queries: Focus on consumption-related columns
       - Production queries: Focus on production-related columns
       - Comparison queries: Include multiple machines and time grouping
    
    Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.
    
    For visualization requests, ensure the query returns data suitable for charting:
    - For line charts: include time-based columns and numeric values
    - For pie charts: include category names and corresponding values
    - For bar charts: include categories and numeric values
    - Limit results to reasonable amounts (e.g., LIMIT 50 for large datasets)
    
    Examples:
    Question: Which machine produced the most output in the last hour?
    SQL Query: SELECT device_name, SUM(production_output) AS total_output FROM hourly_production WHERE actual_start_time >= NOW() - INTERVAL 1 HOUR GROUP BY device_name ORDER BY total_output DESC LIMIT 1
    
    Question: Plot the hourly production with line graph
    SQL Query: SELECT DATE_FORMAT(actual_start_time, '%Y-%m-%d %H:00:00') AS hour, SUM(production_output) AS total_output FROM hourly_production WHERE actual_start_time >= NOW() - INTERVAL 24 HOUR GROUP BY DATE_FORMAT(actual_start_time, '%Y-%m-%d %H:00:00') ORDER BY hour
    
    Question: Give each machine production with pie chart
    SQL Query: SELECT device_name, SUM(production_output) AS total_output FROM hourly_production GROUP BY device_name ORDER BY total_output DESC LIMIT 10
    
    Question: "Show all three machines consumption by each machine in April with bar chart for each 30 day"
    SQL Query: SELECT 
        DATE(actual_start_time) AS consumption_date,
        device_name AS machine_name,
        SUM(COALESCE(consumption_amount, power_consumption, energy_consumed, 0)) AS daily_consumption
    FROM hourly_production 
    WHERE YEAR(actual_start_time) = 2024 
        AND MONTH(actual_start_time) = 4 
    GROUP BY DATE(actual_start_time), device_name
    ORDER BY consumption_date, machine_name
    
    Question: "Compare energy usage by machine for last month"
    SQL Query: SELECT 
        DATE(actual_start_time) AS usage_date,
        device_name AS machine_name,
        SUM(COALESCE(energy_usage, power_consumption, electricity_used, 0)) AS daily_energy_usage
    FROM hourly_production 
    WHERE actual_start_time >= DATE_SUB(NOW(), INTERVAL 1 MONTH)
    GROUP BY DATE(actual_start_time), device_name
    ORDER BY usage_date, machine_name
    
    Question: "Show production efficiency by machine this year"
    SQL Query: SELECT 
        DATE_FORMAT(actual_start_time, '%Y-%m') AS production_month,
        device_name AS machine_name,
        AVG(COALESCE(efficiency_rate, performance_ratio, 
            (production_output / NULLIF(target_output, 0)) * 100, 0)) AS avg_efficiency
    FROM hourly_production 
    WHERE YEAR(actual_start_time) = 2024
    GROUP BY DATE_FORMAT(actual_start_time, '%Y-%m'), device_name
    ORDER BY production_month, machine_name
    
    Question: "Machine downtime comparison by day"
    SQL Query: SELECT 
        DATE(actual_start_time) AS downtime_date,
        device_name AS machine_name,
        SUM(COALESCE(downtime_minutes, offline_duration, maintenance_time, 0)) AS total_downtime_minutes
    FROM hourly_production 
    WHERE actual_start_time >= DATE_SUB(NOW(), INTERVAL 30 DAY)
    GROUP BY DATE(actual_start_time), device_name
    ORDER BY downtime_date, machine_name
    
    6. SAFETY AND VALIDATION:
       - Use COALESCE() to handle NULL values
       - Use NULLIF() to prevent division by zero
       - Filter out negative values where appropriate
       - Include WHERE clauses to ensure data quality
    
    Write only the SQL query and nothing else. Do not wrap it in backticks or other formatting.
    
    Question: {question}
    SQL Query:
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatGroq(model="llama3-8b-8192", temperature=0)
    
    def get_schema(_):
        return db.get_table_info()
    
    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )



def get_pulse_llm_chain():
    """Specialized LLM chain for pulse queries using GPT-4 Turbo"""
    template = """
    You are a pulse calculation specialist. Generate SQL queries specifically for pulse per minute calculations.

    Pulse = Current Length - Previous Length

    Guidelines:
    - Always retrieve device_name, timestamp, length columns
    - Order by device_name, timestamp for proper calculation
    - Use appropriate time filters based on user request

    Question: {question}
    SQL Query:
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-4-0125-preview", temperature=0)
    
    return prompt | llm | StrOutputParser()








def create_enhanced_visualization(df, chart_type, user_query, detected_metric, message_id=None):
    """Enhanced visualization with dynamic metric support"""
    try:
        logger.info(f"Creating visualization: {chart_type} for metric: {detected_metric}")
        logger.info(f"DataFrame shape: {df.shape}")
        logger.info(f"DataFrame columns: {list(df.columns)}")
        logger.info(f"DataFrame sample data:\n{df.head()}")
        
        if df.empty:
            logger.warning("DataFrame is empty")
            st.warning("No data available for visualization")
            return False
        
        
        
        
        
        
        
        # SPECIAL HANDLING FOR PULSE METRIC
        if detected_metric == 'pulse':
            logger.info("Processing pulse metric - calculating pulse per minute")
            
            # Check if we have the required columns for pulse calculation
            length_col = None
            timestamp_col = None
            device_col = None
            
            # Find columns flexibly
            for col in df.columns:
                if 'length' in col.lower():
                    length_col = col
                elif any(time_keyword in col.lower() for time_keyword in ['timestamp', 'time', 'date']):
                    timestamp_col = col
                elif any(device_keyword in col.lower() for device_keyword in ['device', 'machine', 'equipment']):
                    device_col = col
            
            if length_col and timestamp_col and device_col:
                # Calculate pulse per minute
                df = calculate_pulse_per_minute(df, length_col, timestamp_col, device_col)
                
                if df.empty:
                    st.warning("No pulse data available after calculation")
                    return False
                
                # Create line chart for pulse over time
                fig = px.line(
                    df, 
                    x=timestamp_col, 
                    y='pulse_per_minute',
                    color=device_col,
                    title=f"Pulse Per Minute Over Time",
                    labels={
                        timestamp_col: "Time",
                        'pulse_per_minute': "Pulse Per Minute",
                        device_col: "Device"
                    },
                    markers=True
                )
                
                fig.update_layout(
                    showlegend=True,
                    height=600,
                    margin=dict(l=50, r=50, t=80, b=50),
                    xaxis_title_font_size=14,
                    yaxis_title_font_size=14,
                    title_font_size=16
                )
                
                # Display the chart
                st.plotly_chart(fig, use_container_width=True)
                
                # Show pulse statistics
                with st.expander("📊 Pulse Statistics"):
                    st.write(f"**Total readings:** {len(df)}")
                    st.write(f"**Devices:** {', '.join(df[device_col].unique())}")
                    st.write(f"**Average pulse per minute:** {df['pulse_per_minute'].mean():.2f}")
                    st.write(f"**Max pulse per minute:** {df['pulse_per_minute'].max():.2f}")
                    st.write(f"**Min pulse per minute:** {df['pulse_per_minute'].min():.2f}")
                    
                    st.write("**Pulse Data:**")
                    st.dataframe(df)
                
                return True
            else:
                st.error("Required columns not found for pulse calculation. Need: length, timestamp, device_name")
                return False
        
        
        
        
        # Clean and prepare data
        df = df.dropna()
        if df.empty:
            logger.warning("DataFrame is empty after removing NaN values")
            st.warning("No valid data available after cleaning")
            return False
        
        # Identify column types
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64', 'datetime']).columns.tolist()

        
        # Handle datetime columns
        for col in df.columns:
            if df[col].dtype == 'object' and any(keyword in col.lower() for keyword in ['date', 'time']):
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    logger.info(f"Converted column {col} to datetime")
                except:
                    pass
        
        logger.info(f"Numeric columns: {numeric_cols}")
        logger.info(f"Categorical columns: {categorical_cols}")
        
        fig = None
        chart_title = ""
        
        if chart_type == "line":
            if datetime_cols and numeric_cols:
                x_col = datetime_cols[0]
                y_col = numeric_cols[0]
                fig = px.line(df, x=x_col, y=y_col, 
                             title=f"Line Chart: {y_col} over time")
            elif len(df.columns) >= 2:
                x_col = df.columns[0]
                y_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
                fig = px.line(df, x=x_col, y=y_col, 
                             title=f"Line Chart: {y_col} vs {x_col}")
                
        elif chart_type == "pie":
            if len(df.columns) >= 2:
                labels_col = df.columns[0]
                values_col = df.columns[1]
                fig = px.pie(df, names=labels_col, values=values_col, 
                           title=f"Pie Chart: {values_col} by {labels_col}")
                
        elif chart_type == "bar":
            if len(df.columns) >= 2:
                x_col = df.columns[0]
                y_col = df.columns[1]
                fig = px.bar(df, x=x_col, y=y_col, 
                           title=f"Bar Chart: {y_col} by {x_col}")
                
        elif chart_type == "scatter":
            if len(numeric_cols) >= 2:
                x_col = numeric_cols[0]
                y_col = numeric_cols[1]
                fig = px.scatter(df, x=x_col, y=y_col, 
                               title=f"Scatter Plot: {y_col} vs {x_col}")
                
        elif chart_type == "histogram":
            if numeric_cols:
                col = numeric_cols[0]
                fig = px.histogram(df, x=col, title=f"Histogram of {col}")
        
        if fig:
            fig.update_layout(
                showlegend=True,
                height=500,
                margin=dict(l=50, r=50, t=50, b=50)
            )
            st.plotly_chart(fig, use_container_width=True)
            return True
        
        
        
        
        
        
        # Enhanced multi-machine visualization with metric-specific handling
        if chart_type == "multi_machine_bar" or chart_type == "grouped_bar":
            # Flexible column detection
            date_col = None
            machine_col = None
            value_col = None
            
            # Find date column (flexible naming)
            for col in df.columns:
                if any(keyword in col.lower() for keyword in ['date', 'time']):
                    date_col = col
                    break
            
            # Find machine column (flexible naming)
            for col in df.columns:
                if any(keyword in col.lower() for keyword in ['machine', 'device', 'equipment']):
                    machine_col = col
                    break
            
            # Find value column based on detected metric
            metric_column_patterns = {
                'production': ['production', 'output', 'produced', 'manufactured'],
                'consumption': ['consumption', 'consumed', 'usage', 'used'],
                'efficiency': ['efficiency', 'performance', 'yield'],
                'downtime': ['downtime', 'offline', 'maintenance'],
                'energy': ['energy', 'power', 'electricity', 'kwh'],
                'temperature': ['temperature', 'temp', 'heat'],
                'speed': ['speed', 'rpm', 'rate', 'velocity']
            }
            
            # Look for metric-specific columns first
            if detected_metric in metric_column_patterns:
                for col in df.columns:
                    if any(pattern in col.lower() for pattern in metric_column_patterns[detected_metric]):
                        value_col = col
                        break
            
            # If no metric-specific column found, use any numeric column
            if not value_col and numeric_cols:
                value_col = numeric_cols[0]
            
            logger.info(f"Detected columns - Date: {date_col}, Machine: {machine_col}, Value: {value_col}")
            
            if date_col and machine_col and value_col:
                # Create metric-specific title and labels
                metric_labels = {
                    'production': {'title': 'Production Output', 'y_label': 'Production Units'},
                    'consumption': {'title': 'Consumption', 'y_label': 'Consumption Amount'},
                    'efficiency': {'title': 'Efficiency', 'y_label': 'Efficiency %'},
                    'downtime': {'title': 'Downtime', 'y_label': 'Downtime (minutes)'},
                    'energy': {'title': 'Energy Usage', 'y_label': 'Energy (kWh)'},
                    'temperature': {'title': 'Temperature', 'y_label': 'Temperature (°C)'},
                    'speed': {'title': 'Speed', 'y_label': 'Speed (RPM)'}
                }
                
                labels = metric_labels.get(detected_metric, {'title': 'Value', 'y_label': 'Value'})
                chart_title = f"{labels['title']} by Machine and Date"
                
                # Create color palette for machines
                unique_machines = df[machine_col].unique()
                colors = px.colors.qualitative.Set3[:len(unique_machines)]
                
                fig = px.bar(
                    df, 
                    x=date_col, 
                    y=value_col, 
                    color=machine_col,
                    title=chart_title,
                    labels={
                        date_col: "Date",
                        value_col: labels['y_label'],
                        machine_col: "Machine"
                    },
                    barmode='group',
                    color_discrete_sequence=colors
                )
                
                # Enhance the chart appearance
                fig.update_layout(
                    showlegend=True,
                    height=600,
                    margin=dict(l=50, r=50, t=80, b=50),
                    xaxis_title_font_size=14,
                    yaxis_title_font_size=14,
                    title_font_size=16,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                # Improve x-axis for dates
                if df[date_col].dtype == 'datetime64[ns]':
                    fig.update_xaxes(
                        tickangle=45,
                        tickformat='%Y-%m-%d'
                    )
                
                # Add metric-specific formatting
                if detected_metric == 'efficiency':
                    fig.update_yaxes(tickformat='.1%')
                elif detected_metric in ['energy', 'consumption']:
                    fig.update_yaxes(tickformat='.2f')
                
            else:
                # Fallback to regular grouped bar chart
                if len(categorical_cols) >= 2 and len(numeric_cols) >= 1:
                    x_col = categorical_cols[0]
                    color_col = categorical_cols[1]
                    y_col = numeric_cols[0]
                    chart_title = f"Grouped Bar Chart: {y_col} by {x_col} and {color_col}"
                    fig = px.bar(df, x=x_col, y=y_col, color=color_col,
                               title=chart_title,
                               barmode='group')
        
        # Other chart types with metric awareness
        #elif chart_type == "line":
            #if len(df.columns) >= 2:
                #x_col = df.columns[0]
                #y_col = numeric_cols[0] if numeric_cols else df.columns[1]
                #color_col = None
                
                # Check if we have machine data for multi-line chart
                #for col in df.columns:
                    #if any(keyword in col.lower() for keyword in ['machine', 'device']):
                        #color_col = col
                        #break
                
                #metric_name = detected_metric.capitalize()
                #chart_title = f"{metric_name} Trend: {y_col} over {x_col}"
                #fig = px.line(df, x=x_col, y=y_col, color=color_col,
                             #title=chart_title,
                             #markers=True)
                
        #elif chart_type == "pie":
            #if len(df.columns) >= 2:
                #labels_col = categorical_cols[0] if categorical_cols else df.columns[0]
                #values_col = numeric_cols[0] if numeric_cols else df.columns[1]
                #metric_name = detected_metric.capitalize()
                #chart_title = f"{metric_name} Distribution: {values_col} by {labels_col}"
                #fig = px.pie(df, names=labels_col, values=values_col, 
                           #title=chart_title)
                
        elif chart_type == "bar":
            if len(df.columns) >= 2:
                x_col = categorical_cols[0] if categorical_cols else df.columns[0]
                y_col = numeric_cols[0] if numeric_cols else df.columns[1]
                metric_name = detected_metric.capitalize()
                chart_title = f"{metric_name} Comparison: {y_col} by {x_col}"
                fig = px.bar(df, x=x_col, y=y_col, 
                           title=chart_title)
        
        # Display and store the chart
        if fig:
            # Create a unique container key for this visualization
            viz_key = f"viz_{message_id}_{len(st.session_state.stored_visualizations)}"
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=True, key=viz_key)
            
            # Store visualization data for persistence
            viz_data = {
                'figure': fig,
                'chart_type': chart_type,
                'title': chart_title,
                'user_query': user_query,
                'detected_metric': detected_metric,
                'message_id': message_id,
                'key': viz_key,
                'data_summary': {
                    'total_records': len(df),
                    'columns': list(df.columns),
                    'metric_type': detected_metric,
                    'data_sample': df.head().to_dict('records') if len(df) > 0 else []
                }
            }
            
            # Add machine info if available
            machine_col = None
            for col in df.columns:
                if any(keyword in col.lower() for keyword in ['machine', 'device']):
                    machine_col = col
                    break
            
            if machine_col:
                viz_data['data_summary']['unique_machines'] = df[machine_col].nunique()
                viz_data['data_summary']['machines'] = df[machine_col].unique().tolist()
            
            # Store in session state
            st.session_state.stored_visualizations.append(viz_data)
            
            logger.info("Visualization created and stored successfully")
            
            # Show enhanced data summary
            with st.expander(f"📊 {detected_metric.capitalize()} Data Summary"):
                st.write(f"**Metric Type:** {detected_metric.upper()}")
                st.write(f"**Total records:** {len(df)}")
                if machine_col:
                    unique_machines = df[machine_col].nunique()
                    st.write(f"**Unique machines:** {unique_machines}")
                    st.write(f"**Machines:** {', '.join(df[machine_col].unique())}")
                
                # Show statistical summary for numeric columns
                if numeric_cols:
                    st.write("**Statistical Summary:**")
                    st.dataframe(df[numeric_cols].describe())
                
                st.write("**Raw Data:**")
                st.dataframe(df)
            
            return True
        else:
            error_msg = f"Could not create {chart_type} chart for {detected_metric} with available data."
            logger.error(error_msg)
            st.error(error_msg)
            st.write("**Available data:**")
            st.dataframe(df.head(10))
            return False
            
    except Exception as e:
        error_msg = f"Error creating visualization: {str(e)}"
        logger.error(error_msg)
        st.error(error_msg)
        st.write("**Debug information:**")
        st.write("Data sample:")
        st.dataframe(df.head() if not df.empty else pd.DataFrame())
        return False

def is_greeting_or_casual(user_query: str) -> bool:
    """Detect if the user query is a greeting or casual conversation"""
    user_query_lower = user_query.lower().strip()
    
    # Common greetings and casual phrases
    greetings = [
        'hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening',
        'how are you', 'whats up', "what's up", 'yo', 'hiya', 'greetings'
    ]
    
    casual_phrases = [
        'thank you', 'thanks', 'bye', 'goodbye', 'see you', 'ok', 'okay',
        'cool', 'nice', 'great', 'awesome', 'perfect', 'got it', 'understand',
        'help', 'what can you do', 'how does this work', 'test'
    ]
    
    # Check if the query is just a greeting or casual phrase
    if user_query_lower in greetings + casual_phrases:
        return True
    
    # Check if query starts with greeting
    if any(user_query_lower.startswith(greeting) for greeting in greetings):
        return True
    
    # Check if it's a very short query without data-related keywords
    data_keywords = [
        'production', 'machine', 'data', 'show', 'chart', 'graph', 'plot',
        'select', 'table', 'database', 'query', 'april', 'month', 'day',
        'output', 'performance', 'efficiency', 'downtime', 'shift'
    ]
    
    if len(user_query_lower.split()) <= 3 and not any(keyword in user_query_lower for keyword in data_keywords):
        return True
    
    return False

def get_casual_response(user_query: str) -> str:
    """Generate appropriate responses for greetings and casual conversation"""
    user_query_lower = user_query.lower().strip()
    
    if any(greeting in user_query_lower for greeting in ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']):
        return """Hello! 👋 I'm your Althinect Intelligence Bot. I'm here to help you analyze your data and create visualizations."""

    elif any(phrase in user_query_lower for phrase in ['how are you', 'whats up', "what's up"]):
        return """I'm doing great, thank you! 😊 Ready to help you dive into your production data.

Is there any specific production analysis or visualization you'd like me to help you with?"""

    elif any(phrase in user_query_lower for phrase in ['thank you', 'thanks']):
        return """You're welcome! 😊 I'm here whenever you need help with production data analysis or creating visualizations.

Feel free to ask me about any production metrics you'd like to explore!"""

    elif any(phrase in user_query_lower for phrase in ['help', 'what can you do', 'how does this work']):
        return """I'm your Production Analytics Assistant! Here's how I can help:

**🎯 Data Analysis:**
- Query your production database
- Analyze machine performance
- Compare production across different time periods

**📊 Visualizations:**
- Multi-colored bar charts for each machine
- Line charts for trends over time
- Pie charts for production distribution
- Grouped comparisons

**💡 Example Questions:**
- "Show production by all machines in April"
- "Plot hourly production trends"
- "Compare efficiency across shifts"
- "Which machine had highest output last week?"

Just ask me a question about your production data, and I'll generate both the analysis and visualizations for you!"""

    elif user_query_lower in ['test', 'testing']:
        return """System test successful! ✅ 

I'm ready to help you with production data analysis. Try asking me about:
- Machine production data
- Time-based production trends  
- Performance comparisons
- Any other production metrics you'd like to explore

What would you like to analyze?"""

    else:
        return """I'm here to help with production data analysis and visualizations! 

Feel free to ask me about:
- Production data by machine, shift, or time period
- Creating charts and graphs
- Comparing performance metrics

What production data would you like to explore? 📊"""

def get_enhanced_response(user_query: str, db: SQLDatabase, chat_history: list):
    """Enhanced response function with greeting detection and better error handling"""
    try:
        logger.info(f"Processing user query: {user_query}")
        
        # Step 0: Check if this is a greeting or casual conversation
        if is_greeting_or_casual(user_query):
            logger.info("Detected greeting/casual conversation")
            return get_casual_response(user_query)
        
        # Step 1: Detect visualization needs
        detected_metric, needs_viz, chart_type = detect_metric_and_visualization(user_query)
        
        # Step 2: Generate SQL query with enhanced chain
        sql_chain = get_enhanced_sql_chain(db)
        sql_query = sql_chain.invoke({
            "question": user_query,
            "chat_history": chat_history
        })
        
        logger.info(f"Generated SQL query: {sql_query}")
        
        # Step 3: Execute SQL query with better error handling
        try:
            sql_response = db.run(sql_query)
            logger.info(f"SQL query executed successfully. Response length: {len(str(sql_response))}")
            
            # Check if response is empty
            if not sql_response or sql_response == "[]" or sql_response == "()":
                return "No data found for your query. This could be due to:\n\n1. **Date range issue**: The specified date range might not have data\n2. **Table structure**: Column names might be different\n3. **Data availability**: No records match your criteria\n\nPlease try:\n- Checking if data exists for the specified time period\n- Using a different date range\n- Asking about available tables or columns"
            
        except Exception as e:
            error_msg = f"SQL execution error: {str(e)}\n\n**Generated SQL Query:**\n```sql\n{sql_query}\n```\n\n**Possible issues:**\n1. Column names might be incorrect\n2. Table structure might be different\n3. Date format issues\n4. Data type mismatches"
            logger.error(error_msg)
            return error_msg
        
        # Step 4: Create visualization if needed
        chart_created = False
        current_message_id = st.session_state.message_counter
        
        if needs_viz:
            try:
                df = pd.read_sql(sql_query, db._engine)
                logger.info(f"DataFrame created with shape: {df.shape}")
                
                if df.empty:
                    st.warning("Query returned no data for visualization")
                else:
                    chart_created = create_enhanced_visualization(df, chart_type, user_query, current_message_id)
                    
            except Exception as e:
                error_msg = f"Visualization error: {str(e)}"
                logger.error(error_msg)
                st.error(error_msg)
        
        # Step 5: Generate natural language response
        template = """
        You are a data analyst providing insights about production data.
        
        Based on the SQL query results, provide a clear, informative response.
        
        SQL Query: {query}
        User Question: {question}
        SQL Response: {response}
        
        {visualization_note}
        
        Guidelines:
        1. Summarize key findings from the data
        2. Mention specific numbers/values when relevant
        3. If this is multi-machine data, highlight comparisons between machines
        4. If this is time-series data, mention trends or patterns
        5. Keep the response concise but informative
        """
        
        visualization_note = ""
        if needs_viz and chart_created:
            visualization_note = "Note: The visualization above shows the data in an interactive chart format with different colors for each machine."
        elif needs_viz and not chart_created:
            visualization_note = "Note: I attempted to create a visualization but encountered formatting issues. The raw data is available above."
        
        prompt = ChatPromptTemplate.from_template(template)
        llm = ChatGroq(model="llama3-8b-8192", temperature=0)
        
        chain = prompt | llm | StrOutputParser()
        
        response = chain.invoke({
            "question": user_query,
            "query": sql_query,
            "response": sql_response,
            "visualization_note": visualization_note
        })
        
        logger.info("Response generated successfully")
        return response
        
    except Exception as e:
        error_msg = f"An error occurred while processing your request: {str(e)}"
        logger.error(error_msg)
        return error_msg

# Streamlit UI
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! I'm Althinect Intelligence Bot. Ask me anything about your database. "),
    ]

# Initialize stored visualizations
if "stored_visualizations" not in st.session_state:
    st.session_state.stored_visualizations = []

# Initialize message counter for unique keys
if "message_counter" not in st.session_state:
    st.session_state.message_counter = 0

load_dotenv()

st.set_page_config(page_title="Althinect Intelligence Bot", page_icon="📊")

st.title("Althinect Intelligence Bot")

# Enhanced help section
with st.expander("🚀 Enhanced Features & Examples"):
    st.markdown("""
    **🎯 Multi-Machine Analysis:**
    - "Show all three machines production by each machine in April with bar chart"
    - "Compare daily production for all machines this month"
    - "Monthly production trends by machine for 2024"
    
    **📊 Visualization Types:**
    - **Multi-Machine Bar Charts**: Different colors for each machine
    - **Grouped Bar Charts**: Side-by-side comparisons
    - **Line Charts**: Trends over time by machine
    - **Pie Charts**: Production distribution
    
    **🔍 Troubleshooting:**
    - If you get "No data found": Check date ranges and table structure
    - If charts don't appear: Verify column names in your database
    - For best results: Ensure your table has columns like 'device_name', 'actual_start_time', 'production_output'
    """)

with st.sidebar:
    st.subheader("⚙️ Database Settings")
    st.write("Connect to your MySQL database")
    
    st.text_input("Host", value="localhost", key="Host")
    st.text_input("Port", value="3306", key="Port")
    st.text_input("User", value="root", key="User")
    st.text_input("Password", type="password", value="chama:1234", key="Password")
    st.text_input("Database", value="Analyzee_machines", key="Database")
    
    if st.button("🔌 Connect", type="primary"):
        with st.spinner("Connecting to database..."):
            try:
                db = init_database(
                    st.session_state["User"],
                    st.session_state["Password"],
                    st.session_state["Host"],
                    st.session_state["Port"],
                    st.session_state["Database"]
                )
                st.session_state.db = db
                st.success("✅ Connected to database!")
                logger.info("Database connected successfully via UI")
            except Exception as e:
                error_msg = f"❌ Connection failed: {str(e)}"
                st.error(error_msg)
                logger.error(f"Database connection failed via UI: {str(e)}")
    
    if "db" in st.session_state:
        st.success("🟢 Database Connected")
    else:
        st.warning("🔴 Database Not Connected")

# Chat interface with persistent visualizations
for i, message in enumerate(st.session_state.chat_history):
    if isinstance(message, AIMessage):
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(message.content)


user_query = st.chat_input("💬 Ask me anything about your database...")
if user_query is not None and user_query.strip() != "":
    logger.info(f"User query received: {user_query}")
    
    # Increment message counter for unique identification
    st.session_state.message_counter += 1
    current_message_id = len(st.session_state.chat_history)
    
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_query)
        
    with st.chat_message("assistant", avatar="🤖"):
        # Check if it's a greeting first (no database needed)
        if is_greeting_or_casual(user_query):
            response = get_casual_response(user_query)
            st.markdown(response)
        elif "db" in st.session_state:
            with st.spinner("🔄 Analyzing data and creating visualization..."):
                response = get_enhanced_response(user_query, st.session_state.db, st.session_state.chat_history)
                st.markdown(response)
        else:
            response = "⚠️ Please connect to the database first using the sidebar to analyze production data."
            st.markdown(response)
            logger.warning("User attempted to query without database connection")
        
    st.session_state.chat_history.append(AIMessage(content=response))
    logger.info("Conversation turn completed")

