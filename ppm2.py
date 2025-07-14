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
import os

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

def get_optimized_schema(db, max_tables=5):
    """Get optimized schema to reduce token usage"""
    try:
        # Get table names
        table_names = db.get_usable_table_names()
        
        # Limit number of tables to avoid token overflow
        if len(table_names) > max_tables:
            table_names = table_names[:max_tables]
        
        # Get simplified schema
        schema_info = []
        for table in table_names:
            try:
                # Get column info with sample data
                sample_query = f"SELECT * FROM {table} LIMIT 3"
                sample_data = db.run(sample_query)
                
                # Get column names only
                columns_query = f"SHOW COLUMNS FROM {table}"
                columns_info = db.run(columns_query)
                
                schema_info.append(f"Table: {table}")
                schema_info.append(f"Columns: {columns_info}")
                schema_info.append(f"Sample data: {sample_data}")
                schema_info.append("---")
                
            except Exception as e:
                logger.warning(f"Could not get info for table {table}: {str(e)}")
                schema_info.append(f"Table: {table} (info unavailable)")
        
        return "\n".join(schema_info)
    except Exception as e:
        logger.error(f"Error getting optimized schema: {str(e)}")
        return db.get_table_info()

def get_llm_with_fallback():
    """Get LLM with fallback options"""
    try:
        # Try Groq first with smaller model
        return ChatGroq(model="llama3-8b-8192", temperature=0, max_tokens=1000)
    except Exception as e:
        logger.warning(f"Groq failed, trying OpenAI: {str(e)}")
        try:
            # Fallback to OpenAI if available
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if openai_api_key:
                return ChatOpenAI(model="gpt-3.5-turbo", temperature=0, max_tokens=1000)
            else:
                logger.error("No OpenAI API key found")
                return None
        except Exception as e2:
            logger.error(f"OpenAI fallback failed: {str(e2)}")
            return None

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
        'pulse': ['pulse', 'pulse per minute', 'pulse rate', 'length difference', 'increment', 'ppm']
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

def get_compact_sql_chain(db):
    """Compact SQL chain with reduced token usage"""
    template = """
    Based on the schema, write a SQL query for the question.
        
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
       - PULSE: length, length_data, pulse_data (for calculating pulse per minute)
    
    2. FLEXIBLE COLUMN MAPPING - Adapt to available columns:
       - For machine identification: device_name, machine_name, machine_id, equipment_name
       - For time columns: actual_start_time, timestamp, date_time, created_at, time_stamp
       - For metric values: Look for columns that match the requested metric type
    
    3. PULSE-SPECIFIC QUERIES - CRITICAL FOR PULSE CALCULATIONS:
       - For pulse per minute: ALWAYS select device_name, timestamp, length columns
       - Order by device_name, timestamp for proper calculation
       - Include recent data for trend analysis
       - Use time filters to get relevant data (last hour, today, etc.)
       
       PULSE QUERY EXAMPLES:
       - "Show pulse per minute for last hour": 
         SELECT device_name, timestamp, length FROM length_data 
         WHERE timestamp >= DATE_SUB(NOW(), INTERVAL 1 HOUR) 
         ORDER BY device_name, timestamp
         
       - "Real time pulse per minute": 
         SELECT device_name, timestamp, length FROM length_data 
         WHERE timestamp >= DATE_SUB(NOW(), INTERVAL 30 MINUTE) 
         ORDER BY device_name, timestamp
         
       - "Pulse trends for today": 
         SELECT device_name, timestamp, length FROM length_data 
         WHERE DATE(timestamp) = CURDATE() 
         ORDER BY device_name, timestamp
         
       - "Compare pulse rates by machine": 
         SELECT device_name, timestamp, length FROM length_data 
         WHERE timestamp >= DATE_SUB(NOW(), INTERVAL 2 HOUR) 
         ORDER BY device_name, timestamp
    
    4. TIME PERIOD HANDLING:
       - April 2024: WHERE YEAR(time_column) = 2024 AND MONTH(time_column) = 4
       - Daily grouping: GROUP BY DATE(time_column)
       - Monthly grouping: GROUP BY DATE_FORMAT(time_column, '%Y-%m')
       - Hourly grouping: GROUP BY DATE_FORMAT(time_column, '%Y-%m-%d %H:00:00')
       - Real-time/Recent: Use DATE_SUB(NOW(), INTERVAL X MINUTE/HOUR)
       
    5. MULTI-MACHINE QUERIES:
       - Always include machine identifier as separate column
       - Use appropriate aggregation: SUM(), AVG(), MAX(), MIN() based on metric
       - GROUP BY time period AND machine identifier
       - ORDER BY time period, then machine name
    
    6. ADAPTIVE QUERY STRUCTURE:
       - Pulse queries: Focus on length and timestamp columns
       - Consumption queries: Focus on consumption-related columns
       - Production queries: Focus on production-related columns
       - Comparison queries: Include multiple machines and time grouping
    
    7. SAFETY AND VALIDATION:
       - Use COALESCE() to handle NULL values
       - Use NULLIF() to prevent division by zero
       - Filter out negative values where appropriate
       - Include WHERE clauses to ensure data quality
    
    Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.
    
    For visualization requests, ensure the query returns data suitable for charting:
    - For line charts: include time-based columns and numeric values
    - For pie charts: include category names and corresponding values
    - For bar charts: include categories and numeric values
    - Limit results to reasonable amounts (e.g., LIMIT 100 for large datasets)
    
    Examples:
    Question: Which machine produced the most output in the last hour?
    SQL Query: SELECT device_name, SUM(production_output) AS total_output FROM hourly_production WHERE actual_start_time >= NOW() - INTERVAL 1 HOUR GROUP BY device_name ORDER BY total_output DESC LIMIT 1
    
    Question: Plot the hourly production with line graph
    SQL Query: SELECT DATE_FORMAT(actual_start_time, '%Y-%m-%d %H:00:00') AS hour, SUM(production_output) AS total_output FROM hourly_production WHERE actual_start_time >= NOW() - INTERVAL 24 HOUR GROUP BY DATE_FORMAT(actual_start_time, '%Y-%m-%d %H:00:00') ORDER BY hour
    
    Question: Show pulse per minute for all machines
    SQL Query: SELECT device_name, timestamp, length FROM length_data WHERE timestamp >= DATE_SUB(NOW(), INTERVAL 1 HOUR) ORDER BY device_name, timestamp
    
    Question: Real time pulse per minute trends
    SQL Query: SELECT device_name, timestamp, length FROM length_data WHERE timestamp >= DATE_SUB(NOW(), INTERVAL 30 MINUTE) ORDER BY device_name, timestamp
    
    Question: Pulse rate comparison by machine today
    SQL Query: SELECT device_name, timestamp, length FROM length_data WHERE DATE(timestamp) = CURDATE() ORDER BY device_name, timestamp
    
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
    
    Write only the SQL query and nothing else. Do not wrap it in backticks or other formatting.
    
    RULES:
    1. For PULSE queries: Select device_name, timestamp, length columns
    2. For PRODUCTION: Select device_name, production_output, time columns
    3. For CONSUMPTION: Select device_name, consumption_amount, time columns
    4. Use appropriate WHERE clauses for time filters
    5. Order by time for trends
    6. Limit results to 1000 rows max
    
    Question: {question}
    SQL Query:
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Get LLM with fallback
    llm = get_llm_with_fallback()
    if not llm:
        raise Exception("No LLM available")
    
    def get_schema(_):
        return get_optimized_schema(db, max_tables=3)
    
    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )

def get_pulse_sql_chain(db):
    """Specialized compact SQL chain for pulse per minute queries"""
    template = """
    Write SQL for pulse calculation from this schema:
    
    <SCHEMA>{schema}</SCHEMA>
    
    
    PULSE CALCULATION REQUIREMENTS:
    - Pulse = Current Length - Previous Length (calculated per device)
    - Always retrieve: device_name, timestamp, length columns
    - Order by device_name, timestamp for proper calculation
    - Use appropriate time filters based on user request

    TIME FILTERS FOR PULSE QUERIES:
    - "real time" or "current": last 30 minutes
    - "last hour": last 1 hour  
    - "today": current date
    - "recent": last 2 hours
    - No time specified: last 1 hour (default)

    PULSE QUERY PATTERNS:
    - Real-time: WHERE timestamp >= DATE_SUB(NOW(), INTERVAL 30 MINUTE)
    - Last hour: WHERE timestamp >= DATE_SUB(NOW(), INTERVAL 1 HOUR)
    - Today: WHERE DATE(timestamp) = CURDATE()
    - Recent: WHERE timestamp >= DATE_SUB(NOW(), INTERVAL 2 HOUR)

    Write only the SQL query and nothing else. Do not wrap in backticks.


    PULSE RULES:
    - Select: device_name, timestamp, length
    - Order by: device_name, timestamp
    - Time filters: last 30 minutes for real-time, last hour for trends
    - Limit: 500 rows
    
    Question: {question}
    SQL Query:
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Get LLM with fallback
    llm = get_llm_with_fallback()
    if not llm:
        raise Exception("No LLM available")
    
    def get_schema(_):
        return get_optimized_schema(db, max_tables=2)
    
    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )

def create_enhanced_visualization(df, chart_type, user_query, detected_metric, message_id=None):
    """Enhanced visualization with dynamic metric support"""
    try:
        logger.info(f"Creating visualization: {chart_type} for metric: {detected_metric}")
        logger.info(f"DataFrame shape: {df.shape}")
        logger.info(f"DataFrame columns: {list(df.columns)}")
        
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
                df_pulse = calculate_pulse_per_minute(df, length_col, timestamp_col, device_col)
                
                if df_pulse.empty:
                    st.warning("No pulse data available after calculation")
                    return False
                
                # Create real-time style line chart for pulse
                fig = px.line(
                    df_pulse, 
                    x=timestamp_col, 
                    y='pulse_per_minute',
                    color=device_col,
                    title=f"Real-Time Pulse Per Minute",
                    labels={
                        timestamp_col: "Time",
                        'pulse_per_minute': "Pulse Per Minute",
                        device_col: "Device"
                    },
                    markers=True,
                    line_shape='linear'
                )
                
                # Enhanced styling for real-time appearance
                fig.update_layout(
                    showlegend=True,
                    height=600,
                    margin=dict(l=50, r=50, t=80, b=50),
                    xaxis_title_font_size=14,
                    yaxis_title_font_size=14,
                    title_font_size=16,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(128,128,128,0.2)',
                        tickformat='%H:%M:%S'
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(128,128,128,0.2)'
                    )
                )
                
                # Add hover template for better real-time info
                fig.update_traces(
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                  'Time: %{x}<br>' +
                                  'Pulse/Min: %{y:.2f}<br>' +
                                  '<extra></extra>'
                )
                
                # Display the chart
                st.plotly_chart(fig, use_container_width=True)
                
                # Show enhanced pulse statistics
                with st.expander("📊 Real-Time Pulse Statistics"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Readings", len(df_pulse))
                        st.metric("Active Devices", df_pulse[device_col].nunique())
                    
                    with col2:
                        st.metric("Avg Pulse/Min", f"{df_pulse['pulse_per_minute'].mean():.2f}")
                        st.metric("Max Pulse/Min", f"{df_pulse['pulse_per_minute'].max():.2f}")
                    
                    with col3:
                        st.metric("Min Pulse/Min", f"{df_pulse['pulse_per_minute'].min():.2f}")
                        latest_time = df_pulse[timestamp_col].max()
                        st.metric("Latest Reading", latest_time.strftime("%H:%M:%S"))
                    
                    st.write("**Device Summary:**")
                    device_summary = df_pulse.groupby(device_col).agg({
                        'pulse_per_minute': ['count', 'mean', 'max', 'min'],
                        timestamp_col: 'max'
                    }).round(2)
                    device_summary.columns = ['Readings', 'Avg PPM', 'Max PPM', 'Min PPM', 'Latest Time']
                    st.dataframe(device_summary)
                    
                    st.write("**Recent Pulse Data:**")
                    st.dataframe(df_pulse.tail(20))
                
                return True
            else:
                st.error("Required columns not found for pulse calculation. Need: length, timestamp, device_name")
                st.write("Available columns:", list(df.columns))
                return False
        
        # Handle other visualizations with simplified logic
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if not numeric_cols:
            st.error("No numeric columns found for visualization")
            return False
        
        fig = None
        
        if chart_type == "line" and len(df.columns) >= 2:
            x_col = df.columns[0]
            y_col = numeric_cols[0]
            fig = px.line(df, x=x_col, y=y_col, title=f"Line Chart: {y_col}")
        elif chart_type == "bar" and len(df.columns) >= 2:
            x_col = df.columns[0]
            y_col = numeric_cols[0]
            fig = px.bar(df, x=x_col, y=y_col, title=f"Bar Chart: {y_col}")
        elif chart_type == "pie" and len(df.columns) >= 2:
            labels_col = df.columns[0]
            values_col = numeric_cols[0]
            fig = px.pie(df, names=labels_col, values=values_col, title=f"Pie Chart: {values_col}")
        
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("📊 Data Summary"):
                st.write(f"**Total records:** {len(df)}")
                st.write("**Statistical Summary:**")
                st.dataframe(df[numeric_cols].describe())
                st.write("**Raw Data:**")
                st.dataframe(df)
            
            return True
        else:
            st.error("Could not create visualization with available data")
            return False
            
    except Exception as e:
        error_msg = f"Error creating visualization: {str(e)}"
        logger.error(error_msg)
        st.error(error_msg)
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
        'output', 'performance', 'efficiency', 'downtime', 'shift', 'pulse'
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
- Calculate pulse per minute from length data

**📊 Visualizations:**
- Multi-colored bar charts for each machine
- Line charts for trends over time
- Pie charts for production distribution
- Grouped comparisons
- Real-time pulse monitoring

**💡 Example Questions:**
- "Show production by all machines in April"
- "Plot hourly production trends"
- "Show pulse per minute for all machines"
- "Real time pulse per minute trends"
- "Compare efficiency across shifts"

Just ask me a question about your production data, and I'll generate both the analysis and visualizations for you!"""

    elif user_query_lower in ['test', 'testing']:
        return """System test successful! ✅ 

I'm ready to help you with production data analysis. Try asking me about:
- Machine production data
- Time-based production trends  
- Performance comparisons
- Pulse per minute calculations
- Any other production metrics you'd like to explore

What would you like to analyze?"""

    else:
        return """I'm here to help with production data analysis and visualizations! 

Feel free to ask me about:
- Production data by machine, shift, or time period
- Creating charts and graphs
- Pulse per minute calculations
- Comparing performance metrics

What production data would you like to explore? 📊"""

def get_enhanced_response(user_query: str, db: SQLDatabase, chat_history: list):
    """Enhanced response function with better error handling and token management"""
    try:
        logger.info(f"Processing user query: {user_query}")
        
        # Step 0: Check if this is a greeting or casual conversation
        if is_greeting_or_casual(user_query):
            logger.info("Detected greeting/casual conversation")
            return get_casual_response(user_query)
        
        # Step 1: Detect visualization needs
        detected_metric, needs_viz, chart_type = detect_metric_and_visualization(user_query)
        
        # Step 2: Generate SQL query with compact chain
        try:
            if detected_metric == 'pulse':
                sql_chain = get_pulse_sql_chain(db)
            else:
                sql_chain = get_compact_sql_chain(db)
            
            sql_query = sql_chain.invoke({
                "question": user_query,
                "chat_history": chat_history[-2:] if len(chat_history) > 2 else chat_history  # Limit history
            })
            
            logger.info(f"Generated SQL query: {sql_query}")
            
        except Exception as e:
            error_msg = f"Error generating SQL query: {str(e)}"
            logger.error(error_msg)
            
            # Fallback to simple predefined queries for pulse
            if detected_metric == 'pulse':
                sql_query = """
                SELECT device_name, timestamp, length 
                FROM length_data 
                WHERE timestamp >= DATE_SUB(NOW(), INTERVAL 1 HOUR) 
                ORDER BY device_name, timestamp 
                LIMIT 500
                """
            else:
                return f"Error generating query: {str(e)}"
        
        # Step 3: Execute SQL query
        try:
            sql_response = db.run(sql_query)
            logger.info(f"SQL query executed successfully")
            
            if not sql_response or sql_response == "[]" or sql_response == "()":
                return "No data found for your query. Please check if data exists for the specified time period."
            
        except Exception as e:
            error_msg = f"SQL execution error: {str(e)}"
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
                    chart_created = create_enhanced_visualization(df, chart_type, user_query, detected_metric, current_message_id)
                    
            except Exception as e:
                error_msg = f"Visualization error: {str(e)}"
                logger.error(error_msg)
                st.error(error_msg)
        
        # Step 5: Generate simple response
        if chart_created:
            if detected_metric == 'pulse':
                response = f"Here's your real-time pulse per minute analysis. The chart shows the pulse rate trends for all devices with interactive visualization."
            else:
                response = f"Here's your {detected_metric} analysis with visualization. The chart shows the data trends and comparisons."
        else:
            response = f"Query executed successfully. Found {len(str(sql_response))} characters of data."
        
        # Add SQL query info for debugging
        response += f"\n\n**SQL Query Used:**\n```sql\n{sql_query}\n```"
        
        logger.info("Response generated successfully")
        return response
        
    except Exception as e:
        error_msg = f"An error occurred while processing your request: {str(e)}"
        logger.error(error_msg)
        return error_msg

# Streamlit UI
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! I'm Althinect Intelligence Bot. Ask me anything about your database."),
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
    
    **📊 Pulse Analysis:**
    - "Show pulse per minute for all machines"
    - "Real time pulse per minute trends"
    - "Pulse rate comparison by machine today"
    
    **📈 Visualization Types:**
    - **Multi-Machine Bar Charts**: Different colors for each machine
    - **Line Charts**: Trends over time by machine
    - **Pie Charts**: Production distribution
    - **Real-time Pulse Charts**: Live pulse monitoring
    
    **🔧 System Features:**
    - Automatic model fallback (Groq → OpenAI)
    - Optimized token usage
    - Smart error handling
    - Compact schema processing
    """)

with st.sidebar:
    st.subheader("⚙️ Database Settings")
    st.write("Connect to your MySQL database")
    
    st.text_input("Host", value="localhost", key="Host")
    st.text_input("Port", value="3306", key="Port")
    st.text_input("User", value="root", key="User")
    st.text_input("Password", type="password", value="chama:1234", key="Password")
    st.text_input("Database", value="Analyzee_machines", key="Database")
    
    # API Key settings
    st.subheader("🔑 API Keys (Optional)")
    st.text_input("OpenAI API Key", type="password", key="openai_key", help="For fallback when Groq fails")
    
    if st.session_state.get("openai_key"):
        os.environ["OPENAI_API_KEY"] = st.session_state.openai_key
    
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
        
        # Show available LLMs
        st.subheader("🤖 Available Models")
        try:
            llm = get_llm_with_fallback()
            if llm:
                st.success("✅ LLM Ready")
            else:
                st.error("❌ No LLM available")
        except Exception as e:
            st.error(f"❌ LLM Error: {str(e)}")
    
    # Database info section
    if "db" in st.session_state:
        with st.expander("📋 Database Info"):
            try:
                table_names = st.session_state.db.get_usable_table_names()
                st.write(f"**Tables ({len(table_names)}):**")
                for table in table_names:
                    st.write(f"• {table}")
            except Exception as e:
                st.error(f"Error getting table info: {str(e)}")

# Chat interface with persistent visualizations
for i, message in enumerate(st.session_state.chat_history):
    if isinstance(message, AIMessage):
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("user", avatar="👤"):
            st.markdown(message.content)

# Chat input
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

# Footer with system info
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Chat History", len(st.session_state.chat_history))

with col2:
    if "db" in st.session_state:
        st.metric("Database", "Connected", delta="✅")
    else:
        st.metric("Database", "Disconnected", delta="❌")

with col3:
    try:
        llm = get_llm_with_fallback()
        if llm:
            st.metric("LLM Status", "Ready", delta="✅")
        else:
            st.metric("LLM Status", "Unavailable", delta="❌")
    except:
        st.metric("LLM Status", "Error", delta="❌")

# Additional features
st.markdown("---")
st.subheader("🛠️ System Controls")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🔄 Clear Chat History"):
        st.session_state.chat_history = [
            AIMessage(content="Hello! I'm Althinect Intelligence Bot. Ask me anything about your database."),
        ]
        st.rerun()

with col2:
    if st.button("🔍 Test Database Connection"):
        if "db" in st.session_state:
            try:
                tables = st.session_state.db.get_usable_table_names()
                st.success(f"✅ Connection OK! Found {len(tables)} tables.")
            except Exception as e:
                st.error(f"❌ Connection failed: {str(e)}")
        else:
            st.error("No database connection found.")

with col3:
    if st.button("📊 Show System Stats"):
        with st.expander("System Statistics", expanded=True):
            st.write("**Session Information:**")
            st.write(f"- Messages in history: {len(st.session_state.chat_history)}")
            st.write(f"- Current message ID: {st.session_state.message_counter}")
            st.write(f"- Database connected: {'Yes' if 'db' in st.session_state else 'No'}")
            
            if "db" in st.session_state:
                try:
                    tables = st.session_state.db.get_usable_table_names()
                    st.write(f"- Available tables: {len(tables)}")
                    st.write(f"- Table names: {', '.join(tables)}")
                except:
                    st.write("- Table info: Unable to retrieve")
            
            st.write("**Environment:**")
            st.write(f"- Python version: {sys.version}")
            st.write(f"- Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Debug section (only show if there are errors)
if st.session_state.get("debug_mode", False):
    st.markdown("---")
    st.subheader("🐛 Debug Information")
    
    with st.expander("Debug Logs"):
        st.text("Recent log entries will appear here...")
        
        # Show last few log entries if available
        try:
            with open('chatbot.log', 'r') as f:
                logs = f.readlines()
                if logs:
                    st.text("Last 10 log entries:")
                    for log in logs[-10:]:
                        st.text(log.strip())
        except:
            st.text("No log file found or unable to read.")

# Safety and error handling
def safe_execute_query(db, query):
    """Safely execute SQL query with error handling"""
    try:
        result = db.run(query)
        return result, None
    except Exception as e:
        error_msg = f"SQL execution error: {str(e)}"
        logger.error(error_msg)
        return None, error_msg

# Session state cleanup function
def cleanup_session():
    """Clean up session state if needed"""
    if len(st.session_state.chat_history) > 100:  # Limit chat history
        st.session_state.chat_history = st.session_state.chat_history[-50:]
        logger.info("Chat history trimmed for performance")

# Auto-cleanup on every run
cleanup_session()

# Custom CSS for better appearance
st.markdown("""
<style>
    .stAlert {
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    
    .stExpander {
        border: 1px solid #ddd;
        border-radius: 4px;
        margin: 0.5rem 0;
    }
    
    .stMetric {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 4px;
        border: 1px solid #dee2e6;
    }
    
    .stButton button {
        width: 100%;
    }
    
    .success-message {
        padding: 1rem;
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
        border-radius: 4px;
        margin: 1rem 0;
    }
    
    .error-message {
        padding: 1rem;
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
        border-radius: 4px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Main execution guard
if __name__ == "__main__":
    logger.info("Streamlit app started")
    logger.info(f"Session initialized at: {datetime.now()}")
    
    # Show startup message
    if len(st.session_state.chat_history) == 1:  # Only initial message
        st.balloons()
        st.success("🎉 Althinect Intelligence Bot is ready! Connect to your database to get started.")