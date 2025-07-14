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

# Configure logging
def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('chatbot.log'),
            logging.StreamHandler(sys.stdout)  # This will show logs in VS Code terminal
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

def detect_visualization_request(user_query: str):
    """Detect if user wants a visualization and what type with enhanced logging"""
    user_query_lower = user_query.lower()
    logger.info(f"Analyzing query for visualization: {user_query}")
    
    # Keywords that indicate visualization request
    viz_keywords = [
        'plot', 'chart', 'graph', 'visualize', 'show', 'display',
        'bar chart', 'line chart', 'pie chart', 'histogram', 'scatter plot',
        'bar', 'line', 'pie', 'trend', 'comparison', 'grouped bar', 'stacked bar'
    ]
    
    needs_viz = any(keyword in user_query_lower for keyword in viz_keywords)
    
    # Determine chart type based on keywords
    chart_type = "bar"  # default
    if any(word in user_query_lower for word in ['line', 'trend', 'over time', 'hourly', 'daily', 'time series']):
        chart_type = "line"
    elif any(word in user_query_lower for word in ['pie', 'proportion', 'percentage', 'share', 'distribution']):
        chart_type = "pie"
    elif any(word in user_query_lower for word in ['scatter', 'relationship', 'correlation']):
        chart_type = "scatter"
    elif any(word in user_query_lower for word in ['histogram', 'distribution', 'frequency']):
        chart_type = "histogram"
    elif any(word in user_query_lower for word in ['grouped', 'stacked', 'multiple']):
        chart_type = "grouped_bar"
    
    logger.info(f"Visualization needed: {needs_viz}, Chart type: {chart_type}")
    return needs_viz, chart_type

def get_sql_chain(db):
    """Enhanced SQL chain with better query generation"""
    template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.
    
    <SCHEMA>{schema}</SCHEMA>
    
    Conversation History: {chat_history}
    
    Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.
    
    For visualization requests, ensure the query returns data suitable for charting:
    - For line charts: include time-based columns and numeric values
    - For pie charts: include category names and corresponding values
    - For bar charts: include categories and numeric values
    - For grouped/multi-column charts: include multiple categorical columns and values
    - Always include column aliases for clarity
    - Limit results to reasonable amounts (e.g., LIMIT 50 for large datasets)
    - Handle NULL values appropriately
    
    Examples:
    Question: Which machine produced the most output in the last hour?
    SQL Query: SELECT device_name, SUM(production_output) AS total_output FROM hourly_production WHERE actual_start_time >= NOW() - INTERVAL 1 HOUR GROUP BY device_name ORDER BY total_output DESC LIMIT 1
    
    Question: Plot the hourly production with line graph
    SQL Query: SELECT DATE_FORMAT(actual_start_time, '%Y-%m-%d %H:00:00') AS hour, SUM(production_output) AS total_output FROM hourly_production WHERE actual_start_time >= NOW() - INTERVAL 24 HOUR GROUP BY DATE_FORMAT(actual_start_time, '%Y-%m-%d %H:00:00') ORDER BY hour
    
    Question: Show production by machine and shift with grouped bar chart
    SQL Query: SELECT device_name, shift_type, SUM(production_output) AS total_output FROM hourly_production WHERE actual_start_time >= NOW() - INTERVAL 7 DAY GROUP BY device_name, shift_type ORDER BY device_name, shift_type LIMIT 50
    
    Your turn:
    
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

def create_visualization(df, chart_type, user_query):
    """Enhanced visualization function with multi-column support and better error handling"""
    try:
        logger.info(f"Creating visualization: {chart_type}")
        logger.info(f"DataFrame shape: {df.shape}")
        logger.info(f"DataFrame columns: {list(df.columns)}")
        logger.info(f"DataFrame dtypes: {df.dtypes.to_dict()}")
        
        if df.empty:
            logger.warning("DataFrame is empty")
            st.warning("No data available for visualization")
            return False
        
        # Clean and prepare data
        df = df.dropna()  # Remove rows with NaN values
        if df.empty:
            logger.warning("DataFrame is empty after removing NaN values")
            st.warning("No valid data available after cleaning")
            return False
        
        # Get column information
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = []
        
        # Handle potential datetime columns
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    # Try to parse a sample value as datetime
                    sample_val = df[col].iloc[0]
                    if pd.to_datetime(sample_val, errors='coerce') is not pd.NaT:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                        datetime_cols.append(col)
                        logger.info(f"Converted column {col} to datetime")
                except:
                    pass
        
        logger.info(f"Numeric columns: {numeric_cols}")
        logger.info(f"Categorical columns: {categorical_cols}")
        logger.info(f"Datetime columns: {datetime_cols}")
        
        fig = None
        
        if chart_type == "line":
            if datetime_cols and numeric_cols:
                x_col = datetime_cols[0]
                y_col = numeric_cols[0]
                fig = px.line(df, x=x_col, y=y_col, 
                             title=f"Line Chart: {y_col} over {x_col}")
            elif len(df.columns) >= 2:
                x_col = df.columns[0]
                y_col = numeric_cols[0] if numeric_cols else df.columns[1]
                fig = px.line(df, x=x_col, y=y_col, 
                             title=f"Line Chart: {y_col} vs {x_col}")
                
        elif chart_type == "pie":
            if len(df.columns) >= 2:
                labels_col = categorical_cols[0] if categorical_cols else df.columns[0]
                values_col = numeric_cols[0] if numeric_cols else df.columns[1]
                fig = px.pie(df, names=labels_col, values=values_col, 
                           title=f"Pie Chart: {values_col} by {labels_col}")
                
        elif chart_type == "bar":
            if len(df.columns) >= 2:
                x_col = categorical_cols[0] if categorical_cols else df.columns[0]
                y_col = numeric_cols[0] if numeric_cols else df.columns[1]
                fig = px.bar(df, x=x_col, y=y_col, 
                           title=f"Bar Chart: {y_col} by {x_col}")
                
        elif chart_type == "grouped_bar":
            # Multi-column bar chart
            if len(categorical_cols) >= 2 and len(numeric_cols) >= 1:
                x_col = categorical_cols[0]
                color_col = categorical_cols[1]
                y_col = numeric_cols[0]
                fig = px.bar(df, x=x_col, y=y_col, color=color_col,
                           title=f"Grouped Bar Chart: {y_col} by {x_col} and {color_col}",
                           barmode='group')
            elif len(categorical_cols) >= 1 and len(numeric_cols) >= 2:
                # Multiple numeric columns
                x_col = categorical_cols[0]
                y_cols = numeric_cols[:3]  # Take up to 3 numeric columns
                df_melted = df.melt(id_vars=[x_col], value_vars=y_cols, 
                                  var_name='Metric', value_name='Value')
                fig = px.bar(df_melted, x=x_col, y='Value', color='Metric',
                           title=f"Multi-Column Bar Chart: {', '.join(y_cols)} by {x_col}",
                           barmode='group')
            else:
                # Fallback to regular bar chart
                x_col = df.columns[0]
                y_col = numeric_cols[0] if numeric_cols else df.columns[1]
                fig = px.bar(df, x=x_col, y=y_col, 
                           title=f"Bar Chart: {y_col} by {x_col}")
                
        elif chart_type == "scatter":
            if len(numeric_cols) >= 2:
                x_col = numeric_cols[0]
                y_col = numeric_cols[1]
                color_col = categorical_cols[0] if categorical_cols else None
                fig = px.scatter(df, x=x_col, y=y_col, color=color_col,
                               title=f"Scatter Plot: {y_col} vs {x_col}")
                
        elif chart_type == "histogram":
            if numeric_cols:
                col = numeric_cols[0]
                fig = px.histogram(df, x=col, title=f"Histogram of {col}")
        
        if fig:
            fig.update_layout(
                showlegend=True,
                height=500,
                margin=dict(l=50, r=50, t=50, b=50),
                xaxis_title_font_size=12,
                yaxis_title_font_size=12
            )
            st.plotly_chart(fig, use_container_width=True)
            logger.info("Visualization created successfully")
            return True
        else:
            error_msg = f"Could not create {chart_type} chart with available data. Available columns: {list(df.columns)}"
            logger.error(error_msg)
            st.error(error_msg)
            st.write("Data sample:")
            st.dataframe(df.head())
            return False
            
    except Exception as e:
        error_msg = f"Error creating visualization: {str(e)}"
        logger.error(error_msg)
        st.error(error_msg)
        st.write("Data sample:")
        st.dataframe(df.head() if not df.empty else pd.DataFrame())
        return False

def get_enhanced_response(user_query: str, db: SQLDatabase, chat_history: list):
    """Enhanced response function with better error handling and logging"""
    try:
        logger.info(f"Processing user query: {user_query}")
        
        # Step 1: Detect if visualization is needed
        needs_viz, chart_type = detect_visualization_request(user_query)
        
        # Step 2: Generate SQL query
        sql_chain = get_sql_chain(db)
        sql_query = sql_chain.invoke({
            "question": user_query,
            "chat_history": chat_history
        })
        
        logger.info(f"Generated SQL query: {sql_query}")
        
        # Step 3: Execute SQL query
        try:
            sql_response = db.run(sql_query)
            logger.info(f"SQL query executed successfully. Response length: {len(str(sql_response))}")
        except Exception as e:
            error_msg = f"Error executing SQL query: {str(e)}\nSQL Query: {sql_query}"
            logger.error(error_msg)
            return error_msg
        
        # Step 4: Create visualization if needed
        chart_created = False
        if needs_viz:
            try:
                # Get data as DataFrame
                df = pd.read_sql(sql_query, db._engine)
                logger.info(f"DataFrame created with shape: {df.shape}")
                chart_created = create_visualization(df, chart_type, user_query)
            except Exception as e:
                error_msg = f"Error creating chart: {str(e)}"
                logger.error(error_msg)
                st.error(error_msg)
        
        # Step 5: Generate natural language response
        template = """
        You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
        Based on the table schema below, question, sql query, and sql response, write a natural language response.
        
        <SCHEMA>{schema}</SCHEMA>

        Conversation History: {chat_history}
        SQL Query: <SQL>{query}</SQL>
        User question: {question}
        SQL Response: {response}
        
        {visualization_note}
        
        Provide a clear, informative response that explains the data findings.
        """
        
        visualization_note = ""
        if needs_viz and chart_created:
            visualization_note = "Note: I've created the requested visualization above showing the data in chart format."
        elif needs_viz and not chart_created:
            visualization_note = "Note: I attempted to create a visualization but encountered issues with the data format. The raw data is shown above."
        
        prompt = ChatPromptTemplate.from_template(template)
        llm = ChatGroq(model="llama3-8b-8192", temperature=0)
        
        chain = (
            RunnablePassthrough.assign(
                schema=lambda _: db.get_table_info(),
            )
            | prompt
            | llm
            | StrOutputParser()
        )
        
        response = chain.invoke({
            "question": user_query,
            "query": sql_query,
            "response": sql_response,
            "chat_history": chat_history,
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
        AIMessage(content="Hello! I'm Althinect Intelligence Bot. Ask me anything about your database and I can create various visualizations including grouped bar charts! 📊"),
    ]

load_dotenv()

st.set_page_config(page_title="Althinect Intelligence Bot", page_icon=":bar_chart:")

st.title("Althinect Intelligence Bot")

# Add some helpful information
with st.expander("📊 Visualization Options"):
    st.markdown("""
    **Available Chart Types:**
    - **Bar Charts**: "Show production by machine"
    - **Grouped Bar Charts**: "Show production by machine and shift" or "Compare multiple metrics"
    - **Line Charts**: "Plot hourly production trend"
    - **Pie Charts**: "Show distribution of production by machine"
    - **Scatter Plots**: "Show relationship between variables"
    - **Histograms**: "Show distribution of values"
    
    **Example Queries:**
    - "Show me production by machine and shift with grouped bar chart"
    - "Plot the hourly production trend over the last 24 hours"
    - "Compare production output, efficiency, and downtime by machine"
    """)

with st.sidebar:
    st.subheader("⚙️ Database Settings")
    st.write("Connect to your MySQL database to start chatting and creating visualizations.")
    
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
    
    st.divider()
    
    # Add logging status
    st.subheader("📝 Logging")
    st.info("Logs are being saved to 'chatbot.log' and displayed in VS Code terminal")

# Chat interface
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("user", avatar="👤"):
            st.markdown(message.content)

user_query = st.chat_input("💬 Ask me about your data or request a visualization...")
if user_query is not None and user_query.strip() != "":
    logger.info(f"User query received: {user_query}")
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_query)
        
    with st.chat_message("assistant", avatar="🤖"):
        if "db" in st.session_state:
            with st.spinner("🔄 Processing your request..."):
                response = get_enhanced_response(user_query, st.session_state.db, st.session_state.chat_history)
                st.markdown(response)
        else:
            response = "⚠️ Please connect to the database first using the sidebar."
            st.markdown(response)
            logger.warning("User attempted to query without database connection")
        
    st.session_state.chat_history.append(AIMessage(content=response))
    logger.info("Conversation turn completed")