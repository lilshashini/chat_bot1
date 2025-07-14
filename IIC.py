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

def init_database(user: str, password: str, host: str, port: str, database: str) -> SQLDatabase:
    db_uri = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
    return SQLDatabase.from_uri(db_uri)

def detect_visualization_request(user_query: str):
    """Detect if user wants a visualization and what type"""
    user_query_lower = user_query.lower()
    
    # Keywords that indicate visualization request
    viz_keywords = [
        'plot', 'chart', 'graph', 'visualize', 'show', 'display',
        'bar chart', 'line chart', 'pie chart', 'histogram', 'scatter plot',
        'bar', 'line', 'pie', 'trend', 'comparison'
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
    
    return needs_viz, chart_type

def get_sql_chain(db):
    template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.
    
    <SCHEMA>{schema}</SCHEMA>
    
    Conversation History: {chat_history}
    
    Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.
    
    IMPORTANT SQL RULES:
    1. When using aggregate functions (MAX, MIN, SUM, COUNT, AVG), always include GROUP BY for non-aggregated columns
    2. If you want the single record with the maximum value, use ORDER BY with LIMIT instead of MAX without GROUP BY
    3. For "highest", "maximum", "top" queries, consider using ORDER BY DESC LIMIT 1
    4. For "lowest", "minimum", "bottom" queries, consider using ORDER BY ASC LIMIT 1
    5. Always follow MySQL's sql_mode=only_full_group_by requirements
    
    For visualization requests, ensure the query returns data suitable for charting:
    - For line charts: include time-based columns and numeric values
    - For pie charts: include category names and corresponding values
    - For bar charts: include categories and numeric values
    - Limit results to reasonable amounts (e.g., LIMIT 50 for large datasets)
    
    Examples:
    Question: Which machine produced the most output in the last hour?
    SQL Query: SELECT device_name, SUM(production_output) AS total_output FROM hourly_production WHERE actual_start_time >= NOW() - INTERVAL 1 HOUR GROUP BY device_name ORDER BY total_output DESC LIMIT 1
    
    Question: What is the highest efficiency among all devices?
    SQL Query: SELECT device_name, efficiency FROM daily_utilization_1 ORDER BY efficiency DESC LIMIT 1
    
    Question: Show efficiency by device (for grouping/comparison)
    SQL Query: SELECT device_name, MAX(efficiency) AS highest_efficiency FROM daily_utilization_1 GROUP BY device_name ORDER BY highest_efficiency DESC
    
    Question: Which device has the highest utilization?
    SQL Query: SELECT device_name, efficiency FROM daily_utilization_1 ORDER BY efficiency DESC LIMIT 1
    
    Question: Plot the hourly production with line graph
    SQL Query: SELECT DATE_FORMAT(actual_start_time, '%Y-%m-%d %H:00:00') AS hour, SUM(production_output) AS total_output FROM hourly_production WHERE actual_start_time >= NOW() - INTERVAL 24 HOUR GROUP BY DATE_FORMAT(actual_start_time, '%Y-%m-%d %H:00:00') ORDER BY hour
    
    Question: Give each machine production with pie chart
    SQL Query: SELECT device_name, SUM(production_output) AS total_output FROM hourly_production GROUP BY device_name ORDER BY total_output DESC LIMIT 10
    
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

def fix_sql_query(sql_query, error_message):
    """
    Attempt to fix common SQL errors, particularly GROUP BY issues
    """
    if "only_full_group_by" in str(error_message) or "incompatible with sql_mode" in str(error_message):
        # Check if query has aggregate functions without proper GROUP BY
        import re
        
        # Find aggregate functions
        agg_pattern = r'\b(MAX|MIN|SUM|COUNT|AVG)\s*\('
        has_aggregates = re.search(agg_pattern, sql_query, re.IGNORECASE)
        
        # Check if there's already a GROUP BY
        has_group_by = re.search(r'\bGROUP\s+BY\b', sql_query, re.IGNORECASE)
        
        if has_aggregates and not has_group_by:
            # Try to fix by converting to ORDER BY + LIMIT approach
            
            if "MAX(" in sql_query.upper():
                # Convert MAX query to ORDER BY DESC LIMIT 1
                # Extract the column being MAX'd
                max_match = re.search(r'MAX\(([^)]+)\)', sql_query, re.IGNORECASE)
                if max_match:
                    max_column = max_match.group(1)
                    # Remove the MAX function and alias
                    fixed_query = re.sub(
                        r'MAX\([^)]+\)\s+AS\s+\w+',
                        max_column,
                        sql_query,
                        flags=re.IGNORECASE
                    )
                    # Add ORDER BY and LIMIT
                    if not re.search(r'\bORDER\s+BY\b', fixed_query, re.IGNORECASE):
                        fixed_query = fixed_query.rstrip(';') + f' ORDER BY {max_column} DESC LIMIT 1;'
                    return fixed_query
            
            elif "MIN(" in sql_query.upper():
                # Similar fix for MIN
                min_match = re.search(r'MIN\(([^)]+)\)', sql_query, re.IGNORECASE)
                if min_match:
                    min_column = min_match.group(1)
                    fixed_query = re.sub(
                        r'MIN\([^)]+\)\s+AS\s+\w+',
                        min_column,
                        sql_query,
                        flags=re.IGNORECASE
                    )
                    if not re.search(r'\bORDER\s+BY\b', fixed_query, re.IGNORECASE):
                        fixed_query = fixed_query.rstrip(';') + f' ORDER BY {min_column} ASC LIMIT 1;'
                    return fixed_query
    
    return None

def create_visualization(df, chart_type, user_query):
    """Create visualization based on dataframe and chart type"""
    try:
        if df.empty:
            st.warning("No data available for visualization")
            return None
        
        # Get column information
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64', 'datetime']).columns.tolist()
        
        # Handle datetime strings
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try to parse as datetime
                try:
                    pd.to_datetime(df[col].iloc[0])
                    df[col] = pd.to_datetime(df[col])
                    datetime_cols.append(col)
                except:
                    pass
        
        fig = None
        
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
        else:
            st.error(f"Could not create {chart_type} chart with available data")
            return False
            
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")
        return False

def get_enhanced_response(user_query: str, db: SQLDatabase, chat_history: list):
    """Enhanced response function with visualization capabilities and error handling"""
    try:
        # Step 1: Detect if visualization is needed
        needs_viz, chart_type = detect_visualization_request(user_query)
        
        # Step 2: Generate SQL query
        sql_chain = get_sql_chain(db)
        sql_query = sql_chain.invoke({
            "question": user_query,
            "chat_history": chat_history
        })
        
        # Step 3: Execute SQL query with error handling
        try:
            sql_response = db.run(sql_query)
        except Exception as e:
            # Try to fix the SQL query if it's a GROUP BY error
            fixed_query = fix_sql_query(sql_query, str(e))
            if fixed_query:
                try:
                    sql_response = db.run(fixed_query)
                    sql_query = fixed_query  # Use the fixed query for further processing
                    st.info(f"✅ Fixed SQL query automatically: {fixed_query}")
                except Exception as e2:
                    return f"Error executing SQL query: {str(e)}\nOriginal SQL: {sql_query}\nFix attempted but failed: {str(e2)}"
            else:
                return f"Error executing SQL query: {str(e)}\nSQL Query: {sql_query}"
        
        # Step 4: Create visualization if needed
        chart_created = False
        if needs_viz:
            try:
                # Get data as DataFrame
                df = pd.read_sql(sql_query, db._engine)
                chart_created = create_visualization(df, chart_type, user_query)
            except Exception as e:
                st.error(f"Error creating chart: {str(e)}")
        
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
        """
        
        visualization_note = ""
        if needs_viz and chart_created:
            visualization_note = "Note: I've created the requested visualization above showing the data in chart format."
        elif needs_viz and not chart_created:
            visualization_note = "Note: I attempted to create a visualization but encountered issues with the data format."
        
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
        
        return response
        
    except Exception as e:
        return f"An error occurred while processing your request: {str(e)}"

# Streamlit UI
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! I'm Althinect Intelligence Bot. Ask me anything about your database. "),
    ]

load_dotenv()

st.set_page_config(page_title="Althinect Intelligence Bot", page_icon=":bar_chart:")

st.title("Althinect Intelligence Bot")

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
            except Exception as e:
                st.error(f"❌ Connection failed: {str(e)}")
    
    if "db" in st.session_state:
        st.success("🟢 Database Connected")
    else:
        st.warning("🔴 Database Not Connected")
    
    st.divider()
    
    # Add a clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = [
            AIMessage(content="Hello! I'm Industrial Intelligence Bot. Ask me anything about your database. "),
        ]
        st.rerun()

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
        
    st.session_state.chat_history.append(AIMessage(content=response))