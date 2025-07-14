from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import streamlit as st
import pandas as pd
import plotly.express as px
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_database(user: str, password: str, host: str, port: str, database: str) -> SQLDatabase:
    """Initialize database connection"""
    try:
        db_uri = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
        db = SQLDatabase.from_uri(db_uri)
        logger.info("Database connected successfully")
        return db
    except Exception as e:
        logger.error(f"Database connection failed: {str(e)}")
        raise e

def detect_visualization_type(query: str):
    """Detect if visualization is needed and what type"""
    query_lower = query.lower()
    
    # Check for visualization keywords
    viz_keywords = ['plot', 'chart', 'graph', 'visualize', 'show', 'display', 'bar', 'line', 'pie']
    needs_viz = any(keyword in query_lower for keyword in viz_keywords)
    
    # Determine chart type
    if 'line' in query_lower or 'trend' in query_lower:
        chart_type = "line"
    elif 'pie' in query_lower or 'distribution' in query_lower:
        chart_type = "pie"
    elif 'machine' in query_lower and any(word in query_lower for word in ['all', 'each', 'by']):
        chart_type = "grouped_bar"
    else:
        chart_type = "bar"
    
    return needs_viz, chart_type

def get_sql_chain(db):
    """Create SQL query chain"""
    template = """
    Based on the database schema below, write a SQL query to answer the user's question.
    
    <SCHEMA>{schema}</SCHEMA>
    
    Guidelines:
    1. For production data: Use production_output, device_name, actual_start_time
    2. For time-based queries: Use DATE() and DATE_FORMAT() functions
    3. For multi-machine queries: GROUP BY device_name
    4. Always ORDER BY time or relevant column
    5. Limit results to 100 for performance
    
    Examples:
    - "Show production by machine in April": 
      SELECT device_name, SUM(production_output) as total_output 
      FROM hourly_production 
      WHERE YEAR(actual_start_time) = 2024 AND MONTH(actual_start_time) = 4 
      GROUP BY device_name
    
    - "Daily production trend":
      SELECT DATE(actual_start_time) as date, SUM(production_output) as daily_output
      FROM hourly_production 
      WHERE actual_start_time >= DATE_SUB(NOW(), INTERVAL 30 DAY)
      GROUP BY DATE(actual_start_time) 
      ORDER BY date
    
    Write only the SQL query without any formatting or explanations.
    
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

def create_visualization(df, chart_type, title="Data Visualization"):
    """Create visualization based on chart type"""
    if df.empty:
        st.warning("No data available for visualization")
        return False
    
    try:
        fig = None
        
        if chart_type == "line":
            x_col, y_col = df.columns[0], df.columns[1]
            fig = px.line(df, x=x_col, y=y_col, title=f"Line Chart: {title}")
            
        elif chart_type == "pie":
            labels_col, values_col = df.columns[0], df.columns[1]
            fig = px.pie(df, names=labels_col, values=values_col, title=f"Pie Chart: {title}")
            
        elif chart_type == "grouped_bar":
            # For multi-machine data
            if len(df.columns) >= 3:
                x_col, y_col, color_col = df.columns[0], df.columns[1], df.columns[2]
                fig = px.bar(df, x=x_col, y=y_col, color=color_col, 
                           title=f"Grouped Bar Chart: {title}", barmode='group')
            else:
                x_col, y_col = df.columns[0], df.columns[1]
                fig = px.bar(df, x=x_col, y=y_col, title=f"Bar Chart: {title}")
        else:
            x_col, y_col = df.columns[0], df.columns[1]
            fig = px.bar(df, x=x_col, y=y_col, title=f"Bar Chart: {title}")
        
        if fig:
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            return True
            
    except Exception as e:
        st.error(f"Visualization error: {str(e)}")
        return False

def is_greeting(query: str) -> bool:
    """Check if query is a greeting"""
    greetings = ['hello', 'hi', 'hey', 'good morning', 'help', 'what can you do']
    return any(greeting in query.lower() for greeting in greetings)

def get_casual_response(query: str) -> str:
    """Generate response for greetings"""
    if any(word in query.lower() for word in ['hello', 'hi', 'hey']):
        return "Hello! I'm your database assistant. Ask me about your production data!"
    elif 'help' in query.lower():
        return """I can help you with:
- Production data analysis
- Creating charts and visualizations
- Machine performance comparisons
- Time-based trends

Try asking: "Show production by machine" or "Plot daily production trends\""""
    else:
        return "I'm here to help with your database queries and visualizations!"

def get_response(user_query: str, db: SQLDatabase):
    """Main response function"""
    try:
        # Handle greetings
        if is_greeting(user_query):
            return get_casual_response(user_query)
        
        # Detect visualization needs
        needs_viz, chart_type = detect_visualization_type(user_query)
        
        # Generate and execute SQL
        sql_chain = get_sql_chain(db)
        sql_query = sql_chain.invoke({"question": user_query})
        
        logger.info(f"Generated SQL: {sql_query}")
        sql_response = db.run(sql_query)
        
        if not sql_response or sql_response in ["[]", "()"]:
            return "No data found. Please check your query or date range."
        
        # Create visualization if needed
        if needs_viz:
            try:
                df = pd.read_sql(sql_query, db._engine)
                if not df.empty:
                    create_visualization(df, chart_type, "Production Data")
            except Exception as e:
                st.error(f"Visualization error: {str(e)}")
        
        # Generate natural language response
        template = """
        Based on the SQL query results, provide a clear summary of the findings.
        
        Query: {query}
        Results: {results}
        
        Provide a concise analysis of the data.
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        llm = ChatGroq(model="llama3-8b-8192", temperature=0)
        chain = prompt | llm | StrOutputParser()
        
        response = chain.invoke({
            "query": user_query,
            "results": sql_response
        })
        
        return response
        
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit UI
load_dotenv()
st.set_page_config(page_title="Database Chat Bot", page_icon="🤖")
st.title("Database Chat Bot")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! I'm your database assistant. Ask me about your data!")
    ]

# Sidebar for database connection
with st.sidebar:
    st.subheader("Database Connection")
    
    host = st.text_input("Host", value="localhost")
    port = st.text_input("Port", value="3306")
    user = st.text_input("User", value="root")
    password = st.text_input("Password", type="password", value="chama:1234")
    database = st.text_input("Database", value="Analyzee_machines")
    
    if st.button("Connect"):
        try:
            db = init_database(user, password, host, port, database)
            st.session_state.db = db
            st.success("Connected!")
        except Exception as e:
            st.error(f"Connection failed: {str(e)}")
    
    if "db" in st.session_state:
        st.success("🟢 Connected")
    else:
        st.warning("🔴 Not Connected")

# Chat interface
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)
    else:
        with st.chat_message("user"):
            st.markdown(message.content)

# User input
user_query = st.chat_input("Ask me about your database...")
if user_query:
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    with st.chat_message("user"):
        st.markdown(user_query)
        
    with st.chat_message("assistant"):
        if "db" in st.session_state:
            response = get_response(user_query, st.session_state.db)
        else:
            response = "Please connect to the database first."
        
        st.markdown(response)
    
    st.session_state.chat_history.append(AIMessage(content=response))