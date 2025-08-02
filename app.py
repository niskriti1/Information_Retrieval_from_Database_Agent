import os
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime
from sqlalchemy.engine import URL
from get_sql_data_from_natural_query import (
    SQLAgent,
    get_logger,
    get_tables_list,
    get_create_statement,
)


# Get the api key for LLM model
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")


logger = get_logger(__name__)

st.set_page_config(
    page_title="Information Retrieval from Database",
    page_icon="🛢️",
    layout="centered",
    initial_sidebar_state="expanded",
)
logger.info("Streamlit app started")


# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    logger.info("Initialized chat history.")


connection_string = URL.create(drivername="sqlite", database="myinformation.db")

agent = SQLAgent(connection_string=connection_string, api_key=api_key)

# Sidebar showing tables info
st.sidebar.title("📊 Tables Available")
tables = get_tables_list(connection_string)
for table in tables:
    with st.sidebar.expander(table):
        create_stmt = get_create_statement(table)
        st.code(create_stmt, language="sql")


# CSS for chat bubbles and timestamps
chat_css = """
<style>
.chat-bubble-user {
    background-color: #313d27;
    color: white;
    padding: 0.8rem;
    border-radius: 10px;
    margin-bottom: 0.1rem;
    align-self: flex-end;
    max-width: 80%;
    word-wrap: break-word;
}

.chat-bubble-assistant {
    background-color: #1f1111;
    color: white;
    padding: 0.8rem;
    border-radius: 10px;
    margin-bottom: 0.1rem;
    align-self: flex-start;
    border: 1px solid #eee;
    max-width: 80%;
    word-wrap: break-word;
}


.chat-message {
    display: flex;
    flex-direction: column;
    margin-bottom: 0.8rem;
}

.timestamp {
    font-size: 0.7rem;
    color: #999;
    margin-left: 0.5rem;
}
</style>
"""
st.markdown(chat_css, unsafe_allow_html=True)


st.title("💬 Chat with database")

# Display chat messages with timestamps
with st.container():
    for pair in st.session_state.chat_history:
        user_html = f"""
        <div class="chat-message">
            <div class="chat-bubble-user">{pair["user"]}</div>
            <div class="timestamp" style="text-align: right;">{pair["user_time"]}</div>
        </div>
        """
        answer = pair["agent"].get("answer", "No answer available.")
        sql = pair["agent"].get("sql_query", "")

        agent_html = f"""
        <div class="chat-message">
            <div class="chat-bubble-assistant">"Answer: {answer}\nSQL: {sql}</div>
            <div class="timestamp" style="text-align: left;">{pair["agent_time"]}</div>
        </div>
        """

        st.markdown(user_html, unsafe_allow_html=True)
        st.markdown(agent_html, unsafe_allow_html=True)

# Input form
with st.form("chat_form", clear_on_submit=True):
    question = st.text_input(
        "Type your message:", placeholder="e.g. Give me details about student"
    )
    submitted = st.form_submit_button("Send")

if submitted and question.strip():
    with st.spinner("Agents are thinking..."):
        agent_reply = agent.query(question=question)

    # Current timestamp
    now = datetime.now().strftime("%I:%M %p")

    # Append chat with timestamps
    st.session_state.chat_history.append(
        {"user": question, "agent": agent_reply, "user_time": now, "agent_time": now}
    )

    st.rerun()
