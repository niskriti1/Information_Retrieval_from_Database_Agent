import re
import os
from dotenv import load_dotenv
import sqlite3
from typing import Annotated, List, Literal, Dict, TypedDict, Any, Optional
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.tools import Tool
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from langchain_groq.chat_models import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableWithFallbacks, RunnableLambda
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages, AnyMessage
from langgraph.errors import GraphRecursionError
from pydantic import BaseModel
from sqlalchemy.engine import URL

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")


import logging
import os

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname("Logs")), "logs")
os.makedirs(LOG_DIR, exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    """
    Create and return a logger with the given name.
    Supports separate loggers and avoids duplicate handlers.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        log_file = os.path.join(LOG_DIR, f"{name}.log")
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger  # Avoid adding handlers multiple times


def get_tables_list(connection_string):
    db = SQLDatabase.from_uri(
        URL.create(drivername="sqlite", database="myinformation.db")
    )
    return db.get_table_names()


def get_create_statement(table_name: str, db_path: str = "myinformation.db") -> str:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table_name,)
    )
    result = cursor.fetchone()
    conn.close()

    return result[0] if result else "No CREATE TABLE statement found."


class State(TypedDict):
    """Represents the state of our graph"""

    messages: Annotated[list[AnyMessage], add_messages]


class QueryInput(BaseModel):
    query: str


class SQLAgent:
    """SQL Agent that uses LangGraph to interact with a database."""

    def __init__(
        self,
        connection_string: str,
        api_key: str,
        model_name: str = "llama3-70b-8192",
    ):
        """
        Initialize the SQL Agent with a database connection and LLM configuration.

        Args:
            connection_string: Database connection string
            model_name: OpenAI model to use
            api_key: OpenAI API key (defaults to environment variable)
        """

        self.logger = get_logger(__class__.__name__)
        self.logger.info("Initializing SQL Agent...")

        self.db = SQLDatabase.from_uri(connection_string)
        self.logger.info("Database connection established.")

        self.llm = ChatGroq(model=model_name, api_key=api_key)
        self.logger.info(f"LLM initialized with model: {model_name}")

        # Setup components
        self._setup_tools()
        self.logger.info("Tools setup completed.")

        self._setup_prompts()
        self.logger.info("Prompts setup completed.")

        self._build_graph()
        self.logger.info("Graph built successfully.")

    def _setup_tools(self) -> None:
        """Set up the required tools for database interaction."""

        self.logger.info("Setting up tools for SQL Agent...")
        # Initialize toolkit and extract basic tools
        toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
        self.logger.info("SQLDatabaseToolkit initialized.")

        tools = toolkit.get_tools()
        self.logger.info(f"Extracted {len(tools)} tools from the toolkit.")

        # Extract standard tools
        self.list_tables_tool = next(
            tool for tool in tools if tool.name == "sql_db_list_tables"
        )
        self.logger.info("List tables tool initialized.")

        self.get_schema_tool = next(
            tool for tool in tools if tool.name == "sql_db_schema"
        )
        self.logger.info("Get schema tool initialized.")

        # Define the query execution tool
        def db_query_tool(query: str) -> str:
            result = self.db.run_no_throw(query)
            self.logger.info(f"Executing query: {query}")
            if not result:
                self.logger.error("Query failed.")
                return "Error: Query failed. Please rewrite your query and try again."
            return result

        self.db_query_tool = Tool.from_function(
            name="db_query_tool",
            func=db_query_tool,
            description=(
                "Executes a SQL query and returns the result"
                "If the query is not correct, an error message will be returned. "
                "If an error is returned, rewrite the query, check it, and run again."
            ),
            args_schema=QueryInput,
        )
        self.logger.info("Database query tool initialized.")

        # @tool
        # def db_query_tool(query: str) -> str:
        #     """
        #     Execute a SQL query against the database and get back the result.
        #     If the query is not correct, an error message will be returned.
        #     If an error is returned, rewrite the query, check the query, and try again.
        #     """
        #     result = self.db.run_no_throw(query)
        #     if not result:
        #         return "Error: Query failed. Please rewrite your query and try again."
        #     return result

        # self.db_query_tool = db_query_tool

    def _setup_prompts(self) -> None:
        """Set up the system prompts for query generation and checking."""
        # Query generation prompt
        self.logger.info("Setting up prompts for SQL Agent...")
        self.query_gen_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
**You are an SQL expert with a strong focus on precision and clarity. Your role is to assist with SQL query generation, analysis of query results, and interpretation to answer user questions.** 

Follow these instructions carefully:

1. **Understand the Task**: 
   - Identify the user's question, the relevant table schemas (if provided), the executed query (if any), and the query result or error (if present).

2. **Handle Scenarios**:
   - **If no executed query or query result exists**: Create a syntactically correct SQLITE query to answer the user's question. 
     - Ensure the query is designed for readability and does not make any DML (INSERT, UPDATE, DELETE, DROP) changes to the database.
     - Respond with only the query statement. For example: `SELECT id, name FROM pets;`.
   - **If a query was executed but returned an error**: Respond by repeating the exact error message. For example: `"Error: Pets table doesn't exist."`.
   - **If a query was executed successfully**: Interpret the results and respond with an answer in the format: `Answer: <<question answer>>`.
   - **If the user's question is unclear or the query results do not provide a clear answer**: State that additional clarification or information is needed.

3. **Adhere to Best Practices**:
   - Write queries using proper indentation for clarity.
   - Use aliases, filtering, and ordering where necessary for optimized and comprehensible results.
""",
                ),
                ("placeholder", "{messages}"),
            ]
        )
        self.logger.info("Query generation prompt initialized.")

        # Query checking prompt
        self.logger.info("Setting up query checking prompt...")
        self.query_check_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are a SQL expert with a strong attention to detail.
Double check the SQLite query for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins

If there are any of the above mistakes, rewrite the query. If there are no mistakes, just reproduce the original query.

You will call the appropriate tool to execute the query after running this check.
""",
                ),
                ("placeholder", "{messages}"),
            ]
        )
        self.logger.info("Query checking prompt initialized.")

        # Intent classification prompt
        self.logger.info("Setting up intent classification prompt...")

        self.intent_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an intent classifier. Classify the user's input as:
					- Greeting only (e.g: "Hello", "Hi there","Good Morning","Hello, I am <user>") 
					- Query ("e.g: "What is the weather today?", "Show me the sales report for last month", "How many students are enrolled in the course?")

					Respond only with:
					- "greeting" if the input is a greeting
					- "query" if the input is a query
     
					Examples:
					- Input: "Hello, I am niskriti" → Output: "greeting"
					- Input: "What is the weather today?" → Output: "query"
					- Input: "List the courses in IT department" → Output: "query"
					- Input: "Hi there" → Output: "greeting"
					- Input: "Good Morning" → Output: "greeting"
					""",
                ),
                ("human", "{input}"),
            ]
        )
        self.logger.info("Intent classification prompt initialized.")

    def _create_tool_node_with_fallback(self, tools: list) -> RunnableWithFallbacks:
        """
        Create a tool node with error handling.

        Args:
            tools: List of tools to include in the node

        Returns:
            A tool node with error handling fallbacks
        """
        self.logger.info("Creating tool node with error handling...")

        def handle_tool_error(state: Dict) -> Dict:
            """Handle errors from tool execution."""
            error = state.get("error")
            tool_calls = state("messages")[-1].tool_calls

            self.logger.error(f"Tool execution failed: {error}")

            return {
                "messages": [
                    ToolMessage(
                        content=f"Error {repr(error)}\nplease fix your mistakes.",
                        tool_call_id=tc["id"],
                    )
                    for tc in tool_calls
                ]
            }

        self.logger.info("Tool error handler created.")

        return ToolNode(tools).with_fallbacks(
            [RunnableLambda(handle_tool_error)], exception_key="error"
        )

    def _build_graph(self) -> None:
        """Build the LangGraph workflow."""

        self.logger.info("Building the LangGraph workflow...")
        # Initialize graph
        workflow = StateGraph(State)

        # Define node functions
        def first_tool_call(state: State) -> Dict[str, List[AIMessage]]:
            """Initial node to list database tables."""
            self.logger.info("Executing first tool call to list database tables.")
            return {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "sql_db_list_tables",
                                "args": {},
                                "id": "tool_abcd123",
                            }
                        ],
                    )
                ]
            }

        def model_check_query(state: State) -> Dict[str, List[AIMessage]]:
            """Check if the SQL query is correct before executing it."""
            self.logger.info("Checking the SQL query for correctness.")
            query_check = self.query_check_prompt | self.llm.bind_tools(
                [self.db_query_tool], tool_choice="auto"
            )
            self.logger.info("Executing query check...")

            self.logger.info(
                f"Current messages in model query check: {state['messages'][-1]}"
            )

            return {
                "messages": [query_check.invoke({"messages": [state["messages"][-1]]})]
            }

        def model_get_schema(state: State) -> Dict[str, List[AIMessage]]:
            """Get database schema information for relevant tables chosen by LLM."""
            self.logger.info("Retrieving database schema information.")
            messages = state["messages"]
            schema_selection_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """
You are a database schema analyzer. Based on the user's question and the list of available tables, 
you need to intelligently select ALL tables that might be relevant to answer the question.

IMPORTANT GUIDELINES:
1. For questions about students, grades, courses, etc. - select ALL related tables (students, grades, enrollments, courses, etc.)
2. For questions about orders, customers, products - select ALL related tables (customers, orders, order_items, products, etc.)
3. When in doubt, it's better to select MORE tables rather than fewer
4. Tables are often connected through foreign keys - consider relationships
5. You can pass multiple table names to sql_db_schema tool separated by commas

Examples:
- Question: "What are John's grades?" → Get schemas for: students, grades, enrollments, courses
- Question: "Show customer orders" → Get schemas for: customers, orders, order_items, products

Based on the user's question and available tables, call sql_db_schema with ALL potentially relevant table names.
        """,
                    ),
                    ("placeholder", "{messages}"),
                ]
            )
            schema_chain = schema_selection_prompt | self.llm.bind_tools(
                [self.get_schema_tool]
            )
            response = schema_chain.invoke({"messages": messages})

            self.logger.info(f"Current messages from model get schema: {response}")
            self.logger.info("Invoking get schema tool...")
            return {"messages": [response]}

        query_gen = self.query_gen_prompt | self.llm

        def query_gen_node(state: State) -> Dict[str, List[AIMessage]]:
            """Generate SQL query based on user question and context."""
            self.logger.info("Generating SQL query from user question.")
            message = query_gen.invoke(state)
            self.logger.info(f"Generated message in query gen node: {message}")

            return {"messages": [message]}

        # Define edge conditions
        def should_continue(state: State) -> Literal[END, "correct_query", "query_gen"]:
            """Determine next step based on current state."""

            self.logger.info("Determining next step based on current state.")

            messages = state["messages"]
            last_message = messages[-1]
            self.logger.info(
                f"Last message content in should continue: {last_message.content}"
            )

            if last_message.content.startswith("Answer:"):
                self.logger.info("Answer found in last message, ending workflow.")
                return END
            if last_message.content.startswith("Error:"):
                self.logger.info("Error found in last message, generating new query.")
                return "query_gen"
            else:
                self.logger.info("No specific instruction found, correcting query.")
                return "correct_query"

        # Add nodes to graph
        self.logger.info("Adding nodes to the workflow...")

        workflow.add_node("first_tool_call", first_tool_call)
        self.logger.info("First tool call node added.")

        workflow.add_node(
            "list_tables_tool",
            self._create_tool_node_with_fallback([self.list_tables_tool]),
        )
        self.logger.info("List tables tool node added.")

        workflow.add_node(
            "get_schema_tool",
            self._create_tool_node_with_fallback([self.get_schema_tool]),
        ),
        self.logger.info("Get schema tool node added.")

        workflow.add_node("model_get_schema", model_get_schema)
        self.logger.info("Model get schema node added.")

        workflow.add_node("query_gen", query_gen_node)
        self.logger.info("Query generation node added.")

        workflow.add_node("correct_query", model_check_query)
        self.logger.info("Model check query node added.")

        workflow.add_node(
            "execute_query", self._create_tool_node_with_fallback([self.db_query_tool])
        )
        self.logger.info("Execute query node added.")

        # Add edges
        self.logger.info("Adding edges to the workflow...")

        workflow.add_edge(START, "first_tool_call")
        self.logger.info("Start -> first tool call edge added.")

        workflow.add_edge("first_tool_call", "list_tables_tool")
        self.logger.info("First tool call -> list tables tool edge added.")

        workflow.add_edge("list_tables_tool", "model_get_schema")
        self.logger.info("List tables tool -> model get schema edge added.")

        workflow.add_edge("model_get_schema", "get_schema_tool")
        self.logger.info("model get schema -> get schema tool edge added.")

        workflow.add_edge("get_schema_tool", "query_gen")
        self.logger.info("Get schema tool -> query generation edge added.")

        workflow.add_conditional_edges(
            "query_gen",
            should_continue,
        )
        self.logger.info("Conditional edges for query generation added.")

        workflow.add_edge("correct_query", "execute_query")
        self.logger.info("Correct query -> execute query edge added.")

        workflow.add_edge("execute_query", "query_gen")
        self.logger.info("Execute query -> query generation edge added.")

        # Compile the workflow into a runnable
        self.logger.info("Compiling the workflow into a runnable application...")
        self.app = workflow.compile()
        self.logger.info("Workflow compiled successfully.")

    def display(self):
        from IPython.display import display, Image

        display(Image(self.app.get_graph().draw_mermaid_png()))

    def query(self, question: str) -> Dict[str, Any]:
        """
        Execute a query against the database using the agent.

        Args:
            question: Natural language question to answer using the database

        Returns:
            Dictionary containing the SQL query and answer
        """
        self.logger.info(f"Received query: {question}")

        try:
            self.logger.info("Classifying intent of the question...")
            intent = self.intent_prompt | self.llm
            intent_response = intent.invoke({"input": question})
            intent = intent_response.content.strip().lower()
            self.logger.info(f"Intent classification result: '{intent}'")
            if intent == "greeting":
                self.logger.info("Greeting detected. Responding directly.")
                return {
                    "sql_query": None,
                    "answer": "Hello! How can I assist you today?",
                }
        except Exception as e:
            self.logger.warning(f"Failed to classify intent: {e}. Proceeding as query.")

        try:
            messages = self.app.invoke({"messages": [HumanMessage(content=question)]})

            final_sql_query = self._extract_final_sql_query(messages)
            self.logger.info(f"Final SQL query extracted: {final_sql_query}")

            last_message = messages["messages"][-1]
            content = last_message.content
            self.logger.info(f"Last message content: {content}")

            result = {"sql_query": final_sql_query, "answer": None}

            if content and content.startswith("Answer:"):
                result["answer"] = content.split("Answer:", 1)[1].strip()
                self.logger.info(f"Result extracted: {result}")

            return result

        except GraphRecursionError:
            self.logger.error(
                "Graph recursion error occurred. Unable to process the query."
            )

            return {
                "sql_query": None,
                "answer": "Unable to process the query with the current context. Please review the input and try again.",
            }

    def _extract_final_sql_query(self, messages: Dict) -> Optional[str]:
        """
        Extract the final SQL query from the message history.

        Args:
            messages: Message history from the graph execution

        Returns:
            The final SQL query or None if not found
        """
        self.logger.info("Extracting final SQL query from messages...")

        for msg in reversed(messages.get("messages", [])):
            self.logger.info(f"Messages for final sql: {msg}")
            if hasattr(msg, "content") and msg.content:
                content = msg.content
                if "SELECT" in content.upper():
                    query = re.search(r"SELECT.*?;", content, re.IGNORECASE | re.DOTALL)
                    if query:
                        self.logger.info("SQL query found in message content.")
                        sql_query = query.group(0).strip()
                        self.logger.info(f"SQL query found: {sql_query}")
                        return sql_query
        return None
