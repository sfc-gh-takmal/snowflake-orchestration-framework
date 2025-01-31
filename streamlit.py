# streamlit_app.py

# Copyright 2024 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import io
import json
import logging
import os
import queue
import re
import sys
import threading
import uuid
import warnings

import requests
import streamlit as st
from dotenv import load_dotenv
from snowflake.snowpark import Session

from agent_gateway import Agent
from agent_gateway.tools import CortexAnalystTool, CortexSearchTool, PythonTool
from agent_gateway.tools.utils import parse_log_message

import yfinance as yf

###############################################################################
# Environment Setup & Page Config
###############################################################################
warnings.filterwarnings("ignore")
load_dotenv()  # optionally load .env from the same directory
st.set_page_config(page_title="Snowflake Cortex Cube")

logging.getLogger("AgentGatewayLogger").setLevel(logging.INFO)

###############################################################################
# Snowflake Connection
###############################################################################
connection_parameters = {
    "account": os.getenv("SNOWFLAKE_ACCOUNT"),
    "user": os.getenv("SNOWFLAKE_USER"),
    "password": os.getenv("SNOWFLAKE_PASSWORD"),
    "role": os.getenv("SNOWFLAKE_ROLE"),
    "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
    "database": os.getenv("SNOWFLAKE_DATABASE"),
    "schema": os.getenv("SNOWFLAKE_SCHEMA"),
}

###############################################################################
# Custom Python Tool: StockTool
###############################################################################
class StockTool:
    """Retrieves current stock info using yfinance."""

    def __init__(self) -> None:
        pass

    def stock_search(self, ticker: str) -> str:
        """Fetch current stock information for the given ticker."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.fast_info

            current_price = info.last_price
            previous_close = info.previous_close
            market_cap = info.market_cap

            # Format market cap with proper string handling
            market_cap_str = "N/A"
            if isinstance(market_cap, (int, float)):
                billions = market_cap / 1_000_000_000
                market_cap_str = f"${billions:.2f}B"

            # Construct Yahoo Finance link
            url = f"https://finance.yahoo.com/quote/{ticker}"
            
            # Build response with explicit string joins
            response_parts = [
                f"Stock information for {ticker}:",
                f"• Current Price: ${current_price:.2f}",
                f"• Previous Close: ${previous_close:.2f}",
                f"• Market Cap: {market_cap_str}",
                f"• Exchange Timezone: {info.timezone}",
                "",
                f"Source: {url}"
            ]
            
            return "\n".join(response_parts)
        except Exception as e:
            return f"Error fetching data for {ticker}: {str(e)}"


###############################################################################
# Streamlit / Session State Initialization
###############################################################################
if "prompt_history" not in st.session_state:
    st.session_state["prompt_history"] = {}

# Create Snowflake session if not already present
if "snowpark" not in st.session_state or st.session_state.snowpark is None:
    st.session_state.snowpark = Session.builder.configs(connection_parameters).create()

    # Tool configs from the notebook
    analyst_config = {
        "semantic_model": "SALES_MODEL.yaml",
        "stage": "ANALYST",
        "service_topic": "Sales metrics",
        "data_description": "Table with sales metrics",
        "snowflake_connection": st.session_state.snowpark,
    }

    search_config = {
        "service_name": "KBA",
        "service_topic": "Knowledge Base Articles on products",
        "data_description": "Documentation on internal documentation",
        "retrieval_columns": ["CHUNK"],
        "snowflake_connection": st.session_state.snowpark,
    }

    stock_config = {
        "tool_description": "searches for current stock market data based on ticker symbol",
        "output_description": "current stock price, previous close, and market cap",
        "python_func": StockTool().stock_search,
    }

    # Initialize tools
    st.session_state.sales_tool = CortexAnalystTool(**analyst_config)
    st.session_state.kba_tool = CortexSearchTool(**search_config)
    st.session_state.stock_tool = PythonTool(**stock_config)

    # Bundle them together
    st.session_state.snowflake_tools = [
        st.session_state.sales_tool,
        st.session_state.kba_tool,
        st.session_state.stock_tool,
    ]

# Create the Agent if not present
if "agent" not in st.session_state:
    st.session_state.agent = Agent(
        snowflake_connection=st.session_state.snowpark,
        tools=st.session_state.snowflake_tools,
        max_retries=3,
    )

###############################################################################
# Prompt Handling Helpers
###############################################################################
def create_prompt(prompt_key: str):
    """
    Extracts user input from st.session_state[prompt_key],
    and adds a new record to `prompt_history` with a `response="waiting"`.
    """
    if prompt_key in st.session_state:
        prompt_record = dict(prompt=st.session_state[prompt_key], response="waiting")
        st.session_state["prompt_history"][str(uuid.uuid4())] = prompt_record


###############################################################################
# Logging Setup for Live Output
###############################################################################
class StreamlitLogHandler(logging.Handler):
    """Custom logging handler that captures logs in memory for Streamlit."""

    def __init__(self):
        super().__init__()
        self.log_buffer = io.StringIO()
        self.ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        self.logs = []

    def emit(self, record):
        msg = self.format(record)
        clean_msg = self.ansi_escape.sub("", msg)
        self.log_buffer.write(clean_msg + "\n")
        self.logs.append(clean_msg)

    def get_logs(self):
        return self.log_buffer.getvalue()

    def get_log_list(self):
        return self.logs

    def clear_logs(self):
        self.logs = []
        self.log_buffer = io.StringIO()


def setup_logging():
    root_logger = logging.getLogger()
    handler = StreamlitLogHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    return handler

# Initialize logging once
if "logging_setup" not in st.session_state:
    st.session_state.logging_setup = setup_logging()


###############################################################################
# Running the Agent (Async) + Display
###############################################################################
def run_acall(prompt, message_queue, agent):
    """
    Synchronously executes agent.acall(prompt),
    capturing the stdout logs to push them into a queue for real-time updates.
    """
    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout

    # Create event loop for async call
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    response = loop.run_until_complete(agent.acall(prompt))
    loop.close()

    # Restore stdout
    sys.stdout = old_stdout

    # Parse logs line-by-line
    output = new_stdout.getvalue()
    lines = output.split("\n")
    for line in lines:
        if line and "Running" in line and "tool" in line:
            tool_selection_string = extract_tool_name(line)
            message_queue.put({"tool_selection": tool_selection_string})
        elif line:
            logging.info(line)  # Also push to normal logging
            message_queue.put(line)

    # Send final response to queue
    message_queue.put({"output": response})


def process_message(prompt_id: str):
    """
    Continuously reads from a queue until the agent's final output is encountered,
    yielding partial logs along the way.
    """
    prompt = st.session_state["prompt_history"][prompt_id].get("prompt")
    message_queue = queue.Queue()
    agent = st.session_state.agent
    
    # Create containers for different parts of the output
    tool_container = st.empty()
    log_container = st.container()
    response_container = st.empty()
    
    log_handler = setup_logging()

    def run_analysis():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        response = loop.run_until_complete(agent.acall(prompt))
        loop.close()
        message_queue.put({"output": response})

    thread = threading.Thread(target=run_acall, args=(prompt, message_queue, agent))
    thread.start()

    current_tool = None

    while True:
        try:
            response = message_queue.get(timeout=0.1)
            if isinstance(response, dict):
                if "tool_selection" in response:
                    current_tool = response["tool_selection"]
                    with tool_container:
                        st.info(f"🔧 Currently using: {current_tool}")
                elif "output" in response:
                    final_response = f"{response['output']}"
                    st.session_state["prompt_history"][prompt_id]["response"] = final_response
                    
                    # Show final tool usage summary
                    if current_tool:
                        with tool_container:
                            st.success(f"✅ Completed using: {current_tool}")
                    
                    # Show the final logs
                    with log_container:
                        st.expander("View Processing Logs", expanded=False).code(
                            log_handler.get_logs()
                        )
                    
                    response_container.markdown(final_response)
                    yield final_response
                    break
            else:
                # Show intermittent logs in the log container
                log_output = parse_log_message(str(response))
                if log_output:
                    with log_container:
                        st.text(log_output)
        except queue.Empty:
            pass
    st.rerun()


def extract_tool_name(statement):
    """
    Helper to parse lines like: "Running <toolName> tool"
    and return "<toolName>" as a clean string.
    """
    start = statement.find("Running") + len("Running")
    end = statement.find("tool")
    return statement[start:end].strip()


###############################################################################
# Sidebar and Control Functions
###############################################################################
def clear_chat_history():
    st.session_state["prompt_history"] = {}
    st.rerun()

def download_chat_history():
    chat_data = []
    for pid, data in st.session_state["prompt_history"].items():
        chat_data.append({
            "id": pid,
            "prompt": data["prompt"],
            "response": data["response"] if data["response"] != "waiting" else ""
        })
    
    # Convert to JSON string
    json_str = json.dumps(chat_data, indent=2)
    
    # Create download button
    st.download_button(
        label="📥 Download",
        data=json_str,
        file_name="chat_history.json",
        mime="application/json"
    )

###############################################################################
# Streamlit Layout / UI
###############################################################################
# Minimal styling to reposition header items
st.markdown(
    """
    <style>
        div[data-testid="stHeader"] > img, 
        div[data-testid="stSidebarCollapsedControl"] > img {
            height: 2rem;
            width: auto;
        }
        div[data-testid="stHeader"], div[data-testid="stHeader"] > *,
        div[data-testid="stSidebarCollapsedControl"], div[data-testid="stSidebarCollapsedControl"] > * {
            display: flex;
            align-items: center;
        }
        /* Center buttons in columns */
        div.stButton > button {
            width: 100%;
        }
    </style>
""",
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.title("Chat Controls")
    
    # Create two columns for the buttons
    col1, col2 = st.columns(2)
    
    # Clear chat button in first column
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            clear_chat_history()
    
    # Download button in second column
    with col2:
        if len(st.session_state["prompt_history"]) > 0:
            download_chat_history()
        else:
            st.button("📥 Download", disabled=True, use_container_width=True)

st.title("Snowflake Multi Tool Agent")
st.caption("A Multi-Tool Agent System")

###############################################################################
# Display the conversation so far
###############################################################################
for pid in st.session_state["prompt_history"]:
    current_prompt = st.session_state["prompt_history"][pid]
    
    # Create a container for each conversation pair
    conv_container = st.container()
    
    with conv_container:
        with st.chat_message("user"):
            st.write(current_prompt.get("prompt"))

        with st.chat_message("assistant", avatar="🤖"):
            if current_prompt.get("response") == "waiting":
                message_generator = process_message(prompt_id=pid)
                
                with st.spinner("Processing your request..."):
                    for partial_response in message_generator:
                        st.markdown(partial_response)
            else:
                st.markdown(current_prompt.get("response"))

###############################################################################
# Chat Input
###############################################################################
st.chat_input(
    "Ask Anything",
    on_submit=create_prompt,
    key="chat_input",
    args=["chat_input"],
)
