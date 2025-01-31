# Snowflake Multi-Tool Agent Framework

A powerful and flexible orchestration framework that combines multiple Snowflake-powered tools into a unified agent system, enabling natural language interactions with various data sources and services.

## 🌟 Features

- **Interactive Streamlit Interface**: Modern web interface for real-time interaction with the agent system
- **Multiple Specialized Tools**:
  - 📊 **Cortex Analyst Tool**: Analyzes sales metrics and performance data
  - 🔍 **Knowledge Base Search Tool**: Searches through internal documentation and articles
  - 📈 **Stock Market Tool**: Retrieves real-time stock market data using yfinance
- **Asynchronous Processing**: Handles multiple requests efficiently with async operations
- **Real-time Logging**: Live feedback on tool selection and processing status
- **Chat History Management**: Save and download conversation history
- **Snowflake Integration**: Seamless connection with Snowflake data warehouse

## 🚀 Getting Started

### Prerequisites

- Python 3.11.10
- Snowflake account with appropriate access
- Required Python packages (install via pip):
  ```
  orchestration-framework@git+https://github.com/Snowflake-Labs/orchestration-framework.git
  streamlit
  snowflake-ml-python
  python-dotenv
  yfinance
  requests
  ```

### Configuration

1. Clone this repository
2. Copy `.env_example` to `.env` and fill in your Snowflake credentials:
   ```env
   SNOWFLAKE_ROLE="<ROLE>"
   SNOWFLAKE_WAREHOUSE="<WAREHOUSE>"
   SNOWFLAKE_USER="<USER>"
   SNOWFLAKE_PASSWORD="<PASSWORD>"  # Remove if using SSO
   SNOWFLAKE_AUTHENTICATOR="externalbrowser"  # For SSO
   SNOWFLAKE_ACCOUNT="<ACCOUNT>"
   SNOWFLAKE_DATABASE="<DATABASE>"
   SNOWFLAKE_SCHEMA="<SCHEMA>"
   ```

### Running the Application

1. Start the Streamlit app:
   ```
   streamlit run streamlit.py
   ```
2. Access the web interface through your browser (typically http://localhost:8501)

## 💡 Usage

The framework provides a chat-like interface where you can:

1. **Query Sales Data**: Ask questions about sales metrics, revenue, marketing spend, etc.
2. **Search Knowledge Base**: Look up information in internal documentation
3. **Check Stock Data**: Get real-time stock market information by ticker symbol

Example queries:
- "What is the total revenue for the toy category?"
- "Show me marketing campaign details for the beauty products"
- "Get the current stock price for SNOW"

## 🛠 Architecture

The framework consists of three main components:

1. **Agent Gateway**: Core orchestration layer that routes requests to appropriate tools
2. **Specialized Tools**:
   - CortexAnalystTool: Handles structured data analysis
   - CortexSearchTool: Manages unstructured data searches
   - PythonTool: Executes custom Python functions (e.g., stock data retrieval)
3. **Streamlit Interface**: Web-based user interface with real-time updates

## 📝 Development

The project includes:
- `streamlit.py`: Main application file with UI and tool integration
- `Notebook.ipynb`: Jupyter notebook for development and testing
- Configuration files (`.env_example`, `.gitignore`, etc.)

## 🔒 Security

- Sensitive credentials are managed through environment variables
- SSO authentication support for Snowflake
- User input validation and sanitization

## 📄 License

Copyright 2024 Snowflake Inc.
Licensed under the Apache License, Version 2.0 
