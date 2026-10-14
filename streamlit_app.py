import streamlit as st
from agent import query_agent

st.set_page_config(
    page_title="Investment Research Agent",
    page_icon="📈",
    layout="centered"
)

# Mandatory Rubric Requirement: Visible Disclaimer
st.warning("⚠️ Disclaimer: This tool is for informational and educational purposes only. This is not personalized financial advice.")

st.title("📈 Personal Investment Research Agent")
st.caption("Powered by LangChain, Ollama (Llama 3.2), ChromaDB & yfinance")

# Sidebar with tool info and sample queries
with st.sidebar:
    st.header("📌 System Capabilities")
    st.markdown("""
    - **Historical Docs (RAG):** FY24 balance sheets, revenue, PAT (₹ Crores)
    - **Live Market Data:** Real-time stock prices, 52-week highs/lows via yfinance
    """)
    st.markdown("---")
    st.markdown("**Sample Prompts:**")
    st.code("What was Bharat Infotech's FY24 PAT?", language="text")
    st.code("Get the live market price and 52-week range of INFY.NS", language="text")
    st.code("Compare Bharat Infotech FY24 revenue with TCS.NS live price", language="text")

# Initialize message history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I can answer questions by cross-referencing internal company reports and fetching real-time market data. What would you like to research?"}
    ]

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User query input
if prompt := st.chat_input("Enter your investment research query..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Executing agent reasoning and tool lookups..."):
            try:
                response = query_agent(prompt)
                if isinstance(response, dict):
                    output_text = response.get("output", str(response))
                else:
                    output_text = str(response)
            except Exception as e:
                output_text = f"An error occurred during execution: {str(e)}"
            
            st.markdown(output_text)
            st.session_state.messages.append({"role": "assistant", "content": output_text})