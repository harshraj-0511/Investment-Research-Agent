import json
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from tools import retrieve_financial_docs, get_live_market_data

# 1. Initialize local LLM
llm = ChatOllama(model="llama3.2", temperature=0)

# 2. Register tools and bind them directly to the LLM
tools = [retrieve_financial_docs, get_live_market_data]
tools_by_name = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools)

# 3. System prompt enforcing source tagging and explicit units
SYSTEM_PROMPT = """You are an expert Personal Investment Research Agent.
You answer user questions by combining internal financial reports and live stock market data.

You have access to two tools:
1. 'retrieve_financial_docs': Use for historical earnings, annual reports, balance sheet items, dividends, guidance, or quarterly disclosures.
2. 'get_live_market_data': Use for real-time stock prices, 52-week ranges, and intraday highs/lows.

CRITICAL INSTRUCTIONS:
- Identify what the user is asking:
  * If asking about document metrics (e.g., FY24 revenue, Q2 profit) -> call retrieve_financial_docs.
  * If asking about current market price/status -> call get_live_market_data.
  * If asking a blended question -> call BOTH tools in logical sequence.
- Ground every number:
  * You MUST tag each financial figure with its source: [Static Document] or [Live Market].
  * You MUST state units explicitly (e.g., INR in ₹ Crores, USD, % growth). Never output unitless numbers.
- If data is missing or not found in documents/tools, state it clearly.
"""

def query_agent(question: str):
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=question)
    ]
    
    print(f"\nUser Query: {question}\n" + "="*60)
    
    # Run the tool-calling loop (up to 5 turns)
    for step in range(5):
        ai_msg = llm_with_tools.invoke(messages)
        messages.append(ai_msg)
        
        # If the model does not request any more tool calls, we have our final answer
        if not ai_msg.tool_calls:
            return ai_msg.content
        
        # Execute each requested tool call
        for tool_call in ai_msg.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            print(f"[Tool Execution] Calling '{tool_name}' with args: {tool_args}")
            
            selected_tool = tools_by_name.get(tool_name)
            if selected_tool:
                tool_output = selected_tool.invoke(tool_args)
            else:
                tool_output = f"Error: Tool '{tool_name}' not found."
                
            messages.append(ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"]))

    return "Agent reached maximum tool-calling iterations."

if __name__ == "__main__":
    test_query = "What was Bharat Infotech's revenue in Q2 FY25 from the report, and what is the live price of TCS.NS right now?"
    final_answer = query_agent(test_query)
    print("\n--- FINAL AGENT ANSWER ---")
    print(final_answer)