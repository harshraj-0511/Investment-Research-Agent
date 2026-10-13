import sys
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from tools import retrieve_financial_docs, get_live_market_data

# 1. Initialize local LLM
llm = ChatOllama(model="llama3.2", temperature=0)

# 2. Register tools and bind them directly
tools = [retrieve_financial_docs, get_live_market_data]
tools_by_name = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = """You are an expert Personal Investment Research Agent specialized in corporate finance and equity research.
You assist analysts by combining internal financial reports and live stock market data.

Available Tools:
1. 'retrieve_financial_docs': Use for financial statements, annual reports, balance sheets, revenue, margins, PAT, dividends, or guidance.
2. 'get_live_market_data': Use for current stock prices, day's high/low, and 52-week ranges for any valid ticker (e.g., 'TCS.NS', 'INFY.NS', 'AAPL').

CRITICAL RULES:
- Identify query intent:
  * Static report question -> call 'retrieve_financial_docs'.
  * Real-time quote / market metric -> call 'get_live_market_data'.
  * Blended question -> call both tools logically.
- Grounding & Citations:
  * Always tag every metric explicitly: [Static Document] or [Live Market].
  * Always state explicit currency units (e.g., INR in ₹ Crores, USD, % YoY). Never omit units.
- Multi-turn context:
  * Remember previous queries and answers in the session (e.g. follow-up comparisons).
"""

def chat_session():
    print("=" * 65)
    print("  PERSONAL INVESTMENT RESEARCH AGENT (Local LangChain + Ollama)")
    print("=" * 65)
    print("Type your financial or market query below.")
    print("Type 'exit' or 'quit' to terminate the session.\n")

    history = [SystemMessage(content=SYSTEM_PROMPT)]

    while True:
        try:
            user_input = input("\nUser > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("\nSession ended. Goodbye!")
                break

            history.append(HumanMessage(content=user_input))

            # Tool execution loop
            for _ in range(5):
                ai_msg = llm_with_tools.invoke(history)
                history.append(ai_msg)

                if not ai_msg.tool_calls:
                    print(f"\nAgent > {ai_msg.content}")
                    break

                for tool_call in ai_msg.tool_calls:
                    name = tool_call["name"]
                    args = tool_call["args"]
                    print(f"  [Tool Action] -> {name}({args})")

                    tool_fn = tools_by_name.get(name)
                    output = tool_fn.invoke(args) if tool_fn else f"Error: Tool '{name}' not found."
                    history.append(ToolMessage(content=str(output), tool_call_id=tool_call["id"]))

        except KeyboardInterrupt:
            print("\nSession interrupted. Exiting.")
            sys.exit(0)

if __name__ == "__main__":
    chat_session()