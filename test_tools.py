from tools import retrieve_financial_docs, get_live_market_data

print("--- Testing Tool 1: retrieve_financial_docs ---")
doc_output = retrieve_financial_docs.invoke({"query": "What is the FY 2023-24 revenue from operations?"})
print(doc_output[:350])
print("\n" + "="*50 + "\n")

print("--- Testing Tool 2: get_live_market_data ---")
# Testing with an active NSE ticker like TCS
market_output = get_live_market_data.invoke({"ticker": "TCS.NS"})
print(market_output)