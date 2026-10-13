from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2", temperature=0)

try:
    print("Sending prompt to Ollama...")
    response = llm.invoke("Explain what a stock ticker symbol is in one sentence.")
    print("\n--- Response from Ollama ---")
    print(response.content)
    print("----------------------------")
    print("STATUS: Step 1 complete! Ollama is connected and working.")
except Exception as e:
    print(f"Error connecting to Ollama: {e}")
    