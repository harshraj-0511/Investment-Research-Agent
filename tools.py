import yfinance as yf
from langchain_core.tools import tool
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

CHROMA_PATH = "./chroma_db"

# Connect to the persisted Chroma vector store
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_store = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

@tool
def retrieve_financial_docs(query: str) -> str:
    """
    Search and retrieve historical financial records, annual reports, balance sheets,
    and quarterly disclosures from stored static documents.
    Input should be a targeted financial query (e.g., 'FY24 revenue', 'Q2 EBITDA', 'dividends').
    Returns document excerpts containing financial figures, metrics, and explicit units
    such as Indian Rupees (INR in ₹ Crores) or US Dollars (USD in Millions).
    """
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant internal financial documents found for the given query."
    
    formatted_chunks = []
    for i, doc in enumerate(docs, 1):
        formatted_chunks.append(f"[Document Excerpt {i}]:\n{doc.page_content.strip()}")
    
    return "\n\n---\n\n".join(formatted_chunks)

@tool
def get_live_market_data(ticker: str) -> str:
    """
    Fetch real-time stock market data including current price, day's high/low,
    fifty-two week range, and trade currency for a public company ticker symbol.
    Examples of tickers:
      - Indian equities (NSE/BSE): 'TCS.NS', 'INFY.NS', 'RELIANCE.NS', 'TATAMOTORS.NS'
      - US equities: 'AAPL', 'MSFT', 'GOOGL'
    Always returns values alongside explicit currency units (e.g., INR, USD).
    """
    try:
        cleaned_ticker = ticker.strip().upper()
        stock = yf.Ticker(cleaned_ticker)
        fast_info = stock.fast_info
        
        currency = getattr(fast_info, "currency", None) or ("INR" if ".NS" in cleaned_ticker or ".BO" in cleaned_ticker else "USD")
        last_price = getattr(fast_info, "last_price", None)
        day_high = getattr(fast_info, "day_high", None)
        day_low = getattr(fast_info, "day_low", None)
        year_high = getattr(fast_info, "year_high", None)
        year_low = getattr(fast_info, "year_low", None)

        if last_price is None:
            hist = stock.history(period="1d")
            if not hist.empty:
                last_price = float(hist["Close"].iloc[-1])
                day_high = float(hist["High"].iloc[-1])
                day_low = float(hist["Low"].iloc[-1])
            else:
                return f"Could not find live pricing data for ticker symbol '{cleaned_ticker}'."

        output = (
            f"Ticker: {cleaned_ticker}\n"
            f"Live Market Price: {last_price:.2f} {currency}\n"
            f"Day High: {day_high:.2f} {currency}\n"
            f"Day Low: {day_low:.2f} {currency}\n"
            f"52-Week High: {year_high:.2f} {currency if year_high else ''}\n"
            f"52-Week Low: {year_low:.2f} {currency if year_low else ''}"
        )
        return output
    except Exception as e:
        return f"Error retrieving market data for ticker '{ticker}': {str(e)}"