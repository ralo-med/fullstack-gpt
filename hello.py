import os, json
from openai import OpenAI
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
import yfinance as yf
from ddgs import DDGS

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
def get_ticker(inputs: dict) -> str:
    import re
    TICKER_RE = re.compile(r'\(([A-Z]{1,5})[:\s\)]')
    company_name = inputs['company_name']
    query = f"{company_name} stock ticker site:finance.yahoo.com"
    with DDGS() as ddgs:
        for res in ddgs.text(query, safesearch="off", max_results=10):
            m = TICKER_RE.search(res.get("body", "") + res.get("title", ""))
            if m:
                return m.group(1)
    raise ValueError(f"Ticker not found for {company_name}")

def get_income_statement(inputs: dict) -> str:
    return yf.Ticker(inputs["ticker"]).income_stmt.to_json()

def get_balance_sheet(inputs: dict) -> str:
    return yf.Ticker(inputs["ticker"]).balance_sheet.to_json()

def get_daily_stock_performance(inputs: dict) -> str:
    return yf.Ticker(inputs["ticker"]).history(period="3mo").to_json()

functions_map = {
    "get_ticker": get_ticker,
    "get_income_statement": get_income_statement,
    "get_balance_sheet": get_balance_sheet,
    "get_daily_stock_performance": get_daily_stock_performance,
}

tools = [
    {
        "name": "get_ticker",
        "type": "function",
        "description": "Given a company name, return its ticker symbol.",
        "parameters": {
            "type": "object",
            "properties": {"company_name": {"type": "string"}},
            "required": ["company_name"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_income_statement",
        "type": "function",
        "description": "Return the company's income statement.",
        "parameters": {
            "type": "object",
            "properties": {"ticker": {"type": "string"}},
            "required": ["ticker"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_balance_sheet",
        "type": "function",
        "description": "Return the company's balance sheet.",
        "parameters": {
            "type": "object",
            "properties": {"ticker": {"type": "string"}},
            "required": ["ticker"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_daily_stock_performance",
        "type": "function",
        "description": "Return daily stock prices for the last 3 months.",
        "parameters": {
            "type": "object",
            "properties": {"ticker": {"type": "string"}},
            "required": ["ticker"],
            "additionalProperties": False,
        },
    },
]

def investor_assistant(user_msg: str, history: list | None = None) -> list:
    if history is None:
        history = [{"role": "system", "content": "You are a helpful financial analyst."}]
    history.append({"role": "user", "content": user_msg})
    resp = client.responses.create(
        model="gpt-4.1-nano",
        input=history,
        tools=tools,
    )
    if resp.output and resp.output[0].type == "function_call":
        call = resp.output[0]
        result = functions_map[call.name](json.loads(call.arguments))
        history.extend([
            call,
            {
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": result,
            },
        ])
        resp = client.responses.create(
            model="gpt-4o-mini",
            input=history,
            tools=tools,
        )
    print("\n🗨️  Assistant:", resp.output_text)
    return history

if __name__ == "__main__":
    conv = investor_assistant("I want to know if the Salesforce stock is a good buy")
    conv = investor_assistant("Now I want to know if Cloudflare is a good buy", conv)
