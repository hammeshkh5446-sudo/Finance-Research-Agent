"""
Finance Research Assistant (AI Agent)
======================================
An AI agent that answers finance/investing questions by autonomously
deciding which tools to call -- real-time stock prices, historical
performance data, and a calculator -- then reasoning over the results
to produce a final answer.

This is a genuine "agent" (not a simple Q&A bot): the LLM is given a set
of tools and decides for itself, step by step, which ones to call and in
what order, based on the user's question. This is called "tool use" or
"function calling", and looping this process until the model has enough
information to answer is what makes it an "agent" rather than a single
prompt-response system.

Run with:
    streamlit run app.py
"""

import json
import re

import streamlit as st
import streamlit.components.v1 as components
import cohere
import yfinance as yf


# =============================================================================
# CONFIG
# =============================================================================

MODEL = "command-a-plus-05-2026"
MAX_TOOL_STEPS = 10  # safety limit so the agent can't loop forever

st.set_page_config(
    page_title="Finance Research Agent",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# STYLING -- deep burgundy + gold "wealth management" theme
# =============================================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #faf8f5;
    }
    .block-container {
        max-width: 1300px;
        padding-left: 2.5rem;
        padding-right: 2.5rem;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }
    .stApp, .stApp p, .stApp span, .stApp label, .stApp li {
        color: #2b2320;
    }
    h1, h2, h3 { color: #1f1815 !important; }

    /* Sidebar (drawer) -- deep burgundy with gold accents */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #3f0d1a 0%, #1f1815 100%);
        border-right: 1px solid rgba(212, 175, 55, 0.2);
    }
    section[data-testid="stSidebar"] * {
        color: #f3e9d8 !important;
    }
    section[data-testid="stSidebar"] h1 {
        border: 1px solid rgba(212, 175, 55, 0.45) !important;
        background: rgba(212, 175, 55, 0.10);
        border-radius: 12px;
        padding: 12px 14px;
        margin-bottom: 0.6rem;
    }
    section[data-testid="stSidebar"] h3 {
        display: inline-block;
        border: 1px solid rgba(212, 175, 55, 0.5) !important;
        background: rgba(212, 175, 55, 0.12);
        color: #e8c766 !important;
        padding: 4px 14px;
        border-radius: 999px;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.4px;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }

    /* Bordered cards */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #ffffff !important;
        border-radius: 16px !important;
        border: 1px solid #e8ddd0 !important;
        border-top: 3px solid #8c1f3b !important;
        box-shadow: 0 3px 16px rgba(31, 24, 21, 0.06);
        padding: 1.1rem 1.3rem;
        margin-bottom: 0.6rem;
    }

    /* Inputs */
    div[data-baseweb="input"] {
        border-radius: 10px !important;
        border: 1px solid #ddd0c0 !important;
        background: #ffffff !important;
    }
    div[data-baseweb="input"] input {
        color: #1f1815 !important;
    }
    div[data-baseweb="input"]:focus-within {
        border-color: #8c1f3b !important;
        box-shadow: 0 0 0 3px rgba(140, 31, 59, 0.15);
    }

    /* Buttons */
    .stButton > button, .stFormSubmitButton > button {
        border-radius: 10px;
        font-weight: 700;
        background: #8c1f3b;
        color: #e8c766 !important;
        border: 1px solid rgba(212, 175, 55, 0.4);
    }
    .stButton > button *, .stFormSubmitButton > button * {
        color: #e8c766 !important;
    }
    .stButton > button:hover, .stFormSubmitButton > button:hover {
        background: #6e1830;
        border-color: #d4af37;
    }

    /* Answer card */
    .answer-card {
        background: #fffaf0;
        border: 1px solid #ecd9a8;
        border-left: 5px solid #d4af37;
        border-radius: 16px;
        padding: 1.2rem 1.4rem;
        box-shadow: 0 4px 18px rgba(212, 175, 55, 0.10);
        margin-top: 0.6rem;
    }
    .answer-card, .answer-card * {
        color: #241d1a !important;
    }
    .answer-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 3px 12px;
        border-radius: 999px;
        background: #8c1f3b;
        color: #e8c766 !important;
        border: 1px solid rgba(212, 175, 55, 0.4);
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.6rem;
    }

    /* Sidebar collapse/expand toggle arrow -- direct fix from DevTools inspection */
    .st-emotion-cache-12bp31y {
        color: #e8c766 !important;
    }

    /* Sidebar collapse/expand toggle arrow */
    [data-testid="stSidebarCollapsedControl"] svg,
    [data-testid="stSidebarCollapseButton"] svg,
    [data-testid="collapsedControl"] svg,
    button[kind="header"] svg {
        color: #e8c766 !important;
        fill: #e8c766 !important;
    }

    /* Expander */
    div[data-testid="stExpander"] {
        background: #ffffff;
        border: 1px solid #e8ddd0 !important;
        border-radius: 12px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# HEADER
# =============================================================================

def display_header():
    components.html(
        """
        <style>
            html, body { margin: 0; padding: 0; background: transparent; }
        </style>
        <div style="
            font-family: 'Source Sans Pro', sans-serif;
            background: linear-gradient(135deg, #3f0d1a 0%, #1f1815 100%);
            border: 1px solid rgba(212, 175, 55, 0.35);
            border-left: 6px solid #d4af37;
            border-radius: 18px;
            padding: 1.5rem 1.7rem;
            box-sizing: border-box;
        ">
            <h1 style="margin: 0 0 0.4rem 0; font-size: 2.1rem; font-weight: 800; color: #e8c766;">
                Finance Research Agent
            </h1>
            <p style="color: #d9c9b0; margin: 0 0 0.9rem 0; font-size: 0.98rem; line-height: 1.5; max-width: 640px;">
                Ask investing questions -- stocks or crypto -- and watch the agent autonomously decide which tools
                to use -- live prices, historical performance, and calculations --
                to work out the answer.
            </p>
            <div style="display:flex; gap:8px; flex-wrap:wrap;">
                <span style="display:inline-block; padding:4px 13px; border-radius:999px; font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:0.4px; border:1px solid rgba(212,175,55,0.5); background:rgba(212,175,55,0.14); color:#e8c766;">Tool Use</span>
                <span style="display:inline-block; padding:4px 13px; border-radius:999px; font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:0.4px; border:1px solid rgba(212,175,55,0.5); background:rgba(212,175,55,0.14); color:#e8c766;">Live Market Data</span>
                <span style="display:inline-block; padding:4px 13px; border-radius:999px; font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:0.4px; border:1px solid rgba(212,175,55,0.5); background:rgba(212,175,55,0.14); color:#e8c766;">Autonomous Reasoning</span>
            </div>
        </div>
        """,
        height=210,
    )


def display_sidebar():
    with st.sidebar:
        st.title("Finance Research Agent")
        st.caption(
            "An AI agent that autonomously calls tools -- live stock and "
            "crypto prices, plus a calculator -- to answer investing questions."
        )

        st.subheader("Model")
        st.write("Cohere Command A Plus")

        st.subheader("Tools")
        st.write("get_stock_price -- live price lookup")
        st.write("get_stock_history -- historical performance")
        st.write("find_dates_in_price_range -- price-date search")
        st.write("get_fundamentals -- revenue, EPS, margins, debt")
        st.write("calculate -- arithmetic")

        st.subheader("How it works")
        st.write("1. You ask a question")
        st.write("2. The agent decides which tool(s) it needs")
        st.write("3. Tools run, results go back to the model")
        st.write("4. It repeats until it has enough info")
        st.write("5. It gives you a final, grounded answer")

        st.subheader("Tech")
        st.write("Python")
        st.write("Streamlit")
        st.write("Cohere API (tool use)")
        st.write("yfinance (live market data)")


# =============================================================================
# API KEY SETUP
# =============================================================================

def get_api_key():
    """Get the Cohere API key from Streamlit secrets, or ask the user for it."""
    try:
        if "COHERE_API_KEY" in st.secrets:
            return st.secrets["COHERE_API_KEY"]
    except Exception:
        pass
    return st.sidebar.text_input("Enter your Cohere API key", type="password")


# =============================================================================
# TOOLS
# =============================================================================

def get_stock_price(symbol: str) -> dict:
    """Get the current price of a stock or cryptocurrency, given its ticker
    symbol (e.g. AAPL, TSLA for stocks; BTC-USD, ETH-USD for crypto). Also
    includes today's highest and lowest price hit so far."""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        if price is None:
            return {"error": f"Could not find a current price for '{symbol}'."}

        day_high, day_low = None, None
        try:
            intraday = ticker.history(period="1d", interval="5m")
            if not intraday.empty:
                day_high = round(float(intraday["High"].max()), 2)
                day_low = round(float(intraday["Low"].min()), 2)
        except Exception:
            pass
        # Fall back to yfinance's own summary fields if intraday lookup failed
        if day_high is None:
            day_high = info.get("dayHigh")
        if day_low is None:
            day_low = info.get("dayLow")

        return {
            "symbol": symbol.upper(),
            "company_name": info.get("longName", symbol.upper()),
            "current_price": round(float(price), 2),
            "todays_high": day_high,
            "todays_low": day_low,
            "currency": info.get("currency", "USD"),
        }
    except Exception as exc:
        return {"error": f"Failed to look up '{symbol}': {exc}"}


def get_stock_history(symbol: str, period: str = "1y") -> dict:
    """Get historical performance for a stock or cryptocurrency over a period.
    Use ticker format like AAPL for stocks or BTC-USD for crypto. Valid periods:
    '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'."""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period)
        if hist.empty:
            return {"error": f"No historical data found for '{symbol}' over '{period}'."}
        start_price = float(hist["Close"].iloc[0])
        end_price = float(hist["Close"].iloc[-1])
        pct_change = ((end_price - start_price) / start_price) * 100
        return {
            "symbol": symbol.upper(),
            "period": period,
            "start_price": round(start_price, 2),
            "end_price": round(end_price, 2),
            "percent_change": round(pct_change, 2),
        }
    except Exception as exc:
        return {"error": f"Failed to get history for '{symbol}': {exc}"}


def find_dates_in_price_range(symbol: str, low_price: float, high_price: float, period: str = "1y") -> dict:
    """Search historical daily prices to find the date(s) when a stock or
    crypto's closing price fell within a given price range. Use this to
    answer questions like 'when did it hit $X' or 'when was it around $Y'."""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period)
        if hist.empty:
            return {"error": f"No historical data found for '{symbol}' over '{period}'."}

        matches = hist[(hist["Close"] >= low_price) & (hist["Close"] <= high_price)]
        if matches.empty:
            return {
                "symbol": symbol.upper(),
                "low_price": low_price,
                "high_price": high_price,
                "period": period,
                "result": "No dates found in that price range within this period. Try a wider period or a wider price range.",
            }

        dates = [str(d.date()) for d in matches.index]
        return {
            "symbol": symbol.upper(),
            "low_price": low_price,
            "high_price": high_price,
            "period": period,
            "first_date_in_range": dates[0],
            "last_date_in_range": dates[-1],
            "number_of_days_in_range": len(dates),
        }
    except Exception as exc:
        return {"error": f"Failed to search history for '{symbol}': {exc}"}


def get_fundamentals(symbol: str) -> dict:
    """Get key fundamental financial metrics for a stock: market cap, EPS,
    profit margin, revenue growth, debt, cash, P/E ratio, and 52-week price
    range. Note: most of these fields aren't available for cryptocurrencies
    since they don't have earnings or balance sheets."""
    if symbol.upper().endswith("-USD") or symbol.upper() in {"BTC", "ETH", "SOL", "DOGE", "XRP", "ADA"}:
        return {
            "symbol": symbol.upper(),
            "note": "Fundamentals (EPS, profit margin, debt, revenue growth) don't apply to "
                    "cryptocurrencies since they don't have earnings or balance sheets. Use "
                    "get_stock_price or get_stock_history for crypto data instead.",
        }
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        profit_margin = info.get("profitMargins")
        revenue_growth = info.get("revenueGrowth")

        return {
            "symbol": symbol.upper(),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "trailing_eps": info.get("trailingEps"),
            "forward_eps": info.get("forwardEps"),
            "profit_margin_pct": round(profit_margin * 100, 2) if profit_margin is not None else None,
            "revenue_growth_pct": round(revenue_growth * 100, 2) if revenue_growth is not None else None,
            "total_debt": info.get("totalDebt"),
            "total_cash": info.get("totalCash"),
            "debt_to_equity": info.get("debtToEquity"),
            "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
            "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
        }
    except Exception as exc:
        return {"error": f"Failed to get fundamentals for '{symbol}': {exc}"}


def calculate(expression: str) -> dict:
    """Safely evaluate a basic arithmetic expression, e.g. '1000 * 1.25'."""
    allowed_chars = set("0123456789+-*/(). ")
    if not set(expression).issubset(allowed_chars):
        return {"error": "Expression contains characters that aren't allowed. Only numbers and + - * / ( ) are permitted."}
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return {"expression": expression, "result": result}
    except Exception as exc:
        return {"error": f"Could not evaluate '{expression}': {exc}"}


FUNCTIONS_MAP = {
    "get_stock_price": get_stock_price,
    "get_stock_history": get_stock_history,
    "find_dates_in_price_range": find_dates_in_price_range,
    "get_fundamentals": get_fundamentals,
    "calculate": calculate,
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get the current live price of a stock or cryptocurrency given its ticker symbol, including today's highest and lowest price so far. Use the plain symbol for stocks (e.g. 'AAPL', 'TSLA') and the '-USD' suffix format for crypto (e.g. 'BTC-USD' for Bitcoin, 'ETH-USD' for Ethereum).",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "The ticker symbol, e.g. 'AAPL' for Apple stock or 'BTC-USD' for Bitcoin.",
                    },
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_history",
            "description": "Get historical price performance over a given period for a stock or cryptocurrency, including percent change. Use the plain symbol for stocks (e.g. 'AAPL') and the '-USD' suffix for crypto (e.g. 'BTC-USD').",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "The ticker symbol, e.g. 'AAPL' for Apple stock or 'BTC-USD' for Bitcoin.",
                    },
                    "period": {
                        "type": "string",
                        "description": "How far back to look. One of: 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max.",
                    },
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_dates_in_price_range",
            "description": "Search historical daily prices to find when a stock or crypto's price fell within a specific price range. Use this to answer 'when did it hit $X' or 'when was it around $Y' style questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "The ticker symbol, e.g. 'AAPL' or 'SOL-USD' for Solana.",
                    },
                    "low_price": {
                        "type": "number",
                        "description": "The lower bound of the price range to search for.",
                    },
                    "high_price": {
                        "type": "number",
                        "description": "The upper bound of the price range to search for.",
                    },
                    "period": {
                        "type": "string",
                        "description": "How far back to search. One of: 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max. Start with a wider period like '1y' or '2y' if unsure.",
                    },
                },
                "required": ["symbol", "low_price", "high_price"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_fundamentals",
            "description": "Get key fundamental financial metrics for a stock: market cap, P/E ratio, EPS, profit margin, revenue growth, debt, cash, and 52-week price range. Not meaningful for cryptocurrencies (no earnings/balance sheet).",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "The stock ticker symbol, e.g. 'AAPL'.",
                    },
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a basic arithmetic expression, e.g. for computing investment growth.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "An arithmetic expression using only numbers and + - * / ( ), e.g. '5000 * 1.42'.",
                    },
                },
                "required": ["expression"],
            },
        },
    },
]


ANALYSIS_MODE_INSTRUCTIONS = {
    "Trader View": (
        "The user has selected Trader View. Focus on price action: recent price "
        "movement, momentum, volatility, and trading range. Prioritize get_stock_history "
        "and find_dates_in_price_range. Fundamentals are secondary in this mode."
    ),
    "Investor View": (
        "The user has selected Investor View. Focus on fundamentals: revenue growth, "
        "profit margins, valuation (P/E ratio), debt levels, and cash position, using "
        "get_fundamentals. Frame the answer around long-term potential, not short-term "
        "price swings."
    ),
    "Beginner View": (
        "The user has selected Beginner View. Explain everything in simple, plain "
        "language and avoid unexplained jargon -- if you use a term like 'P/E ratio' or "
        "'market cap', briefly define it in one clause."
    ),
}


def build_system_message(mode: str) -> str:
    return f"""## Task & Context
You are a finance research assistant. You help users answer questions about stocks,
cryptocurrency, investments, and financial calculations by using the tools available to
you: live prices, historical performance, price-range date search, fundamentals (for
stocks), and a calculator. Use tools whenever the question requires current data or a
calculation -- never guess numbers yourself. For crypto, use the '-USD' ticker suffix
(e.g. BTC-USD for Bitcoin).

Stay strictly within the scope of finance, investing, and markets (including crypto).
If the user asks about something unrelated to finance (e.g. general trivia, coding help,
personal advice outside investing), politely decline and explain that you're a finance
research assistant focused on stocks, crypto, and investing questions.

## Analysis Mode
{ANALYSIS_MODE_INSTRUCTIONS[mode]}

## Style Guide
Answer clearly and concisely. When you use data from a tool, state it plainly
(e.g. "Tesla (TSLA) is currently trading at $248.50."). This is not financial advice --
if asked for investment recommendations, note that you can provide data and analysis
but not personalized financial advice.
"""


# =============================================================================
# AGENT LOOP
# =============================================================================

def extract_text(message_content) -> str:
    """Find the text content item in a message's content list. Newer
    Cohere models can include a 'thinking' block before the actual text
    answer, so we can't assume content[0] is always the text."""
    for item in message_content:
        if hasattr(item, "text"):
            return item.text
    return ""


def run_agent(client, user_question: str, mode: str, status_container):
    """Run the tool-use loop: repeatedly let the model call tools until it
    has enough information to give a final answer. Yields progress updates
    to the status_container so the user can watch the agent think."""
    messages = [
        {"role": "system", "content": build_system_message(mode)},
        {"role": "user", "content": user_question},
    ]

    steps_taken = []

    for step in range(MAX_TOOL_STEPS):
        try:
            response = client.chat(model=MODEL, messages=messages, tools=TOOLS)
        except Exception as exc:
            error_msg = str(exc)
            if "401" in error_msg or "unauthorized" in error_msg.lower():
                return "Your Cohere API key seems to be invalid or expired. Please check the key in the sidebar.", steps_taken
            elif "429" in error_msg or "rate" in error_msg.lower():
                return "The API rate limit was reached. Please wait a moment and try again.", steps_taken
            else:
                return f"Something went wrong talking to the AI model: {error_msg}", steps_taken

        if not response.message.tool_calls:
            # No more tools needed -- this is the final answer
            final_text = extract_text(response.message.content)
            if not final_text.strip():
                final_text = "The agent finished reasoning but didn't produce a text answer. Please try rephrasing your question."
            return final_text, steps_taken

        # Record the assistant's reasoning + tool call plan
        tool_plan = getattr(response.message, "tool_plan", None)
        if tool_plan:
            status_container.write(f"**Thinking:** {tool_plan}")

        messages.append(response.message)

        for tc in response.message.tool_calls:
            args = {}
            try:
                args = json.loads(tc.function.arguments)
                status_container.write(f"**Calling tool:** `{tc.function.name}({args})`")
                tool_result = FUNCTIONS_MAP[tc.function.name](**args)
            except Exception as exc:
                tool_result = {"error": f"Tool execution failed: {exc}"}

            steps_taken.append({
                "tool": tc.function.name,
                "arguments": args,
                "result": tool_result,
            })

            status_container.write(f"**Result:** `{tool_result}`")

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": [{"type": "document", "document": {"data": json.dumps(tool_result)}}],
            })

    # Safety fallback if the agent looped too many times
    return "I wasn't able to fully resolve this within the allowed number of steps. Please try rephrasing your question.", steps_taken


def style_answer(answer: str) -> str:
    """Light styling pass -- escape nothing risky, just ensure clean line breaks."""
    return answer.replace("\n", "<br>")


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    display_sidebar()
    display_header()

    api_key = get_api_key()
    if not api_key:
        st.info("Enter your Cohere API key in the sidebar to get started.")
        st.stop()

    client = cohere.ClientV2(api_key=api_key)

    st.session_state.setdefault("example_question", "")

    example_triggered = False
    example_text = ""

    st.caption("Try an example:")
    ex_col1, ex_col2, ex_col3 = st.columns(3)
    with ex_col1:
        if st.button("Tesla's current price?", use_container_width=True):
            example_triggered = True
            example_text = "What is the current price of Tesla stock, and today's trading range?"
    with ex_col2:
        if st.button("Apple 5-year growth?", use_container_width=True):
            example_triggered = True
            example_text = "If I'd invested $5000 in Apple 5 years ago, how much would it be worth today?"
    with ex_col3:
        if st.button("Analyze Bitcoin", use_container_width=True):
            example_triggered = True
            example_text = "Analyze Bitcoin's price performance and fundamentals over the past year."

    with st.container(border=True):
        with st.form("ask_form", clear_on_submit=True):
            question_input = st.text_input(
                "Ask a finance question",
                placeholder="e.g. If I'd invested $5000 in Apple 5 years ago, what would it be worth today?",
            )
            mode_input = st.radio(
                "Analysis mode",
                options=["Trader View", "Investor View", "Beginner View"],
                horizontal=True,
            )
            form_submitted = st.form_submit_button("Ask the Agent")

    if example_triggered:
        question = example_text
        analysis_mode = "Investor View"
        submitted = True
    elif form_submitted:
        question = question_input
        analysis_mode = mode_input
        submitted = True
    else:
        question, analysis_mode, submitted = None, None, False

    st.session_state.setdefault("history", [])

    if submitted and question:
        with st.container(border=True):
            st.markdown("**Agent's reasoning:**")
            reasoning_container = st.container()
            answer, steps = run_agent(client, question, analysis_mode, reasoning_container)

        st.session_state["history"].append({
            "question": question,
            "mode": analysis_mode,
            "answer": answer,
            "steps": steps,
        })

    for turn in reversed(st.session_state["history"]):
        st.markdown(f"**You asked:** {turn['question']} &nbsp; *({turn.get('mode', 'Investor View')})*")

        with st.container(border=True):
            st.markdown('<span class="answer-badge">Agent Answer</span>', unsafe_allow_html=True)
            st.markdown(f'<div class="answer-card" style="margin-top:0;">{style_answer(turn["answer"])}</div>', unsafe_allow_html=True)

        if turn["steps"]:
            with st.expander(f"Show the {len(turn['steps'])} tool call(s) the agent made"):
                for i, step in enumerate(turn["steps"], 1):
                    st.markdown(f"**Step {i}: `{step['tool']}`**")
                    st.json(step["arguments"])
                    st.json(step["result"])

    if st.session_state["history"]:
        if st.button("Clear conversation"):
            st.session_state["history"] = []
            st.rerun()


if __name__ == "__main__":
    main()
