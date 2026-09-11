# Finance Research Agent

An AI agent that autonomously decides which tools to call -- live stock/crypto prices, historical performance, fundamentals, and a calculator -- to answer investing questions, instead of following a fixed script.

**🔗 Live App:** https://finance-research-agent-3mhg7imzslrb64kj3jmqvt.streamlit.app/
---

## Overview

Unlike a simple RAG or single-prompt chatbot, this is a genuine **tool-use agent**: given a question, the LLM (Cohere Command A Plus) decides for itself which tools it needs, calls them, reads the results, and repeats until it has enough information to answer -- all visible live in the UI as it happens.

## Features

- **Autonomous multi-step reasoning** -- the agent chains multiple tool calls together to answer compound questions (e.g. "if I'd invested $5000 in Apple 5 years ago, what would it be worth today?")
- **Live market data** -- real-time stock and cryptocurrency prices via Yahoo Finance (no API key required)
- **Analysis modes** -- Trader View (price action/momentum), Investor View (fundamentals/valuation), Beginner View (plain-language explanations)
- **Visible reasoning** -- every tool call and its result is shown step-by-step, not hidden behind the final answer
- **Scoped and honest** -- strictly focused on finance/investing; declines off-topic questions rather than pretending to be a general assistant
- **Graceful error handling** -- invalid API keys, rate limits, and tool failures are caught and explained, not left to crash the app

## Tools

| Tool | Purpose |
|---|---|
| `get_stock_price` | Current price + today's high/low for a stock or crypto ticker |
| `get_stock_history` | Historical performance and % change over a chosen period |
| `find_dates_in_price_range` | Searches history to find when a price fell within a specific range |
| `get_fundamentals` | Market cap, P/E, EPS, profit margin, revenue growth, debt, cash (stocks only) |
| `calculate` | Safe arithmetic evaluation, e.g. for investment growth calculations |

## Tech Stack

- **Language:** Python
- **LLM & Tool Use:** Cohere API (Command A Plus)
- **Market Data:** yfinance
- **App:** Streamlit

## Project Structure

```
AI-Agent-Assistant/
│
├── app.py                 # Streamlit app + agent loop
├── requirements.txt        # Python dependencies (version-pinned)
└── .streamlit/
    └── secrets.toml        # API key (not committed to git)
```

## Getting Started

**1. Clone the repository**
```bash
git clone https://github.com/hammeshkh5446-sudo/AI-Agent-Assistant.git
cd AI-Agent-Assistant
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add your Cohere API key**

Create a free trial key at [dashboard.cohere.com](https://dashboard.cohere.com/api-keys), then either:
- Enter it directly in the app's sidebar when it runs, or
- Create `.streamlit/secrets.toml` with:
  ```toml
  COHERE_API_KEY = "your-key-here"
  ```

**4. Run the app**
```bash
streamlit run app.py
```

## Example Questions

- "What is the current price of Tesla stock, and today's trading range?"
- "If I'd invested $5000 in Apple 5 years ago, how much would it be worth today?"
- "When did Solana's price hit $172-175 in the past year?"
- "Analyze Bitcoin's price performance and fundamentals over the past year." (Investor View)
- "Compare Tesla and Apple's current prices and fundamentals."

## Author

**M. Hammad Shahbaz**
