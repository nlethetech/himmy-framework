# Connectors as agent tools — news + foreign exchange

himmy's **connectors** are typed data-source integrations. This example shows an agent
using **two** of them, on a local Ollama model, with no driver code — just an `agent.yaml`:

- **`news`** — read RSS/Atom feeds (`news_search`, `news_fetch`, `news_sources`)
- **`nepal`** — Nepal Rastra Bank foreign-exchange rates (`nrb_forex`), plus BS dates / NPR

Connectors fetch on demand, so unlike a RAG agent there's nothing to pre-load — the agent
just calls the tool when a question needs it.

## Run it

```bash
pip install -e ".[connectors,nepal]"   # feedparser + openpyxl
ollama pull qwen2.5:3b-instruct
cd examples/connectors-demo
```

**News — fully offline** (uses the bundled `sample-news.xml` via `HIMMY_NEWS_FIXTURE`):

```bash
HIMMY_NEWS_FIXTURE="$PWD/sample-news.xml" \
  himmy run -f agent.yaml -p "Is there any news about trade with India?"
```
```
⚙ news_search {"query": "trade India"}
agent › An article reports the foreign minister will visit India next week for trade
        talks, covering cross-border trade and energy cooperation.
```

**Foreign exchange — live** (calls NRB's real API; needs network):

```bash
himmy run -f agent.yaml -p "What was the Euro buy and sell rate on 2024-06-03?"
```
```
⚙ nrb_forex {"from_date": "2024-06-03"}
agent › On 2024-06-03 the Euro (EUR) buy rate was 144.48 and the sell rate was 145.13.
```

Both verified on `qwen2.5:3b-instruct`.

## How it works

- The agent binds two tools (`news_search`, `nrb_forex`) — small models route reliably
  between a couple of clearly-named tools.
- **News** hits live feeds by default; setting `HIMMY_NEWS_FIXTURE` to a local RSS file
  makes it run with no network (used here so the demo is reproducible offline).
- **`nrb_forex`** always calls NRB's live endpoint, so the forex question needs internet.

## Make it yours

Point `news` at your own feeds (any RSS/Atom), or drop the `nepal` pack and add other
connector-backed tools. The agent definition stays a single YAML file — no Python.
