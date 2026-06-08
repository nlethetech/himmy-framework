# Connectors & Nepal localization

> Domain connectors that fetch directly from public sources (Nepali news RSS/Atom, Nepal Rastra Bank forex + macro Excel) plus the Nepal localization kernel (Bikram Sambat calendar, Devanagari/Nepali formatting).

## Overview

`himmy/connectors/` holds domain-specific data clients and the tools/MCP-server
that expose them; `himmy/nepal/` holds reusable Nepal localization primitives. Both
are intentionally narrow and self-contained.

- Connectors fetch **directly** from the public source (no intermediary), through
  an injectable `Fetcher` seam, so the whole layer is offline-testable with
  fixtures.
- Optional libraries (`feedparser`, `openpyxl`, `nepali_datetime`) are lazily
  imported and gated behind the `[connectors]` / `[nepal]` extras with clear error
  messages.

These connectors surface to agents as ordinary tools via the `news` and `nepal`
tool packs — see [toolkit](../architecture/toolkit.md) — so they run through the
full `ToolService` pipeline (validation, approval, events, lineage); see
[tools](./tools.md).

## Module map

| File | Responsibility |
| --- | --- |
| `connectors/fetcher.py` | `Fetcher` protocol + `HttpxFetcher` (the injectable network seam); `get_json` helper. |
| `connectors/models.py` | Typed models: `NewsSource`, `NewsItem`, `ForexRate`, `MacroReport`, `Workbook`. |
| `connectors/news.py` | `NewsFetcher` — aggregate/parse curated Nepali RSS/Atom feeds; `NEPAL_NEWS_SOURCES`. |
| `connectors/news_tools.py` | `register_news_tools` — `news_sources`, `news_search`, `news_fetch` as local tools. |
| `connectors/news_mcp_server.py` | A standalone stdio MCP **server** exposing the news connector. |
| `connectors/nrb.py` | `NRBClient` — Nepal Rastra Bank forex JSON API + macro reports + Excel workbook parsing. |
| `connectors/nrb_tools.py` | `register_nrb_tools` — `nrb_forex`, `nrb_macro_reports`, `nrb_macro_workbook` as local tools. |
| `nepal/calendar.py` | `BikramDate` + AD↔BS conversion, month/weekday names, Nepal fiscal year. |
| `nepal/language.py` | Cross-script transliteration/normalization (Devanagari↔Roman) + `NepaliEmbedder`. |

## Key abstractions

### `Fetcher` seam (`fetcher.py`)

A `runtime_checkable` protocol with `get_text` / `get_bytes`. `HttpxFetcher` is the
default (polite `HimmyBot` User-Agent, follows redirects, raises on HTTP error).
Tests pass a fixture-backed fetcher and never hit the network. `get_json(fetcher,
url)` fetches + parses JSON.

### News connector (`news.py`, `models.py`)

`NEPAL_NEWS_SOURCES` is a curated list of `NewsSource`s (onlinekhabar, setopati,
kathmandupost, ekantipur, BBC Nepali, etc., each with `lang` and `category`).
`NewsFetcher`:

- `sources()` — the configured feeds,
- `fetch(name, limit)` — parse one feed (via `feedparser`) into `NewsItem`s,
- `fetch_all(per_source, sources)` — batch; **a single failing feed never breaks
  the batch**,
- `search(query, ...)` — case-insensitive keyword search matching **all** query
  words across title + summary.

### NRB connector (`nrb.py`, `models.py`)

`NRBClient` fetches directly from Nepal Rastra Bank's public surface:

- `forex(from_date, to_date=None)` / `latest_forex()` — the forex JSON API
  (`/api/forex/v1/rates`), normalized into `ForexRate`s. Dates are ISO; each
  currency's `unit` is the quantity quoted (e.g. INR per 100); `per_page` is capped
  at 100.
- `list_macro_reports(limit)` — the monthly "Current Macroeconomic & Financial
  Situation" reports from NRB's category RSS feed, parsing `language`
  (nepali/english/tables) and `period` out of each title.
- `fetch_macro_workbook(url)` / `fetch_latest_macro_workbook()` /
  `parse_workbook(bytes)` — download + parse the macro **Excel** via `openpyxl`.
  NRB's 'Tables' report URLs serve the `.xlsx` directly, so the client first tries
  to parse the fetched bytes as a workbook (magic-byte check), falling back to
  scanning an HTML page for a linked spreadsheet. `fetch_latest_macro_workbook`
  needs no URL — it finds the newest 'Tables' report and parses every sheet (live:
  ~93 sheets of CPI/WPI/GDP data) into `Workbook` (`{sheet name: rows}`).

### Nepal calendar (`nepal/calendar.py`)

`BikramDate(year, month, day)` wraps the authoritative `nepali-datetime` library
(the `[nepal]` extra): `to_ad()` / `from_ad()` convert AD↔BS, `month_name(lang)` /
`weekday_name(lang)` render in English or Devanagari, and `fiscal_year()` resolves
the Nepal fiscal-year label (the FY starts Shrawan 1 — the 4th BS month — so months
1–3 belong to the prior FY). Module-level helpers: `ad_to_bs`, `bs_to_ad`,
`today_bs`, `nepali_fiscal_year`, plus the `NEPALI_MONTHS_*` / `NEPALI_WEEKDAYS_*`
name tables (the Nepali week starts on Sunday).

### Nepali language (`nepal/language.py`)

Real Nepali text mixes Devanagari, Romanized Nepali, and English; most embedders
treat those as unrelated. This module folds across scripts:

- `transliterate(text)` — Devanagari → Roman (handles consonant + matra + halant;
  नेपाल → `nepala`), passing Roman/English through.
- `normalize_nepali(text)` — transliterate + lowercase + word-final schwa deletion,
  so `नेपाल` / `Nepal` / `nepal` all fold to `nepal`.
- `NepaliEmbedder` — wraps any embedder and normalizes text before embedding, so
  RAG retrieval matches across scripts (defaults to the offline
  `DeterministicEmbedder`).

## How it works / data flow

### News tools and offline fixtures

`register_news_tools(registry, fetcher=None)` registers three read-only local tools
(`news_sources`, `news_search`, `news_fetch`); the connector's blocking I/O is run
off the event loop with `asyncio.to_thread`. The `news` pack's registrar (in
`himmy/toolkit/pack.py`) honors **`HIMMY_NEWS_FIXTURE`**: when set to a local RSS
file, it builds a `NewsFetcher` over a fixture fetcher that returns that file for
any URL — so the news tools run fully offline (demos/tests with no network).

### NRB tools

`register_nrb_tools(registry, ...)` registers `nrb_forex`, `nrb_macro_reports`,
`nrb_macro_workbook` (all read-only). The workbook tool returns sheet **names** by
default (the full workbook is far too big for a model context and Excel date cells
aren't JSON-serializable); pass `sheet` to read one sheet's rows (capped via
`max_rows`, dates coerced to ISO). These tools are bundled into the `nepal` pack
alongside the calendar/format/transliterate tools.

### News MCP server (`news_mcp_server.py`)

A standalone stdio JSON-RPC MCP **server** (run `python -m
himmy.connectors.news_mcp_server`) exposing `list_sources`, `fetch_news`,
`search_news`. It implements the server side of the protocol consumed by the
`MCPClient` in [mcp](./mcp.md): handles `initialize` (advertising protocol
`2024-11-05` and a `tools` capability), `tools/list`, and `tools/call`, returning
results as text content blocks and reporting tool errors **in-band** via `isError`.
It honors `HIMMY_NEWS_FIXTURE` for offline operation.

### Nepali response guidance (`AgentSpec.language`)

`AgentSpec.language` (`himmy/config/agent_spec.py`) defaults to `"en"`. Setting it
to **`"ne"`** triggers Nepali response guidance: `to_persona()` appends an
instruction (in Devanagari + English) telling the agent to always respond in the
Nepali language in Devanagari script. This is independent of the `nepal` tool pack —
`language` shapes the *output*, the pack adds Nepal-specific *tools*.

## Configuration

- **News offline:** `HIMMY_NEWS_FIXTURE=/path/to/feed.xml` (used by both the `news`
  pack tools and the news MCP server).
- **Extras:** `feedparser` (news + NRB macro RSS) and `openpyxl` (NRB Excel) require
  `pip install 'himmy[connectors]'`; `nepali_datetime` requires
  `pip install 'himmy[nepal]'`. All are lazily imported with actionable errors.
- **Agent language:** `language: ne` in `agent.yaml` → Nepali/Devanagari responses.
- **NRB endpoints** are constants in `nrb.py` (`NRB_FOREX_API`, `NRB_MACRO_FEED`).

## Extension points

- **Add a news source:** append a `NewsSource` to `NEPAL_NEWS_SOURCES` (or pass a
  custom `sources=` to `NewsFetcher`).
- **New domain connector:** model it on `NewsFetcher`/`NRBClient` — take a `Fetcher`
  in the constructor, expose typed methods, then a `register_*_tools` registrar; add
  it to a tool pack in [toolkit](../architecture/toolkit.md).
- **Cross-script RAG:** wrap any embedder in `NepaliEmbedder`.

## Gotchas & invariants

- **Direct fetch, no intermediary.** Feeds and APIs are fetched straight from the
  outlet/NRB; the `Fetcher` seam is the only network dependency.
- **One bad feed never sinks a batch** (`fetch_all` swallows per-feed errors).
- **NRB forex `per_page` caps at 100** — larger values return nothing.
- **The macro workbook is huge** — `nrb_macro_workbook` returns sheet names by
  default; request one `sheet` at a time.
- **Optional extras are lazily imported** — missing `feedparser`/`openpyxl`/
  `nepali_datetime` raises a clear `HimmyError` naming the extra, never an
  `ImportError` at module load.
- **`language: ne` shapes output, not tools.** It only adds a response-language
  instruction; Nepal data tools come from the `nepal` pack.
- **MCP server errors are in-band** (`isError`), matching MCP semantics.

## Related docs

- [toolkit](../architecture/toolkit.md) — the `news` / `nepal` packs that surface
  these connectors.
- [tools](./tools.md) — the pipeline the connector tools run through.
- [mcp](./mcp.md) — the MCP client that consumes the news MCP server.
