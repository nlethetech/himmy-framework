# Chat with your own documents — 100% offline

A ~60-line example that answers questions about **your** documents using a **local**
model. No API keys. No cloud. Nothing leaves your machine.

It ingests the Markdown files in [`docs/`](docs/) into himmy's local knowledge base
(a local hashing embedder — nothing is downloaded), then answers with a small model via
**Ollama**, grounded strictly in those docs and citing the source.

## Run it — no code, just `agent.yaml`

The whole thing is one file ([`agent.yaml`](agent.yaml)): `knowledge: [./docs]`
auto-ingests your documents at startup, so you never write a driver.

```bash
pip install -e ".[knowledge]"          # from the repo root
ollama pull qwen2.5:3b-instruct        # any small local model
cd examples/local-doc-chat

himmy run  -f agent.yaml -p "How many PTO days do I get?"   # one-shot
himmy chat -f agent.yaml                                    # interactive
```

Point it at *your* docs by editing the one line `knowledge: [./docs]`.

### …or use himmy as a library ([`chat.py`](chat.py))

The same agent in ~60 lines of Python, if you'd rather drive it programmatically:

```bash
python chat.py "How many PTO days do I get?"   # one-shot
python chat.py                                  # interactive
```

> **Model size matters.** This demo needs a model that reliably calls tools. A **3B**
> model (qwen2.5:3b) grounds every answer in the docs and cites the source. Tiny models
> (e.g. qwen2.5:**0.5b**) often *skip* the search and answer from their own memory —
> i.e. they hallucinate. If answers look made-up, use a bigger local model.

## What it looks like

```
✓ ingested 3 documents into a local knowledge base — offline, no keys.

you   › How many PTO days do I get per year?
agent › Full-time employees accrue 20 days of paid time off per year, rolling over up
        to 5 days. (source: handbook-pto.md)

you   › Where do I report a phishing email?
agent › Report phishing to security@company.com. (source: handbook-security.md)
```

## Make it yours

Drop your own `.md` files into [`docs/`](docs/) and ask away — a product manual, meeting
notes, a research folder, your company handbook. The model runs locally, so it works on a
plane, behind a firewall, or anywhere you don't want your documents touching a third party.

## Why this is a good showcase of himmy

- **Offline-first** — `build_inference_for("ollama", ...)`; swap in `claude-cli` or a
  cloud provider without touching the rest.
- **Grounded + cited** — the agent must `kb_search` and answer only from what it finds.
- **Auditable** — every run is on the entity lineage spine, and **any** himmy run can be
  recorded and replayed *exactly*, with the provider turned off:

  ```bash
  himmy run -f agent.yaml -p "How many PTO days do I get?" --record session.json
  himmy run -f agent.yaml -p "How many PTO days do I get?" --replay session.json  # no Ollama needed
  ```

  The replay returns byte-identical output from `session.json` — that "re-run a past agent
  session deterministically, no provider" is something most agent frameworks can't do.
