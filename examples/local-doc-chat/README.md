# Chat with your own documents — 100% offline

A ~60-line example that answers questions about **your** documents using a **local**
model. No API keys. No cloud. Nothing leaves your machine.

It ingests the Markdown files in [`docs/`](docs/) into himmy's local knowledge base
(a local hashing embedder — nothing is downloaded), then answers with a small model via
**Ollama**, grounded strictly in those docs and citing the source.

## Run it

```bash
pip install -e ".[knowledge]"          # from the repo root
ollama pull qwen2.5:3b-instruct        # any small local model
cd examples/local-doc-chat

python chat.py                          # interactive
python chat.py "How many PTO days do I get?"   # one-shot
```

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
  himmy run -f agent.yaml -p "Three benefits of running agents locally?" --record session.json
  himmy run -f agent.yaml -p "Three benefits of running agents locally?" --replay session.json  # no Ollama needed
  ```

  The replay returns byte-identical output from `session.json` — that "re-run a past agent
  session deterministically, no provider" is something most agent frameworks can't do.
