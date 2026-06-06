# Offline company helpdesk — a richer himmy example

A specialised agent that answers employee questions from a company handbook — **fully
offline**. It's the detailed sibling of [`local-doc-chat`](../local-doc-chat/), adding the
things a *real* assistant needs:

- a reusable **skill** ([`skills/helpdesk.yaml`](skills/helpdesk.yaml)) that defines the
  agent's behaviour as config, not code;
- a **multi-turn conversation with memory**, so follow-ups (*"…and for part-time?"*) work;
- **grounded** answers that name the source doc, and an honest *"that's not in the
  handbook"* when it isn't — it never makes facts up;
- a **golden-test validator** ([`goldens.py`](goldens.py)) so you can *measure* the agent
  before trusting it.

Everything runs locally: 6 handbook docs are embedded with a local hashing embedder (no
download), and answers come from a local model via Ollama.

## Run it

```bash
pip install -e ".[knowledge]"          # from the repo root
ollama pull qwen2.5:3b-instruct        # a 3B+ model — tiny ones hallucinate (see below)
cd examples/offline-helpdesk

python helpdesk.py        # a scripted demo conversation
python helpdesk.py -i     # interactive (keeps memory across turns)
python goldens.py         # validate against known-good answers
```

## What the conversation looks like

```
✓ helpdesk ready — 6 handbook docs in a local KB (qwen2.5:3b-instruct, offline).

you   › How many paid vacation days do full-time employees get?
agent › Full-time employees accrue 20 days of paid vacation per year (handbook: pto.md).

you   › And what about part-time employees?            ← follow-up uses conversation memory
agent › Part-time employees accrue PTO prorated by their scheduled hours (handbook: pto.md).

you   › Can I plug my own USB drive into my work laptop?
agent › No — personal USB drives are not allowed on work machines (handbook: security.md).

you   › What's the company's policy on bringing pets to the office?
agent › That's not in the handbook. You may want to ask HR.    ← honest, not invented
```

## Measure it before you trust it

`goldens.py` runs a fixed set of questions, each with a fact the answer must contain, and
prints a pass-rate. **This is the point of the example as much as the chat is** — a
specialised agent is only useful if you can prove what it does:

```
5/7 golden answers correct.
```

On a **3B model with the zero-download hashing embedder, expect ~5–6/7**, with some
run-to-run variance. The misses are *conservative*, not dangerous: when the crude embedder
returns a low-confidence match, the model tends to say *"that's not in the handbook"*
rather than guess. It never fabricates a number.

**Want higher recall?** Swap the hashing fallback for a real local embedder — one extra,
still offline, still no API:

```bash
pip install -e ".[embeddings]"   # fastembed (ONNX); downloads a small model once
```

That alone pushes the golden score up, because retrieval confidence jumps. (We left it on
the hashing embedder here so the example needs **zero downloads**.)

## Make it yours

Drop your own `.md` files into [`docs/`](docs/), edit
[`skills/helpdesk.yaml`](skills/helpdesk.yaml) to change the agent's behaviour, and add
your own checks to `goldens.py`. No framework code to touch.

## What's in the box

| File | What it shows |
|---|---|
| `skills/helpdesk.yaml` | the agent's behaviour as a **reusable YAML skill** |
| `helpdesk.py` | ingest → **multi-turn** chat with memory, grounded + cited |
| `goldens.py` | a **measure-your-agent** harness (the local cousin of `himmy eval`) |
| `docs/*.md` | the handbook the agent is specialised on |
