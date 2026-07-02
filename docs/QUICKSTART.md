# Quickstart — Himmy in 5 minutes

This is the fastest path from nothing to a working AI agent on your own machine.
No prior experience needed. Every command below is copy-paste-and-run.

You will:

1. **Install** Himmy.
2. **Bring a model** — either a free, offline local model (no account, no internet
   after download) **or** an API key.
3. **Run** Himmy — the local point-and-click app (**Studio**) and/or the terminal.
4. Do a **60-second "talk to an agent"** example.
5. See **where to go next**.

> **Good to know before you start.** Himmy is *offline-first*. Out of the box it runs
> against a built-in deterministic stand-in (the "stub") that produces canned text — no
> account, no internet, no API key. That's perfect for checking everything is wired up,
> but the answers are placeholders, not real intelligence. To get real answers you bring
> a model: a **free local one (Ollama)** or an **API key**. Both paths are below; the
> local one needs **no API key and works fully offline**.

Works on **macOS** and **Linux**. You need **Python 3.12 or newer** — check with:

```bash
python3 --version
```

If that prints `Python 3.12.x` or higher, you're good. (On macOS you can install Python
from [python.org](https://www.python.org/downloads/); on Linux use your package manager,
e.g. `sudo apt install python3.12`.)

---

## 1. Install Himmy

Install Himmy with the **Studio** (the local app) included:

```bash
pip install 'himmy[studio]'
```

> **If that says "No matching distribution found":** Himmy is pre-1.0 and may not be on
> PyPI yet. In that case install from the source folder you already have — from inside
> the `himmy-agent-test` directory:
>
> ```bash
> pip install -e '.[studio]'
> ```
>
> This source checkout already ships the pre-built Studio app, so Studio runs immediately
> with no extra build step.

The quotes around `'himmy[studio]'` matter — they stop your shell from mangling the
square brackets. (Only want the terminal tool, not the app? Use `pip install himmy`.)

Check it worked:

```bash
himmy --version
```

Expected output:

```
himmy 0.2.0
```

If `himmy` isn't found, your Python scripts folder may not be on your `PATH`; you can
always run it as `python3 -m himmy` instead (e.g. `python3 -m himmy --version`).

---

## 2. Bring a model

Pick **one** of these. Path A is free and offline. Path B uses a paid cloud account.

### Path A — Local model with Ollama (free, offline, no API key)

[Ollama](https://ollama.com) runs open models entirely on your own machine. After the
one-time model download, **no internet and no API key are ever needed.**

1. Install Ollama from [ollama.com/download](https://ollama.com/download)
   (macOS: download the app; Linux: `curl -fsSL https://ollama.com/install.sh | sh`).
2. Download a small, capable model (about 2 GB — a one-time download):

   ```bash
   ollama pull qwen2.5:3b-instruct
   ```

That's it. Himmy auto-detects Ollama and uses it. You never type an API key.

### Path B — Cloud model with an API key

Prefer a hosted model? Set **one** API key in your terminal:

```bash
# OpenAI:
export OPENAI_API_KEY=sk-...

# …or Anthropic (Claude):
export ANTHROPIC_API_KEY=sk-ant-...
```

…then install the provider support:

```bash
pip install 'himmy[providers]'
```

(`export` lasts for the current terminal window. To make it permanent, add the line to
your shell profile, e.g. `~/.zshrc` on macOS or `~/.bashrc` on Linux.)

### Not sure what you have? Ask Himmy

```bash
himmy doctor
```

This shows your Python version, which models it can reach (local + cloud), and a suggested
next step. The **embedders / providers** sections tell you exactly what's available.

> **Want to try Himmy with zero setup?** You can skip step 2 entirely. With no model
> installed, Himmy falls back to the offline stub and still runs — you'll just get
> placeholder text instead of real answers, and a one-line note telling you how to add a
> real model. Great for a first look; come back to step 2 when you want real replies.

---

## 3. Run Himmy

You can use the **app (Studio)**, the **terminal**, or both. They share the same agents
and history.

### The app — Himmy Studio (point-and-click)

```bash
himmy studio
```

Studio starts a small local web app and opens it in your browser at
**http://127.0.0.1:8765**. It's bound to your own machine only (`127.0.0.1`), so nothing
is exposed to the network. Use **Ctrl-C** in the terminal to stop it.

Inside Studio you can chat with an agent, build new agents with a form (no YAML), browse
tools, and see a full audit trail of every run — all without touching the terminal.

> If you installed Himmy from a fresh source clone *without* the pre-built app, `himmy
> studio` will print the exact one-time build command (`cd studio && npm install && npm
> run build`, needs Node 18+) and stop. The provided checkout already has it built.

### The terminal — the `himmy` command

Just ask a question directly:

```bash
himmy "What is the capital of Nepal?"
```

Himmy automatically uses the best model it can find on your machine — your local Ollama
if you set one up, otherwise the offline stub. With Ollama installed you'll get a real
answer like:

```
Kathmandu is the capital of Nepal.
```

With no model installed yet, you'll instead see placeholder text plus a short note on how
to add a real model — that's the offline stub, and it's expected.

For a back-and-forth conversation that remembers the thread:

```bash
himmy chat
```

Type your messages; type `/exit` (or `/quit`) to leave, `/reset` to start the thread over,
`/help` to list commands.

---

## 4. Your first agent in 60 seconds

An "agent" is just a tiny text file describing a role. Let Himmy scaffold one. The
`--classic` flag writes a complete, fully-commented example you can read and edit:

```bash
himmy init my-agent --classic
```

Expected output:

```
wrote my-agent/agent.yaml
wrote my-agent/tools.py
wrote my-agent/himmy.toml
wrote my-agent/skills/my_skill.yaml
wrote my-agent/Dockerfile

Next: himmy run -f my-agent/agent.yaml -p "hello"
```

The `Dockerfile` is a ready-to-build container front door: `docker build` from this
folder layers your spec onto the published runtime image (no framework checkout). It is
never written over one you already have.

> **On a real terminal, plain `himmy init my-agent` (without `--classic`) is
> interactive.** It asks a few short questions — name, what it should do, which model,
> tool packs, memory — and you can just **press Enter to accept each default**. That path
> writes a single minimal `my-agent/agent.yaml` (no `tools.py`/`himmy.toml`/skills) and
> prints `wrote my-agent/agent.yaml`. Use `--classic` when you want the full annotated
> scaffold shown above, or run plain `himmy init my-agent` for the guided setup. (Piped
> or CI runs always get the classic scaffold automatically.)

Now run a prompt through it.

**Offline / no API key (works right now, no model needed):**

```bash
himmy run -f my-agent/agent.yaml -p "Say hello in one sentence."
```

On the offline stub this prints deterministic placeholder text (it begins with `[stub:…]`)
— proof the wiring works end to end, on $0 and with no network.

**With your free local model (real answer):**

```bash
himmy run -f my-agent/agent.yaml -p "Say hello in one sentence." \
  --provider ollama --model qwen2.5:3b-instruct
```

Expected output (a real, friendly reply):

```
Hello! How can I assist you today?
```

**With a cloud key:** set the key (step 2, Path B), then run the same command without the
`--provider`/`--model` flags — Himmy auto-routes to your cloud model.

That's a working agent. Open `my-agent/agent.yaml` in any text editor to see how small it
is — change the `role` and `instructions` lines, save, and re-run.

---

## 4b. Now deploy it — one command

You have an agent that runs. To make it a real, reachable HTTP service (so another program,
a webhook, or a friend can call it), one command stands up **serve + worker together**:

```bash
himmy deploy -f my-agent/agent.yaml
```

That boots the FastAPI server bound to `127.0.0.1:8000` **plus** the background worker
(scheduler + run-queue), mounts your agent as a **signature-verified** webhook at
`POST /v1/connectors/webhook`, and prints a boxed live summary with a **ready-to-paste,
already-signed `curl`** — paste it and you'll get a real answer back from your agent. It is
fail-closed by default: bound to loopback, default-deny, and it never prints the raw signing
secret (only a valid signature for the sample payload).

- **Let a friend try it** without a cloud account: `himmy deploy --share` mints an API key
  and turns auth **on first**, then prints a `cloudflared`/`ngrok` tunnel command — it never
  exposes an unauthenticated endpoint.
- **Scheduled/unattended runs.** Add a routine (`himmy routines add …`) and the worker
  `himmy deploy` started is what actually fires it on schedule.
- **Containers, Compose, Helm, one-click cloud, and wiring the webhook by hand** are in the
  [deployment runbook](./enterprise/deployment.md#deploy-my-agent-a-service) and
  [`RECIPES.md`](../RECIPES.md).

---

## 5. Where to go next

- **See what your agent can do.** List the built-in tool bundles and ready-made skills:

  ```bash
  himmy tools
  himmy skills
  ```

- **Give your agent abilities.** Add a line like `skills: [web_research]` to
  `agent.yaml` to let it search the web (keyless), or `knowledge: [./docs]` to ground it
  in your own documents. The [README](../README.md) — section **"Extend it — tools,
  knowledge, and skills"** — walks through this.

- **Start from a working template** instead of a blank one:

  ```bash
  himmy init my-agent --template researcher   # web research
  himmy init my-agent --template analyst      # live data lookups
  himmy init my-agent --template helpdesk     # answer from your docs
  ```

- **Explore the runnable examples** in the [`examples/`](../examples/) folder — every
  `examples/0N_*.py` script runs offline:

  ```bash
  python3 examples/01_basic_chat.py
  python3 examples/11_web_research.py     # keyless web search
  ```

- **Go deeper.** The full [README](../README.md) covers memory, multi-agent teams,
  the audit/replay trail, and the security posture. For running Himmy for a team or
  organization, see [`docs/enterprise/deployment.md`](./enterprise/deployment.md).

- **Stuck?** Run `himmy doctor` — it almost always tells you the missing piece and the
  one command to fix it.

Welcome to Himmy.
