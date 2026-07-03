# himmy → Enterprise, Sellable — Strategy & Roadmap (2026-07-02)

**Decision:** target = enterprises (top-down, paid); goal = monetizable/sellable.

## The one-line truth
**The product is ~70% enterprise-ready; the *business model* is 0% built.** himmy is unusually strong on exactly the right things (self-host, air-gap, any-model incl. open-weight, audit spine, RBAC, OIDC, KMS, one-command deploy). The revenue blocker is **not code — it's trust/proof + licensing + commercial packaging.** Roadmap to first dollars is ~60% commercial/trust, ~40% product.

## The wedge (defensible, true for himmy, false for competitors)
**"The agent platform you run entirely inside your own perimeter, on your own (open-weight) models, with a built-in audit spine — software you *own*, not a metered cloud."**
Buyer = regulated / sovereign / IP-sensitive enterprises that **can't/won't send data to OpenAI** (DORA, EU AI Act Aug-2026, HIPAA, ITAR/CUI tailwind). This is structurally impossible for: OpenAI Agents SDK (GPT-locked), Google ADK (GCP-metered), LangGraph/CrewAI (self-host is a $60–120k **enterprise-only upsell**, not the default). himmy inverts the whole market: **self-host & air-gap are the product, not the $100k upsell.**

## What makes an enterprise say NO today (the real blockers, in order)
1. **No third-party proof** — no SOC 2 Type II, ISO 42001, pen-test report, Trust page, reference customers, support SLA. *#1 blocker. Not code.*
2. **MIT license** — anyone (a cloud vendor, or the customer's own team) can take himmy + add SSO + resell it. Guts the paid tier before it exists.
3. **"Secure multi-tenant" is aspirational for UNTRUSTED tenants** — himmy's own RBAC report grades B-; auth doesn't fully follow into tool execution (confused-deputy), subject-axis BOLA, coarse Studio perms. → **Don't sell multi-tenant SaaS. Sell single-tenant, one-org-per-deploy** — which is *also* exactly what the sovereign buyer wants, so this is alignment, not a compromise.
4. **No commercial wrapper** — no legal entity, support/SLA, DPA/MSA, security questionnaire (SIG/CAIQ), IP indemnity. Solo-founder viability risk.
5. **Single-instance (no HA/multi-node)** — matters later; the beachhead deploy doesn't need it.

## Competitive reality
Out-gunned on mindshare/ecosystem/funding (single-maintainer, 0.2.0, no SOC2 vs LangGraph/Microsoft/Google with thousands of integrations + logos). **Do NOT compete on breadth — win the narrow beachhead.** Toughest adjacent rival: **Pydantic AI + Logfire** (also model-agnostic + OTel, huge implicit trust). himmy's diff vs them = truly offline/air-gapped incl. open-weight tool-calling + ownership economics (Logfire is cloud).

---

## Roadmap

### P0 — minimum to close a FIRST paying enterprise (a design-partner deal)
| # | Move | Kind | Who |
|---|------|------|-----|
| 1 | **Relicense: core → Apache-2.0; Enterprise features → a proprietary "EE" license.** Do FIRST (messy to relicense later once contributors exist). | package/legal | you + me |
| 2 | **Entitlement/edition seam** — thin license-key + feature-flag layer that gates EE features. (grep billing/entitlement/license today = 0 hits — this is the ONE real build for v1 monetization.) | **build** | me |
| 3 | **Package the "Enterprise Edition" bundle** = SSO/OIDC + advanced RBAC/service-accounts + signed audit-export + air-gap tooling + priority support. All already built except the seam. | package | me |
| 4 | **Reposition to single-tenant / one-org self-host.** Stop claiming untrusted-multi-tenant; align docs + pitch with what's actually secure (and what the buyer wants). | gtm | me (docs) |
| 5 | **Trust/procurement kit** — security whitepaper (generated from the RBAC hardening + 6-round red-team + audit spine), pre-filled SIG-Lite/CAIQ, DPA/MSA templates, "we never see your data" data-flow statement, a Trust page. | proof | me (draft) + you |
| 6 | **Start SOC 2 Type I + observation window NOW** — 3–6 month clock, gates the timeline. A dependency to kick off, not a build. | proof | you |
| 7 | **Legal entity + paid support/SLA tier definition** — someone to sign an MSA + commit support. | gtm | you |
| 8 | **Land 1–2 design-partner reference customers** — unlocks everything (proof + funds the roadmap). | gtm | you |

### P1 — scale deals
- SOC 2 Type II completion + ISO 42001.
- **Hosted CONTROL PLANE over self-hosted agents** (data/inference stay on-prem; SaaS handles fleet inventory, config, audit aggregation, health) — the expansion-revenue SaaS play. **build**, large.
- HA/multi-node (Postgres coordination) for bigger deployments. **build**.
- Enterprise governance/cost/observability dashboard — "make the invisible visible." **build**.
- IP-output indemnity; cross-agent observability + cost chargeback.

### P2 — later
FedRAMP (if gov), integrations/marketplace, partner/FDE motion, deeper multi-tenant certification (only if a specific deal needs it).

## Pricing / packaging
- **License:** Apache-2.0 core + proprietary Enterprise Edition (protects the paid tier; enterprises' legal prefer Apache over MIT for the patent grant).
- **MVP = licensed EE annual bundle** (NOT hosted). Fastest path to first dollars; least new engineering.
- **Metric:** flat annual platform fee now; **per-node / per-agent** for the control plane later. **NOT per-seat** (agents aren't seats), **NOT pure usage** (can't meter on-prem / breaks the "no call-home" promise).
- **Motion:** OSS = multi-year mindshare (don't monetize/strangle it) FEEDING top-down design-partner EE sales.

## Proof artifacts himmy can generate NOW (from work already done)
- **Security whitepaper** — from the RBAC hardening + 6-round red-team + tamper-evident audit spine.
- **Benchmark report** — from the eval harness: himmy vs raw SDK on cost/latency + any-model + the −62%/−70% efficiency wins.
- **Reference air-gap deploy runbook** — from `airgap.md` + `airgap_bundle.py`.

## Honest verdict
Closer to sellable than it looks *because the hard engineering is done* — but the first sale is gated by **~3 non-code moves: fix the license, build the entitlement seam + package EE, and produce the trust kit + start SOC 2 + land a design partner.** Single biggest blocker: **you can't sell trust you can't evidence.**
