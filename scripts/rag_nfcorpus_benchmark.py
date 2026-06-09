#!/usr/bin/env python3
"""Real-corpus RAG benchmark on BEIR NFCorpus (a standard medical-IR dataset).

Unlike the small hand-built regression gate (``tests/integration/test_rag_eval.py``),
this runs Himmy's retrieval against a *real, published* IR benchmark with real queries
and human relevance judgments (qrels), reporting recall@k / MRR / nDCG@k / hit-rate for
dense vs hybrid (BM25+dense RRF). It downloads + caches the dataset itself and is fully
re-runnable.

Because the default embedder is a live Ollama model (``qwen3-embedding``, 4096-d, ~8B),
the run embeds a *fair subsample* (real queries with a moderate number of relevant docs,
their relevant docs, plus seeded random distractors) so it finishes in minutes rather
than embedding all 3,633 docs through an 8B model. Scale knobs are env vars.

Usage:
    python scripts/rag_nfcorpus_benchmark.py
    HIMMY_RAG_QUERIES=30 HIMMY_RAG_DISTRACTORS=600 python scripts/rag_nfcorpus_benchmark.py
    HIMMY_EMBEDDER=deterministic python scripts/rag_nfcorpus_benchmark.py   # offline baseline

Env:
    HIMMY_EMBEDDER         embedder backend (default: ollama → qwen3-embedding)
    HIMMY_OLLAMA_EMBED_MODEL  ollama embed model (default: qwen3-embedding)
    HIMMY_RAG_QUERIES      number of eval queries (default: 25)
    HIMMY_RAG_DISTRACTORS  random distractor docs added to the corpus (default: 500)
    HIMMY_RAG_TOPK         retrieval cutoff k (default: 10)
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

from himmy.services.knowledge import RetrievalConfig
from himmy.services.knowledge.local_embedders import build_embedder, default_dim_for
from himmy.services.knowledge.retrieval_eval import RetrievalEvalCase, compare_retrieval
from himmy.services.knowledge.service import KnowledgeBase
from himmy.services.storage.service import StorageService

_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip"
_CACHE = Path(os.environ.get("HIMMY_BEIR_CACHE", Path.home() / ".cache/himmy/beir"))
_SEED = 42


def _ensure_dataset() -> Path:
    """Download + extract NFCorpus into the cache (idempotent); return its dir."""
    root = _CACHE / "nfcorpus"
    if (root / "corpus.jsonl").exists():
        return root
    _CACHE.mkdir(parents=True, exist_ok=True)
    zip_path = _CACHE / "nfcorpus.zip"
    print(f"downloading NFCorpus → {zip_path} …", flush=True)
    try:
        urllib.request.urlretrieve(_URL, zip_path)  # noqa: S310 - fixed trusted URL
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(_CACHE)
    except Exception as exc:  # noqa: BLE001
        print(
            f"FAILED to fetch the dataset ({exc}). Need network access.",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc
    return root


def _load(root: Path) -> tuple[dict[str, str], dict[str, str], dict[str, set[str]]]:
    """Load corpus {id->title+text}, queries {id->text}, qrels {qid->{relevant doc ids}}."""
    corpus: dict[str, str] = {}
    for line in (root / "corpus.jsonl").read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        corpus[d["_id"]] = f"{d.get('title', '')}\n{d.get('text', '')}".strip()
    queries: dict[str, str] = {}
    for line in (root / "queries.jsonl").read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        queries[d["_id"]] = d["text"]
    qrels: dict[str, set[str]] = {}
    rows = (root / "qrels" / "test.tsv").read_text(encoding="utf-8").splitlines()
    for row in rows[1:]:  # skip header "query-id corpus-id score"
        qid, did, score = row.split("\t")
        if int(score) >= 1:
            qrels.setdefault(qid, set()).add(did)
    return corpus, queries, qrels


async def _run() -> None:
    n_queries = int(os.environ.get("HIMMY_RAG_QUERIES", "25"))
    n_distract = int(os.environ.get("HIMMY_RAG_DISTRACTORS", "500"))
    top_k = int(os.environ.get("HIMMY_RAG_TOPK", "10"))
    backend = os.environ.get("HIMMY_EMBEDDER", "ollama")

    corpus, queries, qrels = _load(_ensure_dataset())

    # Fair subsample: queries with a moderate number of in-corpus relevant docs (so
    # recall@k is meaningful), deterministically ordered, then their relevant docs +
    # seeded random distractors form the eval corpus.
    eligible = sorted(
        q
        for q, rel in qrels.items()
        if q in queries and 3 <= len({d for d in rel if d in corpus}) <= 15
    )
    chosen_q = eligible[:n_queries]
    relevant_docs: set[str] = set()
    for q in chosen_q:
        relevant_docs |= {d for d in qrels[q] if d in corpus}

    rng = random.Random(_SEED)
    pool = [d for d in corpus if d not in relevant_docs]
    distractors = set(rng.sample(pool, min(n_distract, len(pool))))
    doc_ids = sorted(relevant_docs | distractors)

    embedder = build_embedder(backend)
    dim = getattr(embedder, "dim", default_dim_for(backend))
    model = getattr(embedder, "model", backend)
    print(
        f"\nNFCorpus benchmark — embedder={backend}:{model} dim={dim} | "
        f"{len(chosen_q)} queries, {len(doc_ids)} docs "
        f"({len(relevant_docs)} relevant + {len(distractors)} distractors), top_k={top_k}",
        flush=True,
    )

    kb = KnowledgeBase(
        storage=StorageService(),
        embedder=embedder,
        retrieval=RetrievalConfig(mode="hybrid"),
    )
    rec = await kb.create_kb(
        workspace_id="bench", client_id="bench", name="nfcorpus", vector_dim=dim
    )

    t0 = time.perf_counter()
    id_map: dict[str, str] = {}
    for i, did in enumerate(doc_ids, 1):
        doc = await kb.ingest_text(rec.kb_id, corpus[did][:1200], title=did)
        id_map[did] = doc.document_id
        if i % 100 == 0:
            print(
                f"  ingested {i}/{len(doc_ids)} docs ({time.perf_counter() - t0:.0f}s)",
                flush=True,
            )
    print(f"  ingest done in {time.perf_counter() - t0:.0f}s", flush=True)

    cases = [
        RetrievalEvalCase(
            query=queries[q],
            relevant_document_ids=[id_map[d] for d in qrels[q] if d in id_map],
            top_k=top_k,
            label=q,
        )
        for q in chosen_q
    ]

    t1 = time.perf_counter()
    reports = await compare_retrieval(
        kb,
        rec.kb_id,
        cases,
        {
            "dense": RetrievalConfig(mode="dense"),
            "hybrid": RetrievalConfig(mode="hybrid"),
        },
        workspace_id="bench",
        client_id="bench",
    )
    print(f"  retrieval+scoring in {time.perf_counter() - t1:.0f}s\n", flush=True)

    print(
        f"{'config':8} {'recall@' + str(top_k):>10} {'mrr':>8} {'ndcg@' + str(top_k):>10} {'hit_rate':>9}"
    )
    for name in ("dense", "hybrid"):
        r = reports[name]
        print(
            f"{name:8} {r.mean_recall_at_k:>10.3f} {r.mean_mrr:>8.3f} "
            f"{r.mean_ndcg_at_k:>10.3f} {r.hit_rate:>9.3f}"
        )
    d, h = reports["dense"], reports["hybrid"]
    delta = h.mean_ndcg_at_k - d.mean_ndcg_at_k
    verdict = (
        "hybrid helps"
        if delta > 0.01
        else ("dense sufficient" if abs(delta) <= 0.01 else "hybrid hurts")
    )
    print(f"\nnDCG@{top_k} delta (hybrid - dense) = {delta:+.3f} → {verdict}")


if __name__ == "__main__":
    asyncio.run(_run())
