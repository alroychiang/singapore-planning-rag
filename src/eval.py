"""
Retrieval evaluation harness.

Loads data/eval/queries.jsonl (queries + hand-labeled relevant chunk IDs),
runs retrieval for each query, computes Precision@K, Recall@K, MRR, and
per-category breakdowns. Writes:

- reports/eval_results.json     (machine-readable, full detail)
- reports/eval_summary.md       (human-readable, for README + interview)

Metrics:
- Precision@K = |retrieved ∩ relevant ∩ top-K| / K
- Recall@K    = |retrieved ∩ relevant ∩ top-K| / |relevant|
- MRR         = mean over queries of (1 / rank of first relevant chunk),
                or 0 if no relevant chunk appears in top-K

Categories:
- parameter_table, checkbox_table, domain_terminology, prose
  → evaluated on precision/recall/MRR against labeled relevant chunks
- should_refuse
  → no relevant chunks by design; tracked separately. Real evaluation of
    refusal behavior belongs to a generation eval, not this retrieval harness.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from generate import query  # reuse retrieval

QUERIES_PATH = Path("data/eval/queries.jsonl")
RESULTS_PATH = Path("reports/eval_results.json")
SUMMARY_PATH = Path("reports/eval_summary.md")
K_VALUES = [5, 10]


# ---------- Metric primitives ----------

def precision_at_k(retrieved_ids, relevant_ids, k):
    if k == 0:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for cid in top_k if cid in relevant_ids)
    return hits / k


def recall_at_k(retrieved_ids, relevant_ids, k):
    if not relevant_ids:
        return None  # undefined
    top_k = retrieved_ids[:k]
    hits = sum(1 for cid in top_k if cid in relevant_ids)
    return hits / len(relevant_ids)


def first_relevant_rank(retrieved_ids, relevant_ids):
    """1-indexed rank of first relevant chunk, or None if none present."""
    for i, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant_ids:
            return i
    return None


# ---------- Per-query eval ----------

def evaluate_query(q, max_k):
    """Run retrieval and compute all metrics for one query."""
    results = query(q["query"], k=max_k)
    retrieved_ids = results["ids"][0]
    relevant_ids = set(q["relevant_chunk_ids"])

    metrics = {}
    for k in K_VALUES:
        metrics[f"precision_at_{k}"] = precision_at_k(retrieved_ids, relevant_ids, k)
        metrics[f"recall_at_{k}"] = recall_at_k(retrieved_ids, relevant_ids, k)

    rank = first_relevant_rank(retrieved_ids, relevant_ids)
    metrics["first_relevant_rank"] = rank
    metrics["reciprocal_rank"] = (1.0 / rank) if rank else 0.0

    return {
        "id": q["id"],
        "category": q["category"],
        "query": q["query"],
        "relevant_chunk_ids": list(relevant_ids),
        "retrieved_top_k": retrieved_ids,
        "metrics": metrics,
    }


# ---------- Aggregation ----------

def aggregate(per_query, categories_to_score):
    """Compute means over queries. Skips should_refuse from precision/recall/MRR."""
    scored = [pq for pq in per_query if pq["category"] in categories_to_score]

    agg = {}
    for k in K_VALUES:
        agg[f"precision_at_{k}"] = mean(
            pq["metrics"][f"precision_at_{k}"] for pq in scored
        ) if scored else 0.0
        recall_values = [
            pq["metrics"][f"recall_at_{k}"] for pq in scored
            if pq["metrics"][f"recall_at_{k}"] is not None
        ]
        agg[f"recall_at_{k}"] = mean(recall_values) if recall_values else 0.0

    agg["mrr"] = mean(pq["metrics"]["reciprocal_rank"] for pq in scored) if scored else 0.0
    agg["n_queries"] = len(scored)
    return agg


def per_category(per_query, categories_to_score):
    """Same aggregation, grouped by category."""
    by_cat = {}
    for cat in sorted({pq["category"] for pq in per_query if pq["category"] in categories_to_score}):
        cat_queries = [pq for pq in per_query if pq["category"] == cat]
        by_cat[cat] = aggregate(cat_queries, categories_to_score)
    return by_cat


# ---------- Report writers ----------

def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_markdown(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    overall = payload["overall"]
    by_cat = payload["per_category"]
    refuse = payload["should_refuse_queries"]

    def fmt(x, nd=3):
        return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "—"

    lines = []
    lines.append(f"# Retrieval Evaluation\n")
    lines.append(f"Run on {payload['run_at']}  ")
    lines.append(f"Embedding model: `{payload['embedding_model']}`  ")
    lines.append(f"Queries scored (excludes should_refuse): {overall['n_queries']}\n")

    lines.append("## Overall metrics\n")
    lines.append("| Metric | k=5 | k=10 |")
    lines.append("|---|---|---|")
    lines.append(f"| Precision | {fmt(overall['precision_at_5'])} | {fmt(overall['precision_at_10'])} |")
    lines.append(f"| Recall    | {fmt(overall['recall_at_5'])}    | {fmt(overall['recall_at_10'])}    |")
    lines.append(f"| MRR       | {fmt(overall['mrr'])} | — |\n")

    lines.append("## Per-category breakdown\n")
    lines.append("| Category | n | Precision@5 | Recall@5 | Recall@10 | MRR |")
    lines.append("|---|---|---|---|---|---|")
    for cat, m in by_cat.items():
        lines.append(
            f"| {cat} | {m['n_queries']} | {fmt(m['precision_at_5'])} | "
            f"{fmt(m['recall_at_5'])} | {fmt(m['recall_at_10'])} | {fmt(m['mrr'])} |"
        )
    lines.append("")

    lines.append("## Should-refuse queries (guardrail check)\n")
    lines.append("These have no labeled relevant chunks. Retrieval will still return top-K chunks; ")
    lines.append("the actual refusal behavior is tested by the generation layer, not this harness.\n")
    for q in refuse:
        lines.append(f"- **{q['id']}**: {q['query']}")
    lines.append("")

    lines.append("## Per-query detail\n")
    for pq in payload["per_query"]:
        if pq["category"] == "should_refuse":
            continue
        m = pq["metrics"]
        rank = m["first_relevant_rank"] if m["first_relevant_rank"] else "not in top-10"
        lines.append(f"### {pq['id']} ({pq['category']})")
        lines.append(f"*{pq['query']}*")
        lines.append(f"- First relevant chunk rank: `{rank}`")
        lines.append(f"- Precision@5: {fmt(m['precision_at_5'])}, Recall@5: {fmt(m['recall_at_5'])}")
        lines.append(f"- Relevant IDs: `{pq['relevant_chunk_ids']}`\n")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------- Main ----------

def main():
    with open(QUERIES_PATH, encoding="utf-8") as f:
        queries = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(queries)} queries from {QUERIES_PATH}")

    max_k = max(K_VALUES)
    per_query = [evaluate_query(q, max_k) for q in queries]

    categories_to_score = {"parameter_table", "checkbox_table", "domain_terminology", "prose"}
    overall = aggregate(per_query, categories_to_score)
    per_cat = per_category(per_query, categories_to_score)

    payload = {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "embedding_model": "all-MiniLM-L6-v2",
        "k_values": K_VALUES,
        "overall": overall,
        "per_category": per_cat,
        "should_refuse_queries": [
            {"id": pq["id"], "query": pq["query"]}
            for pq in per_query if pq["category"] == "should_refuse"
        ],
        "per_query": per_query,
    }

    write_json(RESULTS_PATH, payload)
    write_markdown(SUMMARY_PATH, payload)

    print(f"\nOverall (scored queries only, n={overall['n_queries']}):")
    print(f"  Precision@5:  {overall['precision_at_5']:.3f}")
    print(f"  Recall@5:     {overall['recall_at_5']:.3f}")
    print(f"  Recall@10:    {overall['recall_at_10']:.3f}")
    print(f"  MRR:          {overall['mrr']:.3f}")
    print(f"\nJSON:     {RESULTS_PATH}")
    print(f"Markdown: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
