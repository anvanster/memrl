# Tempera eval harness

Measure retrieval quality (P@K, R@K, MRR, nDCG@K) against a labeled fixture
so retrieval changes can be A/B'd instead of vibe-checked.

## Layout

```
evals/
├── fixtures/         # labeled query sets (JSONL, one query per line)
│   └── example.jsonl
├── baselines/        # persisted run snapshots, sorted by timestamp
│   └── <YYYYMMDDTHHMMSSZ>-<commit>.json
└── README.md
```

## Fixture format

JSONL, one JSON object per line. Lines starting with `#` and blank lines are skipped.

```jsonc
{
  "id": "q001",                                    // stable query id
  "query": "fix login redirect vulnerability",     // sent to the retriever
  "relevant": [
    {"id": "abc12345", "grade": 3},                // graded relevance 0-3
    {"id": "def67890", "grade": 2}
  ],
  "project": "myproject",                          // optional filter
  "tags": ["security", "auth"],                    // optional, for slicing
  "notes": "open-redirect-class bug"               // optional, labeler notes
}
```

`grade` is a 0-3 relevance grade used for nDCG. Omit it (or set to 3) for
binary relevance. Episode IDs can be full UUIDs or short prefixes — match
whatever your store returns.

## Building your own fixture

1. **List your episodes** to find labeling candidates:
   ```bash
   tempera list --limit 50
   ```
2. **Pick 20-50 queries** that represent the questions you actually ask. Mix:
   - literal-token queries (function names, error codes) — these stress BM25
   - conceptual queries ("how do I X") — these stress vector similarity
   - multi-aspect queries — these stress reranking
3. **For each query, label the top relevant episodes**. Two or three per
   query is enough — exhaustive labeling is not required for P@5/R@5.
4. **Save as `evals/fixtures/<name>.jsonl`** and commit if it doesn't
   contain sensitive content. (Fixtures live in your repo by default.)

## Running

Establish a baseline against the current `main`:

```bash
tempera eval baseline --fixture evals/fixtures/general.jsonl
# → writes evals/baselines/<ts>-<commit>.json
```

After a retrieval change, run and diff:

```bash
tempera eval run --fixture evals/fixtures/general.jsonl
# → prints metrics with green/red deltas vs latest baseline
```

To save the post-change run as a new baseline:

```bash
tempera eval run --fixture evals/fixtures/general.jsonl --save
```

## Metrics

| Metric  | Definition |
|---------|------------|
| P@K     | `|relevant ∩ retrieved[..K]| / K` — fraction of top-K that are relevant |
| R@K     | `|relevant ∩ retrieved[..K]| / |relevant|` — coverage of relevant set |
| MRR     | Mean of `1 / rank_of_first_relevant` across queries |
| nDCG@K  | Normalized discounted cumulative gain, gain `2^grade - 1` |

`K` defaults to 5 (`--k 5`). Standard IR conventions:
- P@K divides by K (penalizes under-retrieval)
- nDCG uses log2 discount

## CI integration

A future GitHub Action will run `tempera eval run` on every PR and fail the
check on a P@5 regression > 2 points (override via PR label).
