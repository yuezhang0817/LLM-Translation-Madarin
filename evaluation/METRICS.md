# Evaluation metrics: what each script computes

This note documents exactly which metric each script in this repository produces, so that the
paper's methodology and results sections describe the same quantities (Reviewer 1, comments 1–2).

## Summary

| Script | Metric reported | Model | Unit of comparison | Compared against |
|---|---|---|---|---|
| `sentiment and semantic analysis.py` → "Semantic Similarity" | Cosine similarity of mean-pooled sentence embeddings (**not BERTScore**) | `all-MiniLM-L6-v2` (Sentence-BERT) | Whole corpus (one score per system per corpus) | Expert translation |
| `semantic similarity compare table/translation_similarity_analysis.py` | Cosine similarity of sentence embeddings, averaged over paragraphs | `all-mpnet-base-v2` (Sentence-BERT) | Paragraph / table row | Other systems pairwise (e.g. Joly–Google, GPT4–GPT4o) |
| `evaluation/compute_mt_metrics.py` (new) | BERTScore P / R / F1 | `roberta-large`, layer 17 | Aligned segment (paragraph or merged paragraphs) | Expert translation |
| `evaluation/compute_mt_metrics.py` (new) | COMET | `Unbabel/wmt22-comet-da` | Aligned segment | Source + expert translation |
| `evaluation/compute_mt_metrics.py --kiwi` (new, optional) | CometKiwi (reference-free) | `Unbabel/wmt22-cometkiwi-da` | Aligned segment | Source only |

## Problems found in the original evaluation

1. **BERTScore was never computed.** The methodology describes BERTScore precision, recall and F1,
   but neither original script calls BERTScore. Both compute a *Sentence-BERT cosine similarity*:
   each text is encoded into one fixed vector and the cosine between vectors is reported. BERTScore is
   a different metric. It greedily matches contextual **token** embeddings between candidate and
   reference, which produces precision (candidate tokens covered by the reference), recall (reference
   tokens covered by the candidate) and their harmonic mean F1. The "semantic similarity" values in the
   results are therefore neither BERTScore F1 nor any BERTScore component. This is the inconsistency
   Reviewer 1 noticed.
2. **The correlation argument does not transfer.** The cited evidence that BERTScore correlates well
   with human judgements on zh→en (Zhang et al., 2020, WMT) is about BERTScore. It does not support
   document-level SBERT cosine.
3. **Granularity.** In the main script the whole corpus of a system (e.g. all six news articles) is
   split into 512-*character* windows. The window embeddings are averaged into one vector and compared
   with one cosine, so each system gets a single number per corpus, with no variance, no significance
   test and little power to discriminate (mean-pooled document vectors of on-topic English text are
   always close).
4. **Header rows included as text.** `load_data` reads the CSVs with `header=None`, so each file's
   label row (e.g. `gpt-4_Translation`, `English_version_news1`) is embedded as translation text.
5. **Two different models.** The main results use `all-MiniLM-L6-v2` and the comparison tables use
   `all-mpnet-base-v2`. The comparison tables also measure agreement *between systems* rather than
   against the expert reference.
6. **Ad-hoc grade bands.** The "Excellent ≥ 0.9 / Very good ≥ 0.8 …" labels applied to cosine scores
   have no calibration against human judgement and should not be presented as quality grades.

## The new evaluation (`evaluation/compute_mt_metrics.py`)

**Segmentation and alignment.** Every machine translation is paragraph-aligned 1:1 with the Chinese
source. The expert translations split or merge paragraphs differently in 7 of the 13 documents. Expert
paragraphs are aligned to source paragraphs with a monotonic dynamic-programming aligner over LaBSE
cross-lingual embeddings, allowing 1–3 : 1–3 merges (similar to vecalign). Each extra merged paragraph costs 0.3. Without that penalty the aligner over-merges, because longer concatenations always embed closer together. With it, documents whose paragraph counts already match align fully 1:1. Segments with alignment cosine below 0.4 are printed for manual checking. These are mostly verse passages in *A Dream of Red Mansions*, where Joly restructures the poems. Each aligned unit is one
evaluation segment, and a system's output for that segment is the concatenation of its paragraphs over
the same source span. Every alignment is saved to `results/alignments/` for manual verification.

**Metrics.** For every segment the script computes:

- BERTScore P, R and F1 against the expert reference (`bert-score`, `roberta-large`, layer 17,
  without baseline rescaling, so scores fall roughly in 0.8–1.0). The exact configuration hash is
  printed so it can be reported.
- COMET (`Unbabel/wmt22-comet-da`) from source, output and expert reference. Scores are roughly in
  0–1; higher is better.
- Optionally, CometKiwi (`Unbabel/wmt22-cometkiwi-da`), a reference-free quality-estimation score.
  This matters for literary text, where an expert translation (Joly, Goldblatt) is one valid rendering
  among many and can be deliberately free.

**Aggregation.** System scores are the mean over segments, with 95% bootstrap confidence intervals
(1,000 resamples, seed 0). Segment-level Pearson, Spearman and Kendall correlations between metrics are
written to `metric_correlations.csv`.

**Known limitation.** Both encoders read at most 512 tokens per input, so longer segments are
truncated. The script prints how many segments this affects (see `results/run.log`). Report that count in the paper.

## Results of the first full run (Oct 2026)

`results/system_scores.csv` contains the full table with confidence intervals. The run covered 409 source
paragraphs in 1,612 segment × system pairs (news 44 segments, Red Mansions 105, Red Sorghum 254).

| Corpus | System | BERTScore F1 | COMET |
|---|---|---|---|
| News | GPT-4 | 0.941 | 0.832 |
| News | GPT-4o | **0.946** | **0.839** |
| News | DeepSeek | 0.938 | 0.821 |
| News | Google | 0.939 | 0.830 |
| Red Mansions | GPT-4 | 0.865 | 0.693 |
| Red Mansions | GPT-4o | **0.867** | **0.713** |
| Red Mansions | DeepSeek | 0.864 | 0.707 |
| Red Mansions | Google | 0.853 | 0.615 |
| Red Sorghum | GPT-4 | 0.909 | 0.750 |
| Red Sorghum | GPT-4o | 0.912 | 0.762 |
| Red Sorghum | DeepSeek | **0.917** | **0.771** |
| Red Sorghum | Google | 0.902 | 0.710 |

- The two metrics agree on the best system in every corpus. COMET separates the systems much more
  clearly: for example, Google trails the LLMs by about 0.09 COMET on Red Mansions but only about
  0.012 BERTScore F1.
- Segment-level agreement across all data: BERTScore F1 vs COMET Pearson 0.71, Spearman 0.71, Kendall
  0.52. They are related but not redundant, which supports reporting both.
- Truncation at 512 tokens: 21 of 1,612 BERTScore candidates (and 21 references) and 47 COMET inputs.
- One alignment is flagged as low-confidence (red_mansions_1, segment 40, a verse passage).

## Data-quality problems found in the system outputs

These affect the **original paper results as well** (semantic and sentiment scores), not only the
new metrics.

| File / segment | System | Problem | Type |
|---|---|---|---|
| news 2, last paragraph | DeepSeek | Text is `[Translation Failed]`, written by `deepseek_translate.ipynb` on an API exception | Pipeline error: re-translate |
| news 2, paragraph 3 | DeepSeek | Good translation followed by about 3,600 characters of repeated "办好" | Degenerate generation (repetition loop) |
| red_sorghum 4, segment 120 | GPT-4o | "I'm sorry, I can't assist with that request." (violent passage) | Safety refusal |
| red_sorghum 3, segment 14 | Google | "Woooo…" repetition loop on an onomatopoeic line | Degenerate generation |

The `[Translation Failed]` row is not the system's output and should be re-translated before the
results are final. The refusal and the repetition loops are genuine system behaviour, so keep them
and discuss them as qualitative failure modes (they are exactly what segment-level COMET exposes).

## Suggested wording for the paper

> **Methodology.** We evaluate each system against the expert translation with two complementary
> automatic metrics. *BERTScore* (Zhang et al., 2020) matches contextual token embeddings from
> RoBERTa-large (layer 17) between a candidate and its reference and reports precision, recall and
> F1. We use F1 as the summary score. *COMET* (Rei et al., 2022; model `wmt22-comet-da`) is a neural
> metric trained on human direct-assessment judgements that conditions on the source sentence as well
> as the reference. [Optional: we also report the reference-free CometKiwi score, which avoids
> penalising legitimate departures from a single literary reference.] Because the expert translations
> do not always preserve the source paragraph boundaries, we align source and reference paragraphs with
> a monotonic LaBSE-based aligner and score each aligned segment. System scores are segment means with
> 95% bootstrap confidence intervals.

> **Results.** Report BERTScore P, R and F1 and COMET for every corpus × system, from
> `results/system_scores.csv`. If the earlier Sentence-BERT cosine similarity is kept, name it
> explicitly as "Sentence-BERT cosine similarity (all-MiniLM-L6-v2)" and do not call it BERTScore. The
> sentiment-weighted "Overall Score" should state which semantic metric it uses.

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate
pip install "unbabel-comet>=2.2" bert-score sentence-transformers pandas scipy
python evaluation/compute_mt_metrics.py          # add --kiwi after `huggingface-cli login`
```

The first run downloads about 5 GB of models (LaBSE, roberta-large, wmt22-comet-da).
