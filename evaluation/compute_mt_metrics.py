"""
Reference-based and reference-free MT evaluation for the Mandarin -> English corpora.

Metrics (all computed per aligned segment, then averaged per system and corpus):
  * BERTScore precision / recall / F1  (Zhang et al., 2020; roberta-large, layer 17,
    the bert-score default for English).  Candidate = system output, reference = expert.
  * COMET  (Rei et al., 2022; Unbabel/wmt22-comet-da).  Uses source + output + expert reference.
  * CometKiwi (optional, --kiwi; Unbabel/wmt22-cometkiwi-da).  Reference-free: source + output.
    The model is gated on Hugging Face: accept its licence and run `huggingface-cli login` first.

Segmentation / alignment
  The machine translations are paragraph-aligned 1:1 with the Chinese source, but the expert
  translations often split or merge paragraphs differently.  Expert paragraphs are therefore
  aligned to source paragraphs with a monotonic dynamic-programming aligner over LaBSE
  cross-lingual embeddings (allowing 1-3 : 1-3 merges with a per-merge penalty, in the spirit of vecalign).  Each aligned
  unit (a span of source paragraphs + the matching expert paragraphs) is one evaluation segment;
  system outputs for that unit are the concatenation of their paragraphs over the same source span.
  Alignments are written to results/alignments/ so they can be checked by hand.

Outputs (in --out, default evaluation/results):
  segment_scores.csv     one row per corpus x document x segment x system
  system_scores.csv      mean and 95% bootstrap CI per corpus x system x metric
  metric_correlations.csv  segment-level Pearson / Spearman / Kendall between metrics
  alignments/<doc>.csv   source <-> expert alignment used for every document

Usage
  python evaluation/compute_mt_metrics.py                # BERTScore + COMET
  python evaluation/compute_mt_metrics.py --kiwi         # + CometKiwi
  python evaluation/compute_mt_metrics.py --limit-docs 1 # quick smoke test
"""
import argparse
import csv
import os
from itertools import combinations

import numpy as np
import pandas as pd
import torch
from scipy import stats

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, 'data set')

SYSTEMS = ['GPT-4', 'GPT-4o', 'DeepSeek', 'Google Translate']

# Paths relative to DATA_DIR; {i} is the document / chapter number.
CORPORA = {
    'news': {
        'docs': range(1, 7),
        'source': 'Madarin original/News_Chinese_Version/Chinese_version_news{i}.csv',
        'reference': 'Human/News_English_Version/English_version_news{i}.csv',
        'systems': {
            'GPT-4': 'gpt4/gpt4_news/Chinese_version_news{i}_gpt-4_translated.csv',
            'GPT-4o': 'gpt4o/gpt4o_news/Chinese_version_news{i}_gpt-4o_translated.csv',
            'DeepSeek': 'deepseek/deepseek_news/Chinese_version_news{i}_deepseek_chat_translated.csv',
            'Google Translate': 'google/google_news/Google_news{i}.csv',
        },
    },
    'red_mansions': {
        'docs': range(1, 4),
        'source': 'Madarin original/Chinese_A Dream of Red Mansions/Red_Mansions_Chapter{i}_Chinese.csv',
        'reference': 'Human/Red_mansion_expert_merge/Red_Mansions_Chapter{i}_Bencraft_Joly.csv',
        'systems': {
            'GPT-4': 'gpt4/gpt4_Red_Mansions/Red_Mansions_Chapter{i}.csv',
            'GPT-4o': 'gpt4o/gpt4o_Red_Mansions/Red_Mansions_Chapter{i}.csv',
            'DeepSeek': 'deepseek/deepseek_Red_Mansions/Red_Mansions_Chapter{i}_Chinese_deepseek_chat_translated.csv',
            'Google Translate': 'google/google_Red_Mansions/Red_Mansions_Chapter{i}.csv',
        },
    },
    'red_sorghum': {
        'docs': range(1, 5),
        'source': 'Madarin original/Chinese_Red sorghum/Red_sorghum_Chapter{i}_Chinese.csv',
        'reference': 'Human/Red_sorghum_expert_merge/Red_sorghum_Chapter{i}_Howard_Goldblatt.csv',
        'systems': {
            'GPT-4': 'gpt4/gpt4_Red_sorghum/Red_sorghum_Chapter{i}.csv',
            'GPT-4o': 'gpt4o/gpt4o_Red_sorghum/Red_sorghum_Chapter{i}.csv',
            'DeepSeek': 'deepseek/deepseek_Red_sorghum/Red_sorghum_Chapter{i}_Chinese_deepseek_chat_translated.csv',
            'Google Translate': 'google/google_Red_sorghum/Red_sorghum_Chapter{i}.csv',
        },
    },
}

ALIGN_MODEL = 'sentence-transformers/LaBSE'
BERTSCORE_MODEL = 'roberta-large'
COMET_MODEL = 'Unbabel/wmt22-comet-da'
KIWI_MODEL = 'Unbabel/wmt22-cometkiwi-da'
MAX_GROUP = 3            # max paragraphs merged on either side of one alignment unit
MERGE_PENALTY = 0.3      # cost added per extra paragraph in a merge; lower values over-merge because
                         # longer concatenations always embed closer together
LOW_ALIGN_COSINE = 0.4   # segments below this are listed for manual checking
N_BOOTSTRAP = 1000


# ==================== 数据读取 ====================
def read_paragraphs(path):
    """Read a one-column CSV; the first row is a file label (e.g. 'gpt-4_Translation'), not text."""
    for encoding in ['utf-8-sig', 'gb18030', 'latin-1']:
        try:
            with open(path, encoding=encoding, newline='') as fh:
                rows = list(csv.reader(fh))
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f'Could not decode {path}')
    paragraphs = [' '.join(cell.strip() for cell in row if cell.strip()) for row in rows[1:]]
    return [p for p in paragraphs if p]


def device():
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


# ==================== 段落对齐 ====================
def align_paragraphs(src, ref, encoder):
    """Monotonic n:m alignment of source paragraphs to reference paragraphs.

    Returns a list of ((src_start, src_end), (ref_start, ref_end), cosine) with half-open spans
    that together cover every paragraph on both sides exactly once.
    """
    def span_texts(paras, sep):
        spans = [(i, i + k) for i in range(len(paras)) for k in range(1, MAX_GROUP + 1) if i + k <= len(paras)]
        return spans, [sep.join(paras[a:b]) for a, b in spans]

    src_spans, src_texts = span_texts(src, '')
    ref_spans, ref_texts = span_texts(ref, ' ')
    src_emb = encoder.encode(src_texts, convert_to_tensor=True, normalize_embeddings=True)
    ref_emb = encoder.encode(ref_texts, convert_to_tensor=True, normalize_embeddings=True)
    sims = (src_emb @ ref_emb.T).cpu().numpy()
    src_index = {s: k for k, s in enumerate(src_spans)}
    ref_index = {s: k for k, s in enumerate(ref_spans)}

    n, m = len(src), len(ref)
    cost = np.full((n + 1, m + 1), np.inf)
    back = {}
    cost[0, 0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            if not np.isfinite(cost[i, j]):
                continue
            for di in range(1, MAX_GROUP + 1):
                for dj in range(1, MAX_GROUP + 1):
                    if i + di > n or j + dj > m:
                        continue
                    sim = sims[src_index[(i, i + di)], ref_index[(j, j + dj)]]
                    c = cost[i, j] + (1 - sim) + MERGE_PENALTY * (di + dj - 2)
                    if c < cost[i + di, j + dj]:
                        cost[i + di, j + dj] = c
                        back[(i + di, j + dj)] = (i, j, sim)
    if not np.isfinite(cost[n, m]):
        raise ValueError(f'No alignment found for {n} source / {m} reference paragraphs; raise MAX_GROUP')

    alignment = []
    i, j = n, m
    while (i, j) != (0, 0):
        pi, pj, sim = back[(i, j)]
        alignment.append(((pi, i), (pj, j), float(sim)))
        i, j = pi, pj
    return alignment[::-1]


def build_segments(limit_docs, encoder, out_dir):
    """Load all corpora and return one row per (corpus, doc, segment, system)."""
    os.makedirs(os.path.join(out_dir, 'alignments'), exist_ok=True)
    rows = []
    for corpus, cfg in CORPORA.items():
        for doc in list(cfg['docs'])[:limit_docs]:
            src = read_paragraphs(os.path.join(DATA_DIR, cfg['source'].format(i=doc)))
            ref = read_paragraphs(os.path.join(DATA_DIR, cfg['reference'].format(i=doc)))
            hyps = {}
            for system, pattern in cfg['systems'].items():
                hyps[system] = read_paragraphs(os.path.join(DATA_DIR, pattern.format(i=doc)))
                if len(hyps[system]) != len(src):
                    raise ValueError(f'{corpus} doc {doc}: {system} has {len(hyps[system])} paragraphs, '
                                     f'source has {len(src)}; system outputs must be 1:1 with source')

            alignment = align_paragraphs(src, ref, encoder)
            doc_id = f'{corpus}_{doc}'
            pd.DataFrame([{
                'segment': k, 'source_paragraphs': f'{s0 + 1}-{s1}', 'expert_paragraphs': f'{r0 + 1}-{r1}',
                'labse_cosine': round(sim, 4), 'source': ''.join(src[s0:s1]), 'expert': ' '.join(ref[r0:r1]),
            } for k, ((s0, s1), (r0, r1), sim) in enumerate(alignment)]).to_csv(
                os.path.join(out_dir, 'alignments', f'{doc_id}.csv'), index=False, encoding='utf-8-sig')
            print(f'{doc_id}: {len(src)} source / {len(ref)} expert paragraphs -> {len(alignment)} segments '
                  f'(min LaBSE cosine {min(a[2] for a in alignment):.3f})')
            for k, ((s0, s1), (r0, r1), sim) in enumerate(alignment):
                if sim < LOW_ALIGN_COSINE:
                    print(f'  check segment {k}: source paragraphs {s0 + 1}-{s1} <-> expert {r0 + 1}-{r1} '
                          f'(cosine {sim:.3f})')

            for k, ((s0, s1), (r0, r1), sim) in enumerate(alignment):
                for system in cfg['systems']:
                    rows.append({
                        'corpus': corpus, 'doc': doc_id, 'segment': k, 'system': system,
                        'align_cosine': sim, 'src': ''.join(src[s0:s1]),
                        'mt': ' '.join(hyps[system][s0:s1]), 'ref': ' '.join(ref[r0:r1]),
                    })
    return pd.DataFrame(rows)


# ==================== 指标计算 ====================
def count_truncated(texts, tokenizer, limit=510):
    return sum(len(tokenizer(t, add_special_tokens=False)['input_ids']) > limit for t in texts)


def add_bertscore(df):
    from bert_score import BERTScorer
    scorer = BERTScorer(model_type=BERTSCORE_MODEL, lang='en', device=device())
    print(f'BERTScore ({BERTSCORE_MODEL}): {count_truncated(df.mt, scorer._tokenizer)} outputs and '
          f'{count_truncated(df.ref.drop_duplicates(), scorer._tokenizer)} references exceed 510 tokens '
          f'and are truncated')
    p, r, f = scorer.score(df.mt.tolist(), df.ref.tolist(), batch_size=16)
    df['bertscore_p'], df['bertscore_r'], df['bertscore_f1'] = p.numpy(), r.numpy(), f.numpy()
    print(f'BERTScore hash: {scorer.hash}')


def add_comet(df, model_name, column, use_reference):
    from comet import download_model, load_from_checkpoint
    model = load_from_checkpoint(download_model(model_name))
    tokenizer = model.encoder.tokenizer
    print(f'{model_name}: {count_truncated(df.mt, tokenizer)} outputs and '
          f'{count_truncated(df.src.drop_duplicates(), tokenizer)} sources exceed 510 tokens and are truncated')
    data = [{'src': s, 'mt': m, 'ref': r} if use_reference else {'src': s, 'mt': m}
            for s, m, r in zip(df.src, df.mt, df.ref)]
    # COMET always uses a fork context on Apple machines, which needs workers, so run there on MPS.
    use_mps = not torch.cuda.is_available() and torch.backends.mps.is_available()
    gpus = 1 if torch.cuda.is_available() or use_mps else 0
    accelerator = 'mps' if use_mps else 'auto'
    output = model.predict(data, batch_size=8, gpus=gpus, accelerator=accelerator, progress_bar=True)
    df[column] = output.scores


# ==================== 汇总 ====================
def bootstrap_ci(values, rng):
    values = np.asarray(values)
    means = rng.choice(values, size=(N_BOOTSTRAP, len(values)), replace=True).mean(axis=1)
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


def summarise(df, metrics, out_dir):
    rng = np.random.default_rng(0)
    rows = []
    for (corpus, system), group in df.groupby(['corpus', 'system'], sort=False):
        row = {'corpus': corpus, 'system': system, 'n_segments': len(group)}
        for metric in metrics:
            lo, hi = bootstrap_ci(group[metric], rng)
            row[metric] = group[metric].mean()
            row[f'{metric}_ci_low'], row[f'{metric}_ci_high'] = lo, hi
        rows.append(row)
    system_df = pd.DataFrame(rows)
    system_df.to_csv(os.path.join(out_dir, 'system_scores.csv'), index=False, float_format='%.4f')

    corr_rows = []
    for corpus, group in [('all', df)] + list(df.groupby('corpus', sort=False)):
        for a, b in combinations(metrics, 2):
            corr_rows.append({
                'corpus': corpus, 'metric_a': a, 'metric_b': b, 'n': len(group),
                'pearson': stats.pearsonr(group[a], group[b])[0],
                'spearman': stats.spearmanr(group[a], group[b])[0],
                'kendall_tau': stats.kendalltau(group[a], group[b])[0],
            })
    pd.DataFrame(corr_rows).to_csv(os.path.join(out_dir, 'metric_correlations.csv'),
                                   index=False, float_format='%.4f')

    print('\n=== System-level means (95% bootstrap CI) ===')
    for corpus, group in system_df.groupby('corpus', sort=False):
        print(f'\n[{corpus}] n_segments={group.n_segments.iloc[0]}')
        for _, row in group.iterrows():
            cells = [f"{m}={row[m]:.4f} [{row[f'{m}_ci_low']:.4f},{row[f'{m}_ci_high']:.4f}]" for m in metrics]
            print(f"  {row.system:<17} " + '  '.join(cells))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', default=os.path.join(REPO_ROOT, 'evaluation', 'results'))
    parser.add_argument('--kiwi', action='store_true', help='also compute reference-free CometKiwi (gated model)')
    parser.add_argument('--limit-docs', type=int, default=None, help='only the first N documents of each corpus')
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(ALIGN_MODEL, device=device())
    df = build_segments(args.limit_docs, encoder, args.out)
    del encoder

    metrics = ['bertscore_p', 'bertscore_r', 'bertscore_f1', 'comet']
    add_bertscore(df)
    add_comet(df, COMET_MODEL, 'comet', use_reference=True)
    if args.kiwi:
        add_comet(df, KIWI_MODEL, 'cometkiwi', use_reference=False)
        metrics.append('cometkiwi')

    df.to_csv(os.path.join(args.out, 'segment_scores.csv'), index=False, encoding='utf-8-sig', float_format='%.4f')
    summarise(df, metrics, args.out)
    print(f'\nResults written to {args.out}')


if __name__ == '__main__':
    main()
