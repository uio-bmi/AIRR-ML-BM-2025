"""
Phase-2 Sequencing Depth Analysis
===================================

Compares each target test dataset (where T1D+ and T1D- examples differ in
sequencing depth) against the cohort-1 reference dataset (q6_t1d_test.csv,
which maps to test_dataset_30 inside submission files, and has matched
sequencing depth between T1D+ and T1D-).

For each target dataset the script produces three figures and companion TSVs:

    sequencing_depth_auc_comparison.png / .tsv
        Paired dot plot: AUC on cohort-1 (depth-matched) vs AUC on the
        target cohort (depth-discordant).  AUC delta = AUC_target − AUC_c1.

    ctrl_score_distributions.png / .tsv
        Boxplot grid (one panel per method), panels ordered by ascending
        T1D- std in the target cohort.  Four groups per panel:
          cohort-1 T1D-  (viridis 0.20)
          target   T1D-  (viridis 0.45)
          cohort-1 T1D+  (viridis 0.70)
          target   T1D+  (viridis 0.95)

    ctrl_mean_shift.png / .tsv
        Grouped bar chart: T1D- mean score shift
        (mean_c1_T1D- − mean_target_T1D-) and AUC delta per method.

Usage
-----
    python assessment/phase2_confounder_analysis.py \\
        --cohort1_metadata   /path/to/q6_t1d_test.csv \\
        --target_metadata    /path/to/test_dataset_N_metadata.csv \\
        --submissions_dir    /path/to/phase_2_submission_files/ \\
        --target_dataset_id  test_dataset_N \\
        --output_dir         /path/for/outputs/test_dataset_N/

Notes
-----
- cohort-1 subjects are matched to submission IDs via the 'filename' column
  (strip .tsv extension) → test_dataset_30 rows in submission files.
- target subjects are matched via the 'repertoire_id' column.
- Every CSV under submissions_dir whose stem matches the label map is loaded
  as a separate method.  Files not in the label map are silently skipped.
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
# Label map: submission filename stem → display label
# ---------------------------------------------------------------------------

LABEL_MAP = {
    'rank_2':              'R2',
    'rank_3':              'R3',
    'rank_7':              'R7',
    'rank_5':              'R5',
    'rank_10':             'R10',
    'rank_1_ds2':          'R1-M1',
    'rank_9':              'R9',
    'rank_4_emerson':      'R4-M1',
    'rank_6':              'R6',
    'rank_8_public_df':    'R8-M1',
    'rank_8_kmer':         'R8-M2',
    'rank_1_ds7':          'R1-M2',
    'rank_4_kmer':         'R4-M2',
    'rank_4_multikmer':    'R4-M3',
    'rank_1_ds5':          'R1-M3',
    'rank_8_embedding_df': 'R8-M3',
    'rank_1_ds8':          'R1-M4',
    'rank_1_ds4':          'R1-M5',
}

METHOD_ORDER = sorted(LABEL_MAP.values())

MIN_POSITIVE_FOR_AUC = 5
COHORT1_DATASET_ID   = 'test_dataset_30'

# Viridis-D truncated colormap starting at 0.2 (matches R scale_fill_viridis option='D' begin=0.2)
_VIRIDIS_TRUNC = mcolors.LinearSegmentedColormap.from_list(
    'viridis_trunc', plt.cm.viridis(np.linspace(0.2, 1.0, 256)))

# 4 colourblind-safe colours sampled from viridis-D
_V_COLS = [plt.cm.viridis(x) for x in np.linspace(0.2, 0.95, 4)]
COLOR_C1_NEG  = _V_COLS[0]   # cohort-1   T1D-
COLOR_TGT_NEG = _V_COLS[1]   # target     T1D-
COLOR_C1_POS  = _V_COLS[2]   # cohort-1   T1D+
COLOR_TGT_POS = _V_COLS[3]   # target     T1D+


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_cohort1_metadata(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['label_positive'] = df['label_positive'].map(
        {True: 1, False: 0, 'True': 1, 'False': 0}
    )
    df['subject_id'] = df['filename'].str.replace(r'\.tsv(\.gz)?$', '', regex=True)
    return df


def load_target_metadata(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['label_positive'] = df['label_positive'].map(
        {True: 1, False: 0, 'True': 1, 'False': 0}
    )
    df = df.rename(columns={'repertoire_id': 'subject_id'})
    return df


def load_submissions(submissions_dir: str,
                     cohort1_dataset_id: str,
                     target_dataset_id: str) -> dict:
    submissions = {}
    for stem, label in LABEL_MAP.items():
        path = os.path.join(submissions_dir, f'{stem}.csv')
        if not os.path.exists(path):
            print(f'  WARNING: {path} not found, skipping {label}.')
            continue
        df = pd.read_csv(path)
        df = df[df['dataset'].isin([cohort1_dataset_id, target_dataset_id])].copy()
        if df.empty:
            print(f'  WARNING: {label} has no rows for '
                  f'{cohort1_dataset_id} or {target_dataset_id}, skipping.')
            continue
        submissions[label] = df
    return submissions


# ---------------------------------------------------------------------------
# Merge helper
# ---------------------------------------------------------------------------

def _merge(metadata: pd.DataFrame,
           submission: pd.DataFrame,
           dataset_id: str) -> pd.DataFrame:
    sub = (submission[submission['dataset'] == dataset_id]
           [['ID', 'label_positive_probability']]
           .copy()
           .rename(columns={'ID': 'subject_id'}))
    merged = metadata.merge(sub, on='subject_id', how='inner')
    if len(merged) != len(metadata):
        print(f'    WARNING: {dataset_id} merge: metadata {len(metadata)} rows, '
              f'matched {len(merged)}.')
    return merged


# ---------------------------------------------------------------------------
# Core metric
# ---------------------------------------------------------------------------

def _auc(y_true, y_score) -> float:
    y_true  = np.asarray(y_true,  dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_score))
    y_true, y_score = y_true[mask], y_score[mask]
    if y_true.sum() < MIN_POSITIVE_FOR_AUC or len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_score))


# ---------------------------------------------------------------------------
# Analysis 1: Overall AUC — depth-discordant vs depth-matched
# ---------------------------------------------------------------------------

def compute_overall_aucs(meta_c1: pd.DataFrame,
                         meta_target: pd.DataFrame,
                         submissions: dict,
                         target_dataset_id: str) -> pd.DataFrame:
    records = []
    for label in METHOD_ORDER:
        if label not in submissions:
            continue
        sub = submissions[label]
        mc1 = _merge(meta_c1,     sub, COHORT1_DATASET_ID)
        mt  = _merge(meta_target, sub, target_dataset_id)
        auc_c1     = _auc(mc1['label_positive'], mc1['label_positive_probability'])
        auc_target = _auc(mt['label_positive'],  mt['label_positive_probability'])
        delta = (auc_target - auc_c1) if not (np.isnan(auc_c1) or
                                               np.isnan(auc_target)) else np.nan
        records.append({'method': label,
                        'AUC_cohort1': auc_c1,
                        'AUC_target':  auc_target,
                        'delta':       delta})
    return pd.DataFrame(records).set_index('method')


def plot_sequencing_depth_auc_comparison(auc_table: pd.DataFrame,
                                         target_dataset_id: str,
                                         output_dir: str):
    """
    Paired dot plot: cohort-1 AUC vs target AUC, connected by a grey line.
    Methods ordered by cohort-1 AUC descending.
    """
    df = auc_table.dropna(subset=['AUC_cohort1', 'AUC_target']).copy()
    df = df.sort_values('AUC_cohort1', ascending=False)

    color_c1  = plt.cm.viridis(0.75)
    color_tgt = plt.cm.viridis(0.2)

    fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.38)))

    for i, (label, row) in enumerate(df.iterrows()):
        ax.plot([row['AUC_cohort1'], row['AUC_target']], [i, i],
                color='grey', linewidth=1.2, zorder=1)
        ax.scatter(row['AUC_cohort1'], i, color=color_c1,  zorder=2, s=60)
        ax.scatter(row['AUC_target'],  i, color=color_tgt, zorder=2, s=60)

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df.index, fontsize=9)
    ax.set_xlabel('ROC-AUC', fontsize=11)
    ax.set_xlim(0.4, 1.02)
    ax.axvline(0.5, color='black', linewidth=0.8, linestyle='--', alpha=0.4)
    ax.set_title(
        f'AUC: depth-matched cohort (cohort-1) vs {target_dataset_id} (depth-discordant)',
        fontsize=10)

    patch_c1  = mpatches.Patch(color=color_c1,  label='depth-matched cohort (cohort-1)')
    patch_tgt = mpatches.Patch(color=color_tgt, label=f'{target_dataset_id} (depth-discordant)')
    ax.legend(handles=[patch_c1, patch_tgt], fontsize=9, loc='lower right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out = os.path.join(output_dir, 'sequencing_depth_auc_comparison.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out}')


# ---------------------------------------------------------------------------
# Analysis A: Score distributions
# ---------------------------------------------------------------------------

def compute_score_stats(meta_c1: pd.DataFrame,
                        meta_target: pd.DataFrame,
                        submissions: dict,
                        target_dataset_id: str) -> pd.DataFrame:
    records = []
    for label in METHOD_ORDER:
        if label not in submissions:
            continue
        sub = submissions[label]
        mc1 = _merge(meta_c1,     sub, COHORT1_DATASET_ID)
        mt  = _merge(meta_target, sub, target_dataset_id)

        neg_c1  = mc1[mc1['label_positive'] == 0]['label_positive_probability'].values
        neg_tgt = mt[mt['label_positive']  == 0]['label_positive_probability'].values
        pos_c1  = mc1[mc1['label_positive'] == 1]['label_positive_probability'].values
        pos_tgt = mt[mt['label_positive']  == 1]['label_positive_probability'].values

        ks_stat, ks_p = scipy_stats.ks_2samp(neg_c1, neg_tgt)
        records.append({
            'method':          label,
            'neg_c1_mean':     neg_c1.mean(),
            'neg_c1_std':      neg_c1.std(),
            'neg_c1_median':   np.median(neg_c1),
            'neg_tgt_mean':    neg_tgt.mean(),
            'neg_tgt_std':     neg_tgt.std(),
            'neg_tgt_median':  np.median(neg_tgt),
            'pos_c1_mean':     pos_c1.mean(),
            'pos_c1_std':      pos_c1.std(),
            'pos_c1_median':   np.median(pos_c1),
            'pos_tgt_mean':    pos_tgt.mean(),
            'pos_tgt_std':     pos_tgt.std(),
            'pos_tgt_median':  np.median(pos_tgt),
            'ks_stat':         ks_stat,
            'ks_pvalue':       ks_p,
        })
    return pd.DataFrame(records).set_index('method')


def plot_ctrl_score_distributions(meta_c1: pd.DataFrame,
                                   meta_target: pd.DataFrame,
                                   submissions: dict,
                                   target_dataset_id: str,
                                   output_dir: str):
    """
    Grid of boxplots (one panel per method), panels ordered by ascending
    T1D- std in the target (depth-discordant) cohort.
    Four groups per panel:
      cohort-1 T1D-  (viridis 0.20)
      target   T1D-  (viridis 0.45)
      cohort-1 T1D+  (viridis 0.70)
      target   T1D+  (viridis 0.95)
    """
    # Collect data for each method and compute ordering key
    method_data = {}
    for label in METHOD_ORDER:
        if label not in submissions:
            continue
        sub = submissions[label]
        mc1 = _merge(meta_c1,     sub, COHORT1_DATASET_ID)
        mt  = _merge(meta_target, sub, target_dataset_id)
        neg_c1  = mc1[mc1['label_positive'] == 0]['label_positive_probability'].values
        neg_tgt = mt[mt['label_positive']  == 0]['label_positive_probability'].values
        pos_c1  = mc1[mc1['label_positive'] == 1]['label_positive_probability'].values
        pos_tgt = mt[mt['label_positive']  == 1]['label_positive_probability'].values
        method_data[label] = {
            'neg_c1':      neg_c1,
            'neg_tgt':     neg_tgt,
            'pos_c1':      pos_c1,
            'pos_tgt':     pos_tgt,
            'neg_tgt_std': neg_tgt.std(),
        }

    # Sort by ascending T1D- std in target cohort
    sorted_labels = sorted(method_data.keys(), key=lambda l: method_data[l]['neg_tgt_std'])

    n = len(sorted_labels)
    ncols = 5
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4, nrows * 4),
                             sharey=True)
    axes_flat = np.array(axes).flatten()

    group_labels = ['cohort-1\nT1D\u2212', f'target\nT1D\u2212',
                    'cohort-1\nT1D+',      f'target\nT1D+']
    colors = [COLOR_C1_NEG, COLOR_TGT_NEG, COLOR_C1_POS, COLOR_TGT_POS]

    for idx, label in enumerate(sorted_labels):
        ax = axes_flat[idx]
        d = method_data[label]
        bp = ax.boxplot([d['neg_c1'], d['neg_tgt'], d['pos_c1'], d['pos_tgt']],
                        patch_artist=True, widths=0.5,
                        medianprops=dict(color='black', linewidth=1.5),
                        whiskerprops=dict(linewidth=0.8),
                        capprops=dict(linewidth=0.8),
                        flierprops=dict(marker='o', markersize=2,
                                        alpha=0.4, linestyle='none'))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        ax.set_xticks([1, 2, 3, 4])
        ax.set_xticklabels(group_labels, fontsize=6.5)
        ax.set_title(f'{label}\nT1D\u2212 std$_{{tgt}}$={d["neg_tgt_std"]:.3f}',
                     fontsize=8, pad=3)
        ax.set_ylabel('Predicted score' if idx % ncols == 0 else '', fontsize=8)
        ax.tick_params(axis='y', labelsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    for idx in range(len(sorted_labels), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    patches = [
        mpatches.Patch(color=COLOR_C1_NEG,  alpha=0.75, label='depth-matched cohort (cohort-1) T1D\u2212'),
        mpatches.Patch(color=COLOR_TGT_NEG, alpha=0.75, label=f'{target_dataset_id} (depth-discordant) T1D\u2212'),
        mpatches.Patch(color=COLOR_C1_POS,  alpha=0.75, label='depth-matched cohort (cohort-1) T1D+'),
        mpatches.Patch(color=COLOR_TGT_POS, alpha=0.75, label=f'{target_dataset_id} (depth-discordant) T1D+'),
    ]
    fig.legend(handles=patches, fontsize=9,
               loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(
        f'Predicted score distributions: depth-matched cohort (cohort-1) vs {target_dataset_id} (depth-discordant)\n'
        'Panels ordered by ascending T1D\u2212 score std in depth-discordant cohort',
        fontsize=11, fontweight='bold')
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    out = os.path.join(output_dir, 'ctrl_score_distributions.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out}')


# ---------------------------------------------------------------------------
# Analysis B: T1D- mean score shift
# ---------------------------------------------------------------------------

def compute_ctrl_mean_shift(meta_c1: pd.DataFrame,
                             meta_target: pd.DataFrame,
                             submissions: dict,
                             overall_auc_table: pd.DataFrame,
                             target_dataset_id: str) -> pd.DataFrame:
    """
    For each method:
        neg_mean_c1     = mean predicted score for cohort-1 T1D-
        neg_mean_target = mean predicted score for target T1D-
        neg_mean_shift  = neg_mean_c1 - neg_mean_target
                          (positive = target T1D- scored lower)
        auc_delta       = AUC_target - AUC_cohort1
    """
    records = []
    for label in METHOD_ORDER:
        if label not in submissions:
            continue
        sub = submissions[label]
        mc1 = _merge(meta_c1,     sub, COHORT1_DATASET_ID)
        mt  = _merge(meta_target, sub, target_dataset_id)

        neg_c1  = mc1[mc1['label_positive'] == 0]['label_positive_probability'].values
        neg_tgt = mt[mt['label_positive']  == 0]['label_positive_probability'].values

        auc_delta = (overall_auc_table.loc[label, 'delta']
                     if label in overall_auc_table.index else np.nan)
        records.append({
            'method':          label,
            'neg_mean_c1':     float(neg_c1.mean()),
            'neg_mean_target': float(neg_tgt.mean()),
            'neg_mean_shift':  float(neg_c1.mean() - neg_tgt.mean()),
            'auc_delta':       auc_delta,
        })
    return pd.DataFrame(records).set_index('method')


def plot_ctrl_mean_shift(shift_df: pd.DataFrame,
                         target_dataset_id: str,
                         output_dir: str):
    """
    Grouped bar chart: T1D- mean shift and AUC delta per method.
    Methods ordered by AUC delta descending.
    """
    df = shift_df.dropna(subset=['auc_delta']).copy()
    df = df.sort_values('auc_delta', ascending=False)

    x = np.arange(len(df))
    width = 0.35

    color_shift = plt.cm.viridis(0.2)
    color_delta = plt.cm.viridis(0.7)

    fig, ax = plt.subplots(figsize=(max(10, len(df) * 0.7), 5))

    bars1 = ax.bar(x - width / 2, df['neg_mean_shift'], width,
                   color=color_shift, alpha=0.85,
                   label='T1D\u2212 mean shift\n(mean_c1 \u2212 mean_target)')
    bars2 = ax.bar(x + width / 2, df['auc_delta'], width,
                   color=color_delta, alpha=0.85,
                   label='AUC delta\n(AUC_target \u2212 AUC_c1)')

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(df.index, fontsize=9, rotation=45, ha='right')
    ax.set_ylabel('Difference', fontsize=10)
    ax.set_title(
        f'T1D\u2212 mean score shift vs AUC difference: cohort-1 vs {target_dataset_id}\n'
        'T1D\u2212 mean shift = mean(cohort-1 T1D\u2212) \u2212 mean(target T1D\u2212)  '
        '|  AUC delta = AUC_target \u2212 AUC_c1\n'
        'Methods ordered by AUC delta descending',
        fontsize=10)
    ax.legend(fontsize=9, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2,
                h + 0.005 if h >= 0 else h - 0.015,
                f'{h:.2f}', ha='center',
                va='bottom' if h >= 0 else 'top', fontsize=6.5)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2,
                h + 0.005 if h >= 0 else h - 0.015,
                f'{h:.2f}', ha='center',
                va='bottom' if h >= 0 else 'top', fontsize=6.5)

    plt.tight_layout()
    out = os.path.join(output_dir, 'ctrl_mean_shift.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Phase-2 sequencing depth analysis (3 outputs per dataset).'
    )
    parser.add_argument('--cohort1_metadata', required=True,
                        help='Path to q6_t1d_test.csv (depth-matched cohort-1 reference)')
    parser.add_argument('--target_metadata', required=True,
                        help='Path to test_dataset_N_metadata.csv')
    parser.add_argument('--submissions_dir', required=True,
                        help='Directory containing phase-2 submission CSVs')
    parser.add_argument('--target_dataset_id', required=True,
                        help='Dataset ID used in submission files, e.g. test_dataset_9')
    parser.add_argument('--output_dir', required=True,
                        help='Output directory for this target dataset')
    return parser.parse_args()


def execute(cohort1_metadata_path: str,
            target_metadata_path: str,
            submissions_dir: str,
            target_dataset_id: str,
            output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    print(f'Loading metadata...')
    meta_c1  = load_cohort1_metadata(cohort1_metadata_path)
    meta_tgt = load_target_metadata(target_metadata_path)
    print(f'  depth-matched cohort (cohort-1): {len(meta_c1)} subjects '
          f'(T1D+={meta_c1["label_positive"].sum():.0f}, '
          f'T1D-={(meta_c1["label_positive"]==0).sum()})')
    print(f'  {target_dataset_id} (depth-discordant): {len(meta_tgt)} subjects '
          f'(T1D+={meta_tgt["label_positive"].sum():.0f}, '
          f'T1D-={(meta_tgt["label_positive"]==0).sum()})')

    print('Loading submissions...')
    submissions = load_submissions(submissions_dir, COHORT1_DATASET_ID, target_dataset_id)
    print(f'  Loaded {len(submissions)} methods: {sorted(submissions.keys())}')

    # --- AUC comparison ---
    print('\n[1/3] Overall AUC — depth-discordant vs depth-matched...')
    overall = compute_overall_aucs(meta_c1, meta_tgt, submissions, target_dataset_id)
    print(overall.to_string(float_format='{:.4f}'.format))
    overall.index.name = 'method'
    overall.to_csv(os.path.join(output_dir, 'sequencing_depth_auc_comparison.tsv'), sep='\t')
    print(f'  Saved: {os.path.join(output_dir, "sequencing_depth_auc_comparison.tsv")}')
    plot_sequencing_depth_auc_comparison(overall, target_dataset_id, output_dir)

    # --- Score distributions ---
    print('\n[2/3] Score distributions...')
    score_stats = compute_score_stats(meta_c1, meta_tgt, submissions, target_dataset_id)
    score_stats.index.name = 'method'
    score_stats.to_csv(os.path.join(output_dir, 'ctrl_score_distributions.tsv'), sep='\t')
    print(f'  Saved: {os.path.join(output_dir, "ctrl_score_distributions.tsv")}')
    plot_ctrl_score_distributions(meta_c1, meta_tgt, submissions, target_dataset_id, output_dir)

    # --- T1D- mean shift ---
    print('\n[3/3] T1D- mean shift...')
    shift_df = compute_ctrl_mean_shift(meta_c1, meta_tgt, submissions, overall, target_dataset_id)
    print(shift_df.to_string(float_format='{:.4f}'.format))
    shift_df.index.name = 'method'
    shift_df.to_csv(os.path.join(output_dir, 'ctrl_mean_shift.tsv'), sep='\t')
    print(f'  Saved: {os.path.join(output_dir, "ctrl_mean_shift.tsv")}')
    plot_ctrl_mean_shift(shift_df, target_dataset_id, output_dir)

    print('\nDone.')


if __name__ == '__main__':
    args = parse_args()
    execute(
        cohort1_metadata_path=args.cohort1_metadata,
        target_metadata_path=args.target_metadata,
        submissions_dir=args.submissions_dir,
        target_dataset_id=args.target_dataset_id,
        output_dir=args.output_dir,
    )
