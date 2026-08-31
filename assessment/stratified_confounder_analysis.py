"""
Stratified Analysis for T1D Cohort
====================================

Retrospective post-hoc analysis investigating how well top competition methods
generalize within the T1D disease population when stratified by biological
variables (HLA genotype, age, sex, control type).

Two test cohorts are analysed:
  - test_dataset_8_2: depth-discordant cohort — T1D+ and T1D- examples were
    sequenced with different protocols.
  - test_dataset_8_3: depth-matched cohort — both T1D+ and T1D- share the
    same sequencing protocol; used for all biological stratification analyses.

Usage
-----
    python assessment/stratified_confounder_analysis.py \
        --metadata_8_2  /path/to/test_dataset_8_2_metadata.csv \
        --metadata_8_3  /path/to/test_dataset_8_3_metadata_with_weights.csv \
        --submissions_dir /path/to/submission_files/ \
        --output_dir /path/for/outputs/

Output files
------------
    overall_auc_table.tsv                  10 ranks × AUC_disc, AUC_match, delta
    sequencing_depth_auc_comparison.png    paired dot plot showing AUC gap
    stratified_auc_8_3_sex.tsv             AUC ± 95% CI per rank × sex
    stratified_auc_8_3_age.tsv             AUC ± 95% CI per rank × age bin
    stratified_auc_8_3_hla.tsv             AUC ± 95% CI per rank × HLA bucket
    stratified_auc_8_3_control_type.tsv    AUC ± 95% CI per rank × control type
    stratified_auc_heatmaps.png            4-panel heatmap (sex, age, HLA, ctrl type)
    ctrl_score_distributions.png           boxplot grid of T1D- and T1D+ scores
                                           in depth-discordant vs depth-matched cohorts
    ctrl_score_distributions.tsv           per-rank score summary statistics
    ctrl_mean_shift.png                    T1D- mean score shift and AUC delta
    ctrl_mean_shift.tsv                    per-rank T1D- mean shift and AUC delta
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
import seaborn as sns
from scipy import stats as scipy_stats
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_POSITIVE_FOR_AUC = 5
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 42
RANK_LABELS = [f'rank_{i}' for i in range(1, 11)]

# HLA merging: DR3/4, DR3/3, DR4/4 → high_risk
_HIGH_RISK_HLA = {'DR3/4', 'DR3/3', 'DR4/4'}
HLA_ORDER  = ['high_risk', 'DR3/X', 'DR4/X', 'DRX/X']
AGE_ORDER  = ['<= 15', '16-22', 'adult (>22)']
SEX_ORDER  = ['F', 'M']
CTRL_ORDER = ['FDR', 'SDR', 'CTRL']

# Viridis-D truncated colormap starting at 0.2 (matches R scale_fill_viridis option='D' begin=0.2)
_VIRIDIS_TRUNC = mcolors.LinearSegmentedColormap.from_list(
    'viridis_trunc', plt.cm.viridis(np.linspace(0.2, 1.0, 256)))

# 4 colourblind-safe colours sampled from viridis-D at even intervals
_V_COLS = [plt.cm.viridis(x) for x in np.linspace(0.2, 0.95, 4)]
COLOR_DISC_NEG  = _V_COLS[0]   # depth-discordant  T1D-
COLOR_MATCH_NEG = _V_COLS[1]   # depth-matched     T1D-
COLOR_DISC_POS  = _V_COLS[2]   # depth-discordant  T1D+
COLOR_MATCH_POS = _V_COLS[3]   # depth-matched     T1D+


# ---------------------------------------------------------------------------
# Data loading & preprocessing
# ---------------------------------------------------------------------------

def _hla_bucket(value):
    if pd.isna(value):
        return 'unknown'
    if value in _HIGH_RISK_HLA:
        return 'high_risk'
    return value


def _age_bucket(value):
    if pd.isna(value):
        return 'unknown'
    if value == '<= 15':
        return '<= 15'
    if value == '16-22':
        return '16-22'
    return 'adult (>22)'


def load_metadata_8_2(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['label_positive'] = df['label_positive'].map({True: 1, False: 0, 'True': 1, 'False': 0})
    df['hla_bucket'] = df['hla_high_risk_type'].apply(_hla_bucket)
    return df


def load_metadata_8_3(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['label_positive'] = df['label_positive'].map({True: 1, False: 0, 'True': 1, 'False': 0})
    df['hla_bucket'] = df['hla_high_risk_type'].apply(_hla_bucket)
    df['age_bucket'] = df['age_class'].apply(_age_bucket)
    return df


def load_submissions(submissions_dir: str) -> dict:
    submissions = {}
    for rank in RANK_LABELS:
        path = os.path.join(submissions_dir, f'{rank}.csv')
        if not os.path.exists(path):
            print(f'  WARNING: {path} not found, skipping.')
            continue
        df = pd.read_csv(path)
        df = df[df['dataset'].isin(['test_dataset_8_2', 'test_dataset_8_3'])].copy()
        submissions[rank] = df
    return submissions


def merge_with_metadata(metadata: pd.DataFrame,
                        submission: pd.DataFrame,
                        dataset_label: str) -> pd.DataFrame:
    sub = submission[submission['dataset'] == dataset_label][['ID', 'label_positive_probability']].copy()
    sub = sub.rename(columns={'ID': 'subject_id'})
    merged = metadata.merge(sub, on='subject_id', how='inner')
    if len(merged) != len(metadata):
        print(f'  WARNING: {dataset_label} merge: metadata has {len(metadata)} rows '
              f'but only {len(merged)} matched in submission.')
    return merged


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def compute_auc(y_true, y_score) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_score))
    y_true, y_score = y_true[mask], y_score[mask]
    if y_true.sum() < MIN_POSITIVE_FOR_AUC:
        return np.nan
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_score))


def bootstrap_ci(y_true, y_score, n_boot: int = N_BOOTSTRAP,
                 seed: int = BOOTSTRAP_SEED):
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_score))
    y_true, y_score = y_true[mask], y_score[mask]
    if y_true.sum() < MIN_POSITIVE_FOR_AUC or len(np.unique(y_true)) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    n = len(y_true)
    boot_aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        boot_aucs.append(roc_auc_score(yt, ys))
    if len(boot_aucs) < 10:
        return np.nan, np.nan
    return float(np.percentile(boot_aucs, 2.5)), float(np.percentile(boot_aucs, 97.5))


# ---------------------------------------------------------------------------
# Analysis 1: AUC comparison — depth-discordant vs depth-matched
# ---------------------------------------------------------------------------

def compute_overall_aucs(meta_disc: pd.DataFrame,
                         meta_match: pd.DataFrame,
                         submissions: dict) -> pd.DataFrame:
    records = []
    for rank in RANK_LABELS:
        if rank not in submissions:
            continue
        sub = submissions[rank]
        md = merge_with_metadata(meta_disc,  sub, 'test_dataset_8_2')
        mm = merge_with_metadata(meta_match, sub, 'test_dataset_8_3')
        auc_disc  = compute_auc(md['label_positive'], md['label_positive_probability'])
        auc_match = compute_auc(mm['label_positive'], mm['label_positive_probability'])
        delta = (auc_disc - auc_match) if not (np.isnan(auc_disc) or np.isnan(auc_match)) else np.nan
        records.append({'rank': rank, 'AUC_disc': auc_disc, 'AUC_match': auc_match, 'delta': delta})
    return pd.DataFrame(records).set_index('rank')


# ---------------------------------------------------------------------------
# Analysis 2: Stratified AUC on depth-matched cohort
# ---------------------------------------------------------------------------

def compute_stratified_aucs(meta_match: pd.DataFrame,
                             submissions: dict,
                             strat_col: str,
                             strata_order: list) -> pd.DataFrame:
    cols = pd.MultiIndex.from_product([strata_order, ['AUC', 'CI_low', 'CI_high', 'n_pos', 'n_neg']])
    result = pd.DataFrame(index=RANK_LABELS, columns=cols, dtype=float)

    for rank in RANK_LABELS:
        if rank not in submissions:
            continue
        sub = submissions[rank]
        merged = merge_with_metadata(meta_match, sub, 'test_dataset_8_3')

        for stratum in strata_order:
            subset = merged[merged[strat_col] == stratum]
            y_true  = subset['label_positive'].values
            y_score = subset['label_positive_probability'].values
            n_pos = int(y_true.sum())
            n_neg = int((y_true == 0).sum())
            auc = compute_auc(y_true, y_score)
            ci_low, ci_high = bootstrap_ci(y_true, y_score)
            result.loc[rank, (stratum, 'AUC')]     = auc
            result.loc[rank, (stratum, 'CI_low')]  = ci_low
            result.loc[rank, (stratum, 'CI_high')] = ci_high
            result.loc[rank, (stratum, 'n_pos')]   = n_pos
            result.loc[rank, (stratum, 'n_neg')]   = n_neg

    return result


def compute_control_type_aucs(meta_match: pd.DataFrame,
                               submissions: dict) -> pd.DataFrame:
    cols = pd.MultiIndex.from_product([CTRL_ORDER, ['AUC', 'CI_low', 'CI_high', 'n_pos', 'n_neg']])
    result = pd.DataFrame(index=RANK_LABELS, columns=cols, dtype=float)

    for rank in RANK_LABELS:
        if rank not in submissions:
            continue
        sub = submissions[rank]
        merged = merge_with_metadata(meta_match, sub, 'test_dataset_8_3')
        t1d_rows = merged[merged['label_positive'] == 1]

        for ctrl_type in CTRL_ORDER:
            ctrl_rows = merged[merged['diabetes_status'] == ctrl_type]
            subset = pd.concat([t1d_rows, ctrl_rows], ignore_index=True)
            y_true  = subset['label_positive'].values
            y_score = subset['label_positive_probability'].values
            n_pos = int(y_true.sum())
            n_neg = int((y_true == 0).sum())
            auc = compute_auc(y_true, y_score)
            ci_low, ci_high = bootstrap_ci(y_true, y_score)
            result.loc[rank, (ctrl_type, 'AUC')]     = auc
            result.loc[rank, (ctrl_type, 'CI_low')]  = ci_low
            result.loc[rank, (ctrl_type, 'CI_high')] = ci_high
            result.loc[rank, (ctrl_type, 'n_pos')]   = n_pos
            result.loc[rank, (ctrl_type, 'n_neg')]   = n_neg

    return result


# ---------------------------------------------------------------------------
# Output: TSV helpers
# ---------------------------------------------------------------------------

def save_stratified_tsv(df: pd.DataFrame, path: str):
    flat = df.copy()
    flat.columns = ['_'.join(c) for c in flat.columns]
    flat.index.name = 'rank'
    flat.to_csv(path, sep='\t')
    print(f'  Saved: {path}')


def save_tsv(df: pd.DataFrame, path: str):
    df.index.name = 'rank'
    df.to_csv(path, sep='\t')
    print(f'  Saved: {path}')


# ---------------------------------------------------------------------------
# Plotting: AUC comparison — depth-discordant vs depth-matched
# ---------------------------------------------------------------------------

def plot_sequencing_depth_auc_comparison(auc_table: pd.DataFrame, output_dir: str):
    """
    Paired dot plot: for each rank, two dots connected by a line.
    Ordered by AUC_match descending.
    """
    df = auc_table.dropna(subset=['AUC_disc', 'AUC_match']).copy()
    df = df.sort_values('AUC_match', ascending=False)

    color_disc  = plt.cm.viridis(0.2)
    color_match = plt.cm.viridis(0.75)

    fig, ax = plt.subplots(figsize=(8, 5))
    y_pos = np.arange(len(df))

    for i, (rank, row) in enumerate(df.iterrows()):
        ax.plot([row['AUC_disc'], row['AUC_match']], [i, i],
                color='grey', linewidth=1.2, zorder=1)
        ax.scatter(row['AUC_disc'],  i, color=color_disc,  zorder=2, s=60)
        ax.scatter(row['AUC_match'], i, color=color_match, zorder=2, s=60)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df.index, fontsize=9)
    ax.set_xlabel('ROC-AUC', fontsize=11)
    ax.set_title('AUC: depth-discordant cohort vs depth-matched cohort', fontsize=11)
    ax.set_xlim(0.5, 1.0)
    ax.axvline(0.5, color='black', linewidth=0.8, linestyle='--', alpha=0.4)

    patch_disc  = mpatches.Patch(color=color_disc,  label='depth-discordant cohort')
    patch_match = mpatches.Patch(color=color_match, label='depth-matched cohort')
    ax.legend(handles=[patch_disc, patch_match], fontsize=9, loc='lower right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out = os.path.join(output_dir, 'sequencing_depth_auc_comparison.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'  Saved: {out}')


# ---------------------------------------------------------------------------
# Plotting: stratified heatmaps (viridis-D, begin=0.2)
# ---------------------------------------------------------------------------

def _extract_auc_matrix(strat_df: pd.DataFrame, strata_order: list) -> pd.DataFrame:
    rows = []
    for rank in strat_df.index:
        row = {}
        for s in strata_order:
            try:
                row[s] = float(strat_df.loc[rank, (s, 'AUC')])
            except KeyError:
                row[s] = np.nan
        rows.append(row)
    return pd.DataFrame(rows, index=strat_df.index, columns=strata_order)


def _extract_npos_matrix(strat_df: pd.DataFrame, strata_order: list) -> pd.DataFrame:
    rows = []
    for rank in strat_df.index:
        row = {}
        for s in strata_order:
            try:
                row[s] = int(strat_df.loc[rank, (s, 'n_pos')])
            except (KeyError, ValueError):
                row[s] = 0
        rows.append(row)
    return pd.DataFrame(rows, index=strat_df.index, columns=strata_order)


def plot_stratified_heatmaps(sex_df, age_df, hla_df, ctrl_df, output_dir: str):
    """
    4-panel heatmap: sex, age group, HLA risk, control type.
    Colourmap: viridis-D beginning at 0.2 (matches manuscript palette).
    """
    panels = [
        ('Sex',          sex_df,  SEX_ORDER),
        ('Age group',    age_df,  AGE_ORDER),
        ('HLA risk',     hla_df,  HLA_ORDER),
        ('Control type\n(T1D+ vs.)', ctrl_df, CTRL_ORDER),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5.5))
    vmin, vmax = 0.4, 1.0

    for ax, (title, df, order) in zip(axes, panels):
        auc_mat  = _extract_auc_matrix(df, order)
        npos_mat = _extract_npos_matrix(df, order)

        annot = auc_mat.applymap(
            lambda v: f'{v:.2f}' if not np.isnan(v) else '—'
        )
        mask = auc_mat.isna()

        sns.heatmap(
            auc_mat.astype(float),
            ax=ax,
            annot=annot,
            fmt='',
            cmap=_VIRIDIS_TRUNC,
            vmin=vmin,
            vmax=vmax,
            mask=mask,
            linewidths=0.5,
            linecolor='white',
            cbar_kws={'shrink': 0.7, 'label': 'AUC'},
        )

        for r_idx in range(auc_mat.shape[0]):
            for c_idx in range(auc_mat.shape[1]):
                if mask.iloc[r_idx, c_idx]:
                    ax.add_patch(plt.Rectangle(
                        (c_idx, r_idx), 1, 1,
                        fill=True, color='#CCCCCC', zorder=3
                    ))
                    ax.text(c_idx + 0.5, r_idx + 0.5, '—',
                            ha='center', va='center', fontsize=8,
                            color='#666666', zorder=4)

        col_labels = []
        for s in order:
            n = npos_mat[s].iloc[0]
            col_labels.append(f'{s}\n(n_T1D+={n})')
        ax.set_xticklabels(col_labels, rotation=30, ha='right', fontsize=8)
        ax.set_yticklabels(auc_mat.index, rotation=0, fontsize=8)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_ylabel('')

    plt.suptitle('Stratified ROC-AUC on depth-matched cohort',
                 fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    out = os.path.join(output_dir, 'stratified_auc_heatmaps.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out}')


# ---------------------------------------------------------------------------
# Analysis A: Score distributions — depth-discordant vs depth-matched
# ---------------------------------------------------------------------------

def compute_ctrl_score_stats(meta_disc: pd.DataFrame,
                              meta_match: pd.DataFrame,
                              submissions: dict) -> pd.DataFrame:
    """
    Per-rank summary statistics and KS test comparing T1D- score distributions
    between depth-discordant and depth-matched cohorts.
    """
    records = []
    for rank in RANK_LABELS:
        if rank not in submissions:
            continue
        sub = submissions[rank]
        md = merge_with_metadata(meta_disc,  sub, 'test_dataset_8_2')
        mm = merge_with_metadata(meta_match, sub, 'test_dataset_8_3')
        neg_disc  = md[md['label_positive'] == 0]['label_positive_probability'].values
        neg_match = mm[mm['label_positive'] == 0]['label_positive_probability'].values
        pos_disc  = md[md['label_positive'] == 1]['label_positive_probability'].values
        pos_match = mm[mm['label_positive'] == 1]['label_positive_probability'].values
        ks_stat, ks_p = scipy_stats.ks_2samp(neg_disc, neg_match)
        records.append({
            'rank': rank,
            'neg_disc_mean':   neg_disc.mean(),
            'neg_disc_std':    neg_disc.std(),
            'neg_disc_median': np.median(neg_disc),
            'neg_match_mean':  neg_match.mean(),
            'neg_match_std':   neg_match.std(),
            'neg_match_median':np.median(neg_match),
            'pos_disc_mean':   pos_disc.mean(),
            'pos_disc_std':    pos_disc.std(),
            'pos_disc_median': np.median(pos_disc),
            'pos_match_mean':  pos_match.mean(),
            'pos_match_std':   pos_match.std(),
            'pos_match_median':np.median(pos_match),
            'ks_stat':         ks_stat,
            'ks_pvalue':       ks_p,
        })
    return pd.DataFrame(records).set_index('rank')


def plot_ctrl_score_distributions(meta_disc: pd.DataFrame,
                                   meta_match: pd.DataFrame,
                                   submissions: dict,
                                   output_dir: str):
    """
    2-row × 5-column grid of boxplots (one per rank), ordered by ascending
    T1D- std in the depth-discordant cohort.
    Four groups per panel:
      depth-discordant T1D-  (viridis 0.20)
      depth-matched    T1D-  (viridis 0.45)
      depth-discordant T1D+  (viridis 0.70)
      depth-matched    T1D+  (viridis 0.95)
    """
    # Collect per-rank data and compute ordering key (T1D- std in disc cohort)
    rank_data = {}
    for rank in RANK_LABELS:
        if rank not in submissions:
            continue
        sub = submissions[rank]
        md = merge_with_metadata(meta_disc,  sub, 'test_dataset_8_2')
        mm = merge_with_metadata(meta_match, sub, 'test_dataset_8_3')
        neg_disc  = md[md['label_positive'] == 0]['label_positive_probability'].values
        neg_match = mm[mm['label_positive'] == 0]['label_positive_probability'].values
        pos_disc  = md[md['label_positive'] == 1]['label_positive_probability'].values
        pos_match = mm[mm['label_positive'] == 1]['label_positive_probability'].values
        rank_data[rank] = {
            'neg_disc':  neg_disc,
            'neg_match': neg_match,
            'pos_disc':  pos_disc,
            'pos_match': pos_match,
            'neg_disc_std': neg_disc.std(),
        }

    # Sort by ascending T1D- std in depth-discordant cohort
    sorted_ranks = sorted(rank_data.keys(), key=lambda r: rank_data[r]['neg_disc_std'])

    n_ranks = len(sorted_ranks)
    ncols = 5
    nrows = int(np.ceil(n_ranks / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 8), sharey=True)
    axes_flat = axes.flatten()

    group_labels = ['depth-disc.\nT1D\u2212', 'depth-match.\nT1D\u2212',
                    'depth-disc.\nT1D+',       'depth-match.\nT1D+']
    colors = [COLOR_DISC_NEG, COLOR_MATCH_NEG, COLOR_DISC_POS, COLOR_MATCH_POS]

    for idx, rank in enumerate(sorted_ranks):
        ax = axes_flat[idx]
        d = rank_data[rank]
        data = [d['neg_disc'], d['neg_match'], d['pos_disc'], d['pos_match']]

        bp = ax.boxplot(data, patch_artist=True, widths=0.5,
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
        ax.set_title(f'{rank}\nT1D\u2212 std$_{{disc}}$={d["neg_disc_std"]:.3f}',
                     fontsize=8, pad=3)
        ax.set_ylabel('Predicted score' if idx % ncols == 0 else '', fontsize=8)
        ax.tick_params(axis='y', labelsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    for idx in range(len(sorted_ranks), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    patches = [
        mpatches.Patch(color=COLOR_DISC_NEG,  alpha=0.75, label='depth-discordant  T1D\u2212'),
        mpatches.Patch(color=COLOR_MATCH_NEG, alpha=0.75, label='depth-matched     T1D\u2212'),
        mpatches.Patch(color=COLOR_DISC_POS,  alpha=0.75, label='depth-discordant  T1D+'),
        mpatches.Patch(color=COLOR_MATCH_POS, alpha=0.75, label='depth-matched     T1D+'),
    ]
    fig.legend(handles=patches, fontsize=9,
               loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        'Predicted score distributions: depth-discordant vs depth-matched cohorts\n'
        'Panels ordered by ascending T1D\u2212 score std in depth-discordant cohort',
        fontsize=11, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out = os.path.join(output_dir, 'ctrl_score_distributions.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out}')


# ---------------------------------------------------------------------------
# Analysis B: T1D- mean score shift
# ---------------------------------------------------------------------------

def compute_ctrl_mean_shift(meta_disc: pd.DataFrame,
                             meta_match: pd.DataFrame,
                             submissions: dict,
                             overall_auc_table: pd.DataFrame) -> pd.DataFrame:
    """
    For each rank:
        neg_mean_disc    = mean predicted score for depth-discordant T1D-
        neg_mean_match   = mean predicted score for depth-matched     T1D-
        neg_mean_shift   = neg_mean_match - neg_mean_disc
                           (positive = T1D- scored lower in depth-discordant cohort)
        auc_delta        = AUC_disc - AUC_match
    """
    records = []
    for rank in RANK_LABELS:
        if rank not in submissions:
            continue
        sub = submissions[rank]
        md = merge_with_metadata(meta_disc,  sub, 'test_dataset_8_2')
        mm = merge_with_metadata(meta_match, sub, 'test_dataset_8_3')
        neg_disc  = md[md['label_positive'] == 0]['label_positive_probability'].values
        neg_match = mm[mm['label_positive'] == 0]['label_positive_probability'].values
        auc_delta = overall_auc_table.loc[rank, 'delta'] if rank in overall_auc_table.index else np.nan
        records.append({
            'rank':           rank,
            'neg_mean_disc':  float(neg_disc.mean()),
            'neg_mean_match': float(neg_match.mean()),
            'neg_mean_shift': float(neg_match.mean() - neg_disc.mean()),
            'auc_delta':      auc_delta,
        })
    return pd.DataFrame(records).set_index('rank')


def plot_ctrl_mean_shift(shift_df: pd.DataFrame, output_dir: str):
    """
    Grouped bar chart: T1D- mean shift and AUC delta per rank.
    Both threshold-free. Ranks ordered by AUC delta descending.
    """
    df = shift_df.dropna(subset=['auc_delta']).copy()
    df = df.sort_values('auc_delta', ascending=False)

    x = np.arange(len(df))
    width = 0.35

    color_shift = plt.cm.viridis(0.2)
    color_delta = plt.cm.viridis(0.7)

    fig, ax = plt.subplots(figsize=(12, 5))

    bars1 = ax.bar(x - width / 2, df['neg_mean_shift'], width,
                   color=color_shift, alpha=0.85,
                   label='T1D\u2212 mean shift\n(mean_match \u2212 mean_disc)')
    bars2 = ax.bar(x + width / 2, df['auc_delta'], width,
                   color=color_delta, alpha=0.85,
                   label='AUC delta\n(AUC_disc \u2212 AUC_match)')

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(df.index, fontsize=9)
    ax.set_ylabel('Difference (predicted score units)', fontsize=10)
    ax.set_title(
        'T1D\u2212 mean score shift vs AUC difference: depth-discordant vs depth-matched cohorts\n'
        'T1D\u2212 mean shift = mean(depth-matched T1D\u2212) \u2212 mean(depth-discordant T1D\u2212) '
        '| ranks ordered by AUC delta descending',
        fontsize=10)
    ax.legend(fontsize=9, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2,
                h + 0.005 if h >= 0 else h - 0.015,
                f'{h:.2f}', ha='center', va='bottom' if h >= 0 else 'top',
                fontsize=7)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2,
                h + 0.005 if h >= 0 else h - 0.015,
                f'{h:.2f}', ha='center', va='bottom' if h >= 0 else 'top',
                fontsize=7)

    plt.tight_layout()
    out = os.path.join(output_dir, 'ctrl_mean_shift.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'  Saved: {out}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Stratified analysis for T1D AIRR-ML benchmark.'
    )
    parser.add_argument('--metadata_8_2', required=True,
                        help='Path to test_dataset_8_2_metadata.csv')
    parser.add_argument('--metadata_8_3', required=True,
                        help='Path to test_dataset_8_3_metadata_with_weights.csv')
    parser.add_argument('--submissions_dir', required=True,
                        help='Directory containing rank_1.csv … rank_10.csv')
    parser.add_argument('--output_dir', required=True,
                        help='Directory for output tables and figures')
    return parser.parse_args()


def execute(metadata_8_2_path: str,
            metadata_8_3_path: str,
            submissions_dir: str,
            output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    print('Loading metadata...')
    meta_disc  = load_metadata_8_2(metadata_8_2_path)
    meta_match = load_metadata_8_3(metadata_8_3_path)
    print(f'  depth-discordant cohort: {len(meta_disc)} subjects '
          f'(T1D+={meta_disc["label_positive"].sum():.0f}, '
          f'T1D-={(meta_disc["label_positive"]==0).sum()})')
    print(f'  depth-matched cohort:    {len(meta_match)} subjects '
          f'(T1D+={meta_match["label_positive"].sum():.0f}, '
          f'T1D-={(meta_match["label_positive"]==0).sum()})')

    print('Loading submissions...')
    submissions = load_submissions(submissions_dir)
    print(f'  Loaded {len(submissions)} submission files: {sorted(submissions.keys())}')

    # ------------------------------------------------------------------
    # Analysis 1: overall AUC
    # ------------------------------------------------------------------
    print('\n[1/4] Computing overall AUC per rank...')
    overall = compute_overall_aucs(meta_disc, meta_match, submissions)
    print(overall.to_string(float_format='{:.4f}'.format))
    save_tsv(overall, os.path.join(output_dir, 'overall_auc_table.tsv'))
    plot_sequencing_depth_auc_comparison(overall, output_dir)

    # ------------------------------------------------------------------
    # Analysis 2: stratified AUC on depth-matched cohort
    # ------------------------------------------------------------------
    print('\n[2/4] Computing stratified AUC on depth-matched cohort...')

    print('  Sex...')
    sex_df = compute_stratified_aucs(meta_match, submissions, 'sex', SEX_ORDER)
    save_stratified_tsv(sex_df, os.path.join(output_dir, 'stratified_auc_8_3_sex.tsv'))

    print('  Age...')
    age_df = compute_stratified_aucs(meta_match, submissions, 'age_bucket', AGE_ORDER)
    save_stratified_tsv(age_df, os.path.join(output_dir, 'stratified_auc_8_3_age.tsv'))

    print('  HLA...')
    hla_df = compute_stratified_aucs(meta_match, submissions, 'hla_bucket', HLA_ORDER)
    save_stratified_tsv(hla_df, os.path.join(output_dir, 'stratified_auc_8_3_hla.tsv'))

    print('  Control type...')
    ctrl_df = compute_control_type_aucs(meta_match, submissions)
    save_stratified_tsv(ctrl_df, os.path.join(output_dir, 'stratified_auc_8_3_control_type.tsv'))

    # ------------------------------------------------------------------
    # Analysis 3: heatmaps
    # ------------------------------------------------------------------
    print('\n[3/4] Generating stratified AUC heatmaps...')
    plot_stratified_heatmaps(sex_df, age_df, hla_df, ctrl_df, output_dir)

    # ------------------------------------------------------------------
    # Analysis A: Score distributions
    # ------------------------------------------------------------------
    print('\n[4/4] Analysis A — Score distributions (depth-discordant vs depth-matched)...')
    ctrl_stats = compute_ctrl_score_stats(meta_disc, meta_match, submissions)
    print(ctrl_stats[['neg_disc_mean', 'neg_disc_std', 'neg_match_mean', 'neg_match_std',
                       'ks_stat', 'ks_pvalue']].to_string(float_format='{:.4f}'.format))
    ctrl_stats.index.name = 'rank'
    ctrl_stats.to_csv(os.path.join(output_dir, 'ctrl_score_distributions.tsv'), sep='\t')
    print(f'  Saved: {os.path.join(output_dir, "ctrl_score_distributions.tsv")}')
    plot_ctrl_score_distributions(meta_disc, meta_match, submissions, output_dir)

    # ------------------------------------------------------------------
    # Analysis B: T1D- mean score shift
    # ------------------------------------------------------------------
    print('\n[4/4] Analysis B — T1D- mean score shift (threshold-free)...')
    shift_df = compute_ctrl_mean_shift(meta_disc, meta_match, submissions, overall)
    print(shift_df.to_string(float_format='{:.4f}'.format))
    shift_df.index.name = 'rank'
    shift_df.to_csv(os.path.join(output_dir, 'ctrl_mean_shift.tsv'), sep='\t')
    print(f'  Saved: {os.path.join(output_dir, "ctrl_mean_shift.tsv")}')
    plot_ctrl_mean_shift(shift_df, output_dir)

    print('\nDone.')


if __name__ == '__main__':
    args = parse_args()
    execute(
        metadata_8_2_path=args.metadata_8_2,
        metadata_8_3_path=args.metadata_8_3,
        submissions_dir=args.submissions_dir,
        output_dir=args.output_dir,
    )
