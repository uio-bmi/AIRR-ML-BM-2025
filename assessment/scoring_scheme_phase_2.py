import glob
import os

import pandas as pd
import numpy as np
import pandas.api.types
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score

N_ROWS_TOTAL = 4782000
COLUMNS_NAMES = ['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']


class ParticipantVisibleError(Exception):
    pass


def assert_submission_shape(submission: pd.DataFrame):
    if submission.shape[1] != len(COLUMNS_NAMES):
        raise ParticipantVisibleError(
            f'Submission has {submission.shape[1]} columns, but expected {len(COLUMNS_NAMES)} columns. Please check sample_submission.csv.')
    if submission.shape[0] != N_ROWS_TOTAL:
        raise ParticipantVisibleError(
            f'Submission is missing rows, please make sure you submitted {N_ROWS_TOTAL} rows as described in the submission instruction')


def assert_matching_dataset_counts(submission: pd.DataFrame, solution: pd.DataFrame):
    original_dataset_counts = solution['dataset'].value_counts()
    original_dataset_counts = original_dataset_counts[
        original_dataset_counts.index.str.contains('test_dataset')].to_dict()
    for i in range(1, 96):
        original_dataset_counts[f'train_dataset_{i}'] = 50000
    dataset_counts = submission['dataset'].value_counts().to_dict()
    for dataset, count in original_dataset_counts.items():
        if dataset not in dataset_counts or dataset_counts[dataset] != count:
            raise ParticipantVisibleError(
                f'Submission is missing {count} rows for dataset {dataset}. Please check sample_submission.csv.')


def assert_column_names(submission: pd.DataFrame):
    sub_cols = submission.columns
    missing_columns = set(COLUMNS_NAMES) - set(sub_cols)
    if len(missing_columns) > 0:
        raise ParticipantVisibleError(f"Submission is missing the following columns: {','.join(missing_columns)}")


def assert_column_types(submission: pd.DataFrame):
    test_mask = submission['dataset'].astype(str).str.contains('test_dataset')
    if submission.loc[test_mask, 'label_positive_probability'].isnull().any():
        raise ParticipantVisibleError(
            'Submission column label_positive_probability contains missing values for test datasets')
    if not pandas.api.types.is_float_dtype(submission.loc[test_mask, 'label_positive_probability']):
        raise ParticipantVisibleError('Submission column label_positive_probability must be a float for test datasets')
    if not ((submission.loc[test_mask, 'label_positive_probability'] >= 0) & (
            submission.loc[test_mask, 'label_positive_probability'] <= 1)).all():
        raise ParticipantVisibleError(
            'Submission column label_positive_probability must contain values between 0 and 1 for test datasets')
    for col in ['junction_aa', 'v_call', 'j_call']:
        if submission[col].isnull().any():
            raise ParticipantVisibleError(f"{col} contains missing values.")
        train_mask = submission['dataset'].astype(str).str.contains('train_dataset')
        non_train_mask = ~train_mask
        if not submission.loc[train_mask, col].apply(lambda x: isinstance(x, str) and x != '-999.0').all():
            raise ParticipantVisibleError(f"{col} must be a string and not '-999.0' for train datasets")
        if not (submission.loc[non_train_mask, col] == '-999.0').all():
            raise ParticipantVisibleError(f"{col} must be -999.0 for non-train datasets")


def assert_missing_test_datasets(solution: pd.DataFrame, submission: pd.DataFrame) -> list:
    relevant_test_datasets = solution['dataset'].unique()
    relevant_test_datasets = [dataset for dataset in relevant_test_datasets if 'test_dataset' in dataset]
    missing_test_datasets = set(relevant_test_datasets) - set(submission['dataset'].unique())
    if missing_test_datasets:
        raise ParticipantVisibleError(
            f'Submission is missing predictions for the following test datasets: {", ".join(missing_test_datasets)}')
    return relevant_test_datasets


def assert_matching_ids(solution: pd.DataFrame, submission: pd.DataFrame):
    if not solution['ID'].isin(submission['ID']).all():
        raise ParticipantVisibleError("Submission is missing some unique IDs that should have been present.")


def align_by_id(solution: pd.DataFrame, submission: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    # Align submission to solution by ID and check for matching IDs
    solution_sorted = solution.sort_values('ID').reset_index(drop=True)
    submission = submission[submission['ID'].isin(solution_sorted['ID'])]
    submission_sorted = submission.sort_values('ID').reset_index(drop=True)
    if not solution_sorted['ID'].equals(submission_sorted['ID']):
        raise ValueError("Submission and solution IDs do not match or are not in the same order.")
    return solution_sorted, submission_sorted


def compute_all_roc_aucs(submission: pd.DataFrame, solution: pd.DataFrame) -> dict:
    roc_aucs = {}
    relevant_test_datasets = assert_missing_test_datasets(solution, submission)
    print("Relevant test datasets:", relevant_test_datasets)
    for test_dataset in relevant_test_datasets:
        sub_mask = submission['dataset'] == test_dataset
        sol_mask = solution['dataset'] == test_dataset

        relevant_solutions, relevant_submissions = align_by_id(solution.loc[sol_mask], submission.loc[sub_mask])
        pred_probs = relevant_submissions['label_positive_probability']
        true_labels = relevant_solutions['label_positive']
        roc_aucs[test_dataset] = roc_auc_score(true_labels, pred_probs)
    print("ROC AUCs:", roc_aucs)
    return roc_aucs

def compute_all_balanced_accuracies(submission: pd.DataFrame, solution: pd.DataFrame, threshold: float = 0.5) -> dict:
    balanced_accuracies = {}
    relevant_test_datasets = assert_missing_test_datasets(solution, submission)

    for test_dataset in relevant_test_datasets:
        sub_mask = submission['dataset'] == test_dataset
        sol_mask = solution['dataset'] == test_dataset

        relevant_solutions, relevant_submissions = align_by_id(solution.loc[sol_mask], submission.loc[sub_mask])
        pred_probs = relevant_submissions['label_positive_probability']
        pred_labels = (pred_probs >= threshold).astype(int)
        true_labels = relevant_solutions['label_positive']
        balanced_accuracies[test_dataset] = balanced_accuracy_score(true_labels, pred_labels)

    print("Balanced accuracies:", balanced_accuracies)
    return balanced_accuracies


def compute_pr_auc(predicted_sequences: pd.DataFrame, true_sequences: pd.DataFrame) -> float:
    predicted_sequences = predicted_sequences.drop_duplicates()
    true_sequences = true_sequences.drop_duplicates()

    n_true = true_sequences.shape[0]
    needed_dummy_count = n_true - predicted_sequences.shape[0]

    if predicted_sequences.shape[0] < n_true:
        dummy_df = pd.DataFrame({
            'junction_aa': [f'dummy_{i}' for i in range(needed_dummy_count)],
            'v_call': [f'dummy_{i}' for i in range(needed_dummy_count)],
            'j_call': [f'dummy_{i}' for i in range(needed_dummy_count)]
        })
        predicted_sequences = pd.concat([predicted_sequences, dummy_df], ignore_index=True)

    predicted_sequences = predicted_sequences.head(n_true).reset_index(drop=True)
    true_sequences = true_sequences.reset_index(drop=True)

    predicted_tuples = list(map(tuple, predicted_sequences[['junction_aa', 'v_call', 'j_call']].to_numpy()))
    true_tuple_set = set(map(tuple, true_sequences[['junction_aa', 'v_call', 'j_call']].to_numpy()))

    y_true = np.array([1 if seq in true_tuple_set else 0 for seq in predicted_tuples], dtype=int)

    if y_true.sum() == 0:
        return 0.0

    # Uses row order as ranking: earlier rows are treated as higher-confidence predictions.
    y_score = np.arange(len(y_true), 0, -1, dtype=float)

    return average_precision_score(y_true, y_score)


def compute_all_pr_aucs(submission: pd.DataFrame, solution: pd.DataFrame) -> dict:
    pr_aucs = {}
    relevant_cols = ['junction_aa', 'v_call', 'j_call']
    train_datasets = [item for item in solution['dataset'].unique() if 'train' in item]

    for train_dataset in train_datasets:
        sub_mask = submission['dataset'] == train_dataset
        sol_mask = solution['dataset'] == train_dataset
        relevant_submissions = submission.loc[sub_mask, relevant_cols]
        relevant_solutions = solution.loc[sol_mask, relevant_cols]
        computed_auc = compute_pr_auc(relevant_submissions, relevant_solutions)
        pr_aucs[train_dataset] = computed_auc

    print("PR AUCs:", pr_aucs)
    return pr_aucs


def compute_all_jaccard_similarities(submission: pd.DataFrame, solution: pd.DataFrame) -> dict:
    jaccard_similarities = {}
    relevant_cols = ['junction_aa', 'v_call', 'j_call']
    train_datasets = [item for item in solution['dataset'].unique() if 'train' in item]
    for train_dataset in train_datasets:
        sub_mask = submission['dataset'] == train_dataset
        sol_mask = solution['dataset'] == train_dataset
        relevant_submissions = submission.loc[sub_mask, relevant_cols]
        relevant_solutions = solution.loc[sol_mask, relevant_cols]
        computed_index = compute_jaccard_similarity(relevant_submissions, relevant_solutions)
        jaccard_similarities[train_dataset] = computed_index
    print("Jaccard Similarities:", jaccard_similarities)
    return jaccard_similarities


def compute_jaccard_similarity(predicted_sequences: pd.DataFrame, true_sequences: pd.DataFrame) -> float:
    predicted_sequences = predicted_sequences.drop_duplicates()
    true_sequences = true_sequences.drop_duplicates()
    n_true = true_sequences.shape[0]
    needed_dummy_count = n_true - predicted_sequences.shape[0]
    if predicted_sequences.shape[0] < n_true:
        dummy_df = pd.DataFrame({
            'junction_aa': [f'dummy_{i}' for i in range(needed_dummy_count)],
            'v_call': [f'dummy_{i}' for i in range(needed_dummy_count)],
            'j_call': [f'dummy_{i}' for i in range(needed_dummy_count)]
        })
        predicted_sequences = pd.concat([predicted_sequences, dummy_df])
    predicted_sequences = predicted_sequences.head(n_true)
    intersection = pd.merge(predicted_sequences, true_sequences, how='inner')
    union = pd.concat([predicted_sequences, true_sequences]).drop_duplicates()
    jaccard_index = intersection.shape[0] / union.shape[0]
    return jaccard_index


def score(solution: pd.DataFrame, submission: pd.DataFrame) -> tuple:
    """
    This is a tailored metric for suiting adaptive immune profiling challenge 2025.
    After asserting that the submission is formatted correctly and does not have missing data, it computes a
    weighted average of the ROC AUC and Jaccard similarity scores for the submission. The weights are determined
    based on the leaderboard type (private or public). The scoring process includes the following steps:

    1. Computes the ROC AUC scores for all test datasets in the submission.
    2. Computes the Jaccard similarity scores for all training datasets in the submission.


    :param solution: A DataFrame containing the solution data.
    :param submission: A DataFrame containing the submission data.
    :return: Four dictionaries containing per-dataset ROC AUC, balanced accuracy, Jaccard similarity, and PR AUC scores.
    :raises ParticipantVisibleError: If any of the assertions fail, indicating that the submission is
    not formatted correctly or is missing required data.

    """
    assert_matching_dataset_counts(submission, solution)
    assert_submission_shape(submission)
    assert_column_names(submission)
    assert_column_types(submission)
    assert_matching_ids(solution, submission)
    roc_aucs = compute_all_roc_aucs(submission, solution)
    balanced_accuracies = compute_all_balanced_accuracies(submission, solution)
    jaccard_similarities = compute_all_jaccard_similarities(submission, solution)
    pr_aucs = compute_all_pr_aucs(submission, solution)
    return roc_aucs, balanced_accuracies, jaccard_similarities, pr_aucs


if __name__ == '__main__':
    baseline_submissions_files = []
    challenge_submission_files = []
    for i in range(1, 11):
        pattern = f"phase_2_submission_files/rank_{i}*.csv"
        challenge_submission_files.extend(glob.glob(pattern))
    challenge_submission_files = list(set(challenge_submission_files))
    all_submission_files = baseline_submissions_files + challenge_submission_files
    solutions_path = "solutions_file.csv"
    solution = pd.read_csv(solutions_path, header=0)
    roc_aucs = {}
    balanced_accuracies = {}
    jaccard_similarities = {}
    pr_aucs = {}
    for submissions_path in all_submission_files:
        if os.path.exists(submissions_path):
            print("Scoring submission file:", submissions_path)
            submission = pd.read_csv(submissions_path, header=0)
            aucs, bals, jacc, pr_auc_values = score(solution, submission)
            roc_aucs[os.path.basename(submissions_path)] = aucs
            balanced_accuracies[os.path.basename(submissions_path)] = bals
            jaccard_similarities[os.path.basename(submissions_path)] = jacc
            pr_aucs[os.path.basename(submissions_path)] = pr_auc_values
    roc_aucs_df = pd.DataFrame.from_dict(roc_aucs, orient='index')
    balanced_accuracies_df = pd.DataFrame.from_dict(balanced_accuracies, orient='index')
    jaccard_similarities_df = pd.DataFrame.from_dict(jaccard_similarities, orient='index')
    pr_aucs_df = pd.DataFrame.from_dict(pr_aucs, orient='index')
    roc_aucs_df.to_csv("roc_aucs_df.tsv", sep='\t', index=True)
    balanced_accuracies_df.to_csv("balanced_accuracies_df.tsv", sep='\t', index=True)
    jaccard_similarities_df.to_csv("jaccard_similarities_df.tsv", sep='\t', index=True)
    pr_aucs_df.to_csv("pr_aucs_df.tsv", sep='\t', index=True)