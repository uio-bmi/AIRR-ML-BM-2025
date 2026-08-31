import os

import pandas as pd
import numpy as np
import pandas.api.types
from sklearn.metrics import roc_auc_score

N_ROWS_PUBLIC = 726
N_ROWS_PRIVATE = 403487
N_ROWS_TOTAL = N_ROWS_PUBLIC + N_ROWS_PRIVATE  # should be 404213
COLUMNS_NAMES = ['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']
TRAIN_DATASETS = [f'train_dataset_{i}' for i in range(1,
                                                      7)]  # bcause we have only 6 training datasets where we have ground truth labels for sequences


class ParticipantVisibleError(Exception):
    pass


def assert_submission_shape(submission: pd.DataFrame):
    if submission.shape[1] != len(COLUMNS_NAMES):
        raise ParticipantVisibleError(
            f'Submission has {submission.shape[1]} columns, but expected {len(COLUMNS_NAMES)} columns. Please check sample_submission.csv.')
    if submission.shape[0] != N_ROWS_TOTAL:
        raise ParticipantVisibleError(
            f'Submission is missing rows, please make sure you submitted {N_ROWS_TOTAL} rows as described in the submission instruction')


def assert_matching_dataset_counts(submission: pd.DataFrame):
    original_dataset_counts = {'test_dataset_1': 400, 'test_dataset_2': 400, 'test_dataset_3': 400,
                               'test_dataset_4': 400, 'test_dataset_5': 400, 'test_dataset_6': 400,
                               'test_dataset_7_1': 76, 'test_dataset_7_2': 100, 'test_dataset_8_1': 390,
                               'test_dataset_8_2': 857, 'test_dataset_8_3': 390, 'train_dataset_1': 50000,
                               'train_dataset_2': 50000, 'train_dataset_3': 50000, 'train_dataset_4': 50000,
                               'train_dataset_5': 50000, 'train_dataset_6': 50000, 'train_dataset_7': 50000,
                               'train_dataset_8': 50000}
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
    if submission['ID'].nunique() != submission.shape[0]:
        raise ParticipantVisibleError("Submission IDs must be unique.")
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


def compute_all_roc_aucs(submission: pd.DataFrame, solution: pd.DataFrame) -> list:
    roc_aucs = []
    relevant_test_datasets = assert_missing_test_datasets(solution, submission)
    print("Relevant test datasets:", relevant_test_datasets)
    for test_dataset in relevant_test_datasets:
        sub_mask = submission['dataset'] == test_dataset
        sol_mask = solution['dataset'] == test_dataset

        relevant_solutions, relevant_submissions = align_by_id(solution.loc[sol_mask], submission.loc[sub_mask])
        pred_probs = relevant_submissions['label_positive_probability']
        true_labels = relevant_solutions['label_positive']
        relevant_weights = relevant_solutions['weights'] if 'weights' in relevant_solutions.columns else None
        if relevant_weights is not None and (relevant_weights >= 0).all():
            roc_aucs.append(roc_auc_score(true_labels, pred_probs, sample_weight=relevant_weights))
        else:
            roc_aucs.append(roc_auc_score(true_labels, pred_probs))
    print("ROC AUCs:", roc_aucs)
    return roc_aucs


def compute_all_jaccard_similarities(submission: pd.DataFrame, solution: pd.DataFrame) -> list:
    jaccard_similarities = []
    relevant_cols = ['junction_aa', 'v_call', 'j_call']
    print("Relevant train datasets for Jaccard similarity:", TRAIN_DATASETS)
    for train_dataset in TRAIN_DATASETS:
        sub_mask = submission['dataset'] == train_dataset
        sol_mask = solution['dataset'] == train_dataset
        relevant_submissions = submission.loc[sub_mask, relevant_cols]
        relevant_solutions = solution.loc[sol_mask, relevant_cols]
        computed_index = compute_jaccard_similarity(relevant_submissions, relevant_solutions)
        jaccard_similarities.append(computed_index)
    print("Jaccard similarities:", jaccard_similarities)
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


def get_relevant_weights(leaderboard: str) -> np.ndarray:
    if leaderboard == "private":
        return np.array([1, 1, 1, 1, 1, 1, 1, 3, 2, 3, 3] + [1] * 6)
    elif leaderboard == "public":
        return np.array([1, 1, 1, 1, 1, 1, 1, 5])
    else:
        raise ValueError("Invalid leaderboard type. Use 'private' or 'public'.")


def score(solution: pd.DataFrame, submission: pd.DataFrame) -> float:
    """
    This is a tailored metric for suiting adaptive immune profiling challenge 2025.
    After asserting that the submission is formatted correctly and does not have missing data, it computes a
    weighted average of the ROC AUC and Jaccard similarity scores for the submission. The weights are determined
    based on the leaderboard type (private or public). The scoring process includes the following steps:

    1. Computes the ROC AUC scores for all test datasets in the submission.
    2. Computes the Jaccard similarity scores for all training datasets in the submission.
    3. Computes the weighted average of the ROC AUC and Jaccard similarity scores
    based on the relevant weights for the leaderboard type (private or public).

    :param solution: A DataFrame containing the solution data.
    :param submission: A DataFrame containing the submission data.
    :return: A floating point number representing the weighted average score of the submission.
    :raises ParticipantVisibleError: If any of the assertions fail, indicating that the submission is
    not formatted correctly or is missing required data.

    """
    assert_matching_dataset_counts(submission)
    assert_submission_shape(submission)
    assert_column_names(submission)
    assert_column_types(submission)
    assert_matching_ids(solution, submission)
    roc_aucs = compute_all_roc_aucs(submission, solution)
    if len(solution) > 1000:  # meaning for final private leaderboard
        jaccard_similarities = compute_all_jaccard_similarities(submission, solution)
    else:
        jaccard_similarities = []
    return roc_aucs, jaccard_similarities


if __name__ == '__main__':
    baseline_submissions_files = ["emerson_submissions.csv", "logistic_submissions.csv"]
    challenge_submission_files = [f"submission_files/rank_{i}.csv" for i in range(1, 11)]
    all_submission_files = baseline_submissions_files + challenge_submission_files
    solutions_path = "solutions_file.csv"
    solution = pd.read_csv(solutions_path, header=0)
    roc_aucs = {}
    jaccard_similarities = {}
    for submissions_path in all_submission_files:
        if os.path.exists(submissions_path):
            print("Scoring submission file:", submissions_path)
            submission = pd.read_csv(submissions_path, header=0)
            aucs, jacc = score(solution, submission)
            roc_aucs[os.path.basename(submissions_path)] = aucs
            jaccard_similarities[os.path.basename(submissions_path)] = jacc

    roc_aucs_df = pd.DataFrame.from_dict(roc_aucs, orient='index',
                                         columns=['test_dataset_1', 'test_dataset_2', 'test_dataset_3',
                                                  'test_dataset_4', 'test_dataset_5', 'test_dataset_6',
                                                  'test_dataset_7_1', 'test_dataset_7_2', 'test_dataset_8_1',
                                                  'test_dataset_8_2', 'test_dataset_8_3'])

    jaccard_similarities_df = pd.DataFrame.from_dict(jaccard_similarities, orient='index',
                                                     columns=['train_dataset_1', 'train_dataset_2', 'train_dataset_3',
                                                              'train_dataset_4', 'train_dataset_5', 'train_dataset_6'])

    roc_aucs_df.to_csv("roc_aucs_df.tsv", sep="\t", index=True)
    jaccard_similarities_df.to_csv("jaccard_similarities_df.tsv", sep="\t", index=True)
