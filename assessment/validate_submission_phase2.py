import pandas as pd
import pandas.api.types

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


def assert_matching_dataset_counts(submission: pd.DataFrame, sample_submission: pd.DataFrame):
    original_dataset_counts = sample_submission['dataset'].value_counts()
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


def assert_matching_ids(sample_submission: pd.DataFrame, submission: pd.DataFrame):
    if not sample_submission['ID'].isin(submission['ID']).all():
        raise ParticipantVisibleError("Submission is missing some unique IDs that should have been present.")


def validate_submission(sample_submission: pd.DataFrame, submission: pd.DataFrame) -> None:
    """
    A utility function to validate that the submission file indeed matches what is expected.

    :param sample_submission: A DataFrame containing the sample submission data provided.
    :param submission: A DataFrame containing the submission data.
    :return: None.
    :raises ParticipantVisibleError: If any of the assertions fail, indicating that the submission is
    not formatted correctly or is missing required data.

    """
    assert_matching_dataset_counts(submission, sample_submission)
    assert_submission_shape(submission)
    assert_column_names(submission)
    assert_column_types(submission)
    assert_matching_ids(sample_submission, submission)
    print("Submission validation passed successfully.")


if __name__ == '__main__':
    sample_submission_path = "sample_submission_phase2.csv"
    sample_submission_df = pd.read_csv(sample_submission_path, header=0)
    submission_files_path = "phase_2_submission_files"
    import glob
    submission_files = glob.glob(f"{submission_files_path}/*.csv")
    for submission_file in submission_files:
        print(f"Validating submission file: {submission_file}")
        submission_df = pd.read_csv(submission_file, header=0)
        try:
            validate_submission(sample_submission_df, submission_df)
        except ParticipantVisibleError as e:
            print(f"Validation failed for {submission_file}: {str(e)}")

