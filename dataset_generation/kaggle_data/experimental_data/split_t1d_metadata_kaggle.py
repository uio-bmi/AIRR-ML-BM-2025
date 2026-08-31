import pandas as pd
import secrets
from sklearn.model_selection import train_test_split


def add_anonymized_filenames(metadata_csv, modified_metadata_csv):
    metadata = pd.read_csv(metadata_csv)
    metadata = metadata.rename(columns={"filename": "original_filename", "subject_id": "original_subject_id"})
    metadata["subject_id"] = [secrets.token_hex(16) for i in range(len(metadata))]
    metadata["filename"] = metadata["subject_id"] + ".tsv"
    metadata.to_csv(modified_metadata_csv, index=False)


def split_metadata(metadata_csv, train_dataset_file, test_dataset_1_file, test_dataset_3_file):
    metadata = pd.read_csv(metadata_csv)
    metadata['label_positive'] = metadata['ML_class'] == 'T1D'
    test_dataset_3 = metadata[metadata['dataset'] != 1]
    cohort_1 = metadata[metadata['dataset'] == 1]
    cohort_1_train, cohort_1_test = train_test_split(cohort_1, test_size=0.3, stratify=cohort_1['ML_class'],
                                                     random_state=2709)
    test_dataset_1_final, test_dataset_1_continuous = train_test_split(cohort_1_test, test_size=0.25,
                                                                       stratify=cohort_1_test['ML_class'],
                                                                       random_state=2709)
    test_dataset_1_continuous['continuous_leaderboard'] = True
    test_dataset_1_final['continuous_leaderboard'] = False
    test_dataset_1 = pd.concat([test_dataset_1_final, test_dataset_1_continuous])
    train_dataset_1 = cohort_1_train
    train_dataset_1.to_csv(train_dataset_file, index=False)
    test_dataset_1.to_csv(test_dataset_1_file, index=False)
    test_dataset_3.to_csv(test_dataset_3_file, index=False)


def write_relevant_fields_to_csv(metadata_csv, output_csv):
    metadata = pd.read_csv(metadata_csv)
    training_relevant_cols = ['subject_id', 'filename', 'label_positive', 'diabetes_status', 'sex',
                              'age', 'A', 'B', 'C', 'DPA1', 'DPB1',
                              'DQA1', 'DQB1', 'DRB1', 'DRB3', 'DRB4', 'DRB5']
    metadata = metadata[training_relevant_cols]
    metadata.to_csv(output_csv, index=False)


def combine_train_test_metadata(train_metadata_csv, test_metadata_csv, combined_metadata_csv):
    train_metadata = pd.read_csv(train_metadata_csv)
    test_metadata = pd.read_csv(test_metadata_csv)
    combined_metadata = pd.concat([train_metadata, test_metadata])
    combined_metadata.to_csv(combined_metadata_csv, index=False)

# usage example #

# if __name__ == "__main__":

# add_anonymized_filenames("/path/to/original_metadata_csv",
#                          "/path/to/original_metadata_anonymized_csv")
# full_metadata_files = ["t1d_train_dataset_1_full_metadata.csv",
#                        "t1d_test_dataset_1_full_metadata.csv",
#                        "t1d_test_dataset_3_full_metadata.csv"]
# relevant_metadata_files = ["t1d_train_dataset_1_relevant_metadata.csv",
#                            "t1d_test_dataset_1_relevant_metadata.csv",
#                            "t1d_test_dataset_3_relevant_metadata.csv"]
# split_metadata(
#     "/path/to/original_metadata_anonymized_csv",
#     f"/path/to/{full_metadata_files[0]}",
#     f"/path/to/{full_metadata_files[1]}",
#     f"/path/to/{full_metadata_files[2]}")
# for i in range(3):
#     write_relevant_fields_to_csv(
#         f"/path/to/{full_metadata_files[i]}",
#         f"/path/to/{relevant_metadata_files[i]}")
