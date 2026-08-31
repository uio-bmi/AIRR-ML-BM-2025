import pandas as pd
import secrets
import os
from dataset_generation.util import makedir_if_not_exists


def process_metadata(metadata_tsv, modified_metadata_csv):
    metadata = pd.read_csv(metadata_tsv, sep='\t')
    metadata["subject_id"] = [secrets.token_hex(16) for i in range(len(metadata))]
    metadata["filename"] = metadata["subject_id"] + ".tsv"
    metadata['label_positive'] = metadata['label'] == 1
    metadata = metadata.rename(columns={"race": "ancestry"})

    metadata["dataset_type"] = metadata["train_validation_holdout"].replace(
        {"train": "train_dataset", "validation": "test_dataset_1", "holdout": "test_dataset_2"})
    metadata["batch"] = metadata["collection"].replace({"cro_deep_v4b": "batch_1", "cro_ultradeep_v4b": "batch_2", "cro_temp": "batch_3"})
    metadata.to_csv(modified_metadata_csv, index=False)


def move_anonymized_data(metadata_file, concatenated_repertoire_file, destination_path):
    metadata = pd.read_csv(metadata_file)
    all_repertoires = pd.read_csv(concatenated_repertoire_file, header=0, sep='\t')
    all_repertoires = all_repertoires.rename(columns={"duplicate_count": "templates"})
    all_repertoires = all_repertoires[["junction_aa", "v_call", "j_call", "templates", "sample_id"]]
    train_path = os.path.join(destination_path, "train_dataset")
    test_dataset_1_path = os.path.join(destination_path, "test_dataset_1")
    test_dataset_2_path = os.path.join(destination_path, "test_dataset_2")
    for dest_path in [train_path, test_dataset_1_path, test_dataset_2_path]:
        makedir_if_not_exists(dest_path)
    for i in range(len(metadata)):
        print("Processing file", i)
        sample_id = metadata.loc[i, "sample_id"]
        dataset_type = metadata.loc[i, "dataset_type"]
        df = all_repertoires[all_repertoires["sample_id"] == sample_id]
        df = df.drop(columns=["sample_id"])
        df = df.reset_index(drop=True)
        destination_file = os.path.join(destination_path, dataset_type, metadata.loc[i, "filename"])
        df.to_csv(destination_file, sep='\t', index=False)
    write_metadata_files(destination_path, metadata)


def write_metadata_files(destination_path, metadata):
    train_metadata_file = os.path.join(destination_path, "train_metadata.csv")
    test_dataset_1_metadata_file = os.path.join(destination_path, "test_dataset_1_metadata.csv")
    test_dataset_2_metadata_file = os.path.join(destination_path, "test_dataset_2_metadata.csv")
    metadata = metadata[["subject_id", "filename", "label_positive", "age", "sex", "ancestry", "batch", "dataset_type"]]
    metadata_train = metadata[metadata["dataset_type"] == "train_dataset"]
    metadata_test_dataset_1 = metadata[metadata["dataset_type"] == "test_dataset_1"]
    metadata_test_dataset_2 = metadata[metadata["dataset_type"] == "test_dataset_2"]
    metadata_train.drop(columns=["dataset_type"], inplace=True)
    metadata_test_dataset_1.drop(columns=["dataset_type"], inplace=True)
    metadata_test_dataset_2.drop(columns=["dataset_type"], inplace=True)
    metadata_train.to_csv(train_metadata_file, index=False)
    metadata_test_dataset_1.to_csv(test_dataset_1_metadata_file, index=False)
    metadata_test_dataset_2.to_csv(test_dataset_2_metadata_file, index=False)

def combine_train_test_metadata(train_metadata_csv, test_metadata_csv, combined_metadata_csv):
    train_metadata = pd.read_csv(train_metadata_csv)
    test_metadata = pd.read_csv(test_metadata_csv)
    combined_metadata = pd.concat([train_metadata, test_metadata])
    combined_metadata.to_csv(combined_metadata_csv, index=False)

# usage example #

# if __name__ == '__main__':
#     source_path = "hsv2_data_for_competition/"
#     original_metadata_file = os.path.join(source_path, "dx_competition_sample_metadata.tsv")
#     modified_metadata_file = os.path.join(source_path, "modified_metadata.csv")
#     concatenated_repertoire_file = os.path.join(source_path, "dx_competition_HSV2_repertoires.tsv")
#     process_metadata(original_metadata_file, modified_metadata_file)
#     move_anonymized_data(modified_metadata_file, concatenated_repertoire_file, source_path)
#     train_metadata_file = os.path.join(source_path, "train_metadata.csv")
#     test_dataset_1_metadata_file = os.path.join(source_path, "test_dataset_1_metadata.csv")
#     test_dataset_2_metadata_file = os.path.join(source_path, "test_dataset_2_metadata.csv")
#     train_and_test_1_metadata_file = os.path.join(source_path, "train_and_test_dataset_1_metadata.csv")
#     train_and_test_2_metadata_file = os.path.join(source_path, "train_and_test_dataset_2_metadata.csv")
#     combine_train_test_metadata(train_metadata_file, test_dataset_1_metadata_file, train_and_test_1_metadata_file)
#     combine_train_test_metadata(train_metadata_file, test_dataset_2_metadata_file, train_and_test_2_metadata_file)