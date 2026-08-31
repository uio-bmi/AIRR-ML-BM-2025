import os
import secrets
import pandas as pd


def generate_t1d_metadata_files(metadata_file, out_path, dataset_type="test"):
    pos_metadata_file = os.path.join(out_path, f"{dataset_type}_t1d_pos_metadata.csv")
    neg_metadata_file = os.path.join(out_path, f"{dataset_type}_t1d_neg_metadata.csv")
    if dataset_type == "test":
        n_examples = 100
    else:
        n_examples = 200
    metadata = pd.read_csv(metadata_file)
    metadata["subject_id"] = [secrets.token_hex(16) for i in range(len(metadata))]
    metadata["filename"] = metadata["subject_id"] + ".tsv"
    relevant_cols = ['original_filename', 'subject_id', 'filename', 'label_positive']
    metadata = metadata[relevant_cols]
    pos_examples = metadata[metadata["label_positive"] == True]
    pos_examples = pos_examples.sample(n=n_examples, random_state=2709)
    neg_examples = metadata[metadata["label_positive"] == False]
    neg_examples = neg_examples.sample(n=n_examples, random_state=2709)
    pos_examples.to_csv(pos_metadata_file, index=False)
    neg_examples.to_csv(neg_metadata_file, index=False)


def generate_cmv_metadata_files(metadata_file, out_path, dataset_type="test"):
    metadata = pd.read_csv(metadata_file)
    metadata = metadata.rename(columns={"filename": "original_filename", "sim_item": "label_positive"})
    metadata["subject_id"] = [secrets.token_hex(16) for i in range(len(metadata))]
    metadata["filename"] = metadata["subject_id"] + ".tsv"
    pos_metadata_file = os.path.join(out_path, f"{dataset_type}_cmv_pos_metadata.csv")
    neg_metadata_file = os.path.join(out_path, f"{dataset_type}_cmv_neg_metadata.csv")
    if dataset_type == "test":
        n_examples = 100
        metadata = metadata[(metadata["seq_depth"] >= 50000) & (metadata["seq_depth"] <= 160000)]
    else:
        n_examples = 200
        metadata = metadata[(metadata["seq_depth"] > 160000)]
    relevant_cols = ['original_filename', 'subject_id', 'filename', 'label_positive']
    metadata = metadata[relevant_cols]
    pos_examples = metadata[metadata["label_positive"] == True]
    pos_examples = pos_examples.sample(n=n_examples, random_state=2709)
    neg_examples = metadata[metadata["label_positive"] == False]
    neg_examples = neg_examples.sample(n=n_examples, random_state=2709)
    pos_examples.to_csv(pos_metadata_file, index=False)
    neg_examples.to_csv(neg_metadata_file, index=False)


def generate_covid_train_and_test(metadata_file, out_path, n_test_examples=100, n_train_examples=200):
    metadata = pd.read_csv(metadata_file, sep='\t', header=0, index_col=False)
    metadata = metadata[(metadata['total_rearrangements'] >= 50000) & (metadata['total_rearrangements'] <= 140000)]
    metadata['original_filename'] = metadata['sample_name'] + '.tsv'
    metadata.reset_index(drop=True, inplace=True)
    metadata["subject_id"] = [secrets.token_hex(16) for i in range(len(metadata))]
    metadata["filename"] = metadata["subject_id"] + ".tsv"
    metadata['label_positive'] = True
    relevant_cols = ['original_filename', 'subject_id', 'filename', 'label_positive']
    metadata = metadata[relevant_cols]
    metadata = metadata.sample(n=n_test_examples + n_train_examples, random_state=2709)
    test_metadata = metadata[:n_test_examples]
    train_metadata = metadata[n_test_examples:]
    test_metadata.to_csv(os.path.join(out_path, "test_covid_pos_metadata.csv"), index=False)
    train_metadata.to_csv(os.path.join(out_path, "train_covid_pos_metadata.csv"), index=False)


def concatenate_t1d_metadata_files(files_path):
    metadata_files = [f for f in os.listdir(files_path) if "t1d" in f]
    metadata_dfs = []
    for metadata_file in metadata_files:
        metadata_df = pd.read_csv(os.path.join(files_path, metadata_file))
        metadata_dfs.append(metadata_df)
    concatenated_metadata = pd.concat(metadata_dfs, ignore_index=True)
    concatenated_metadata["dataset"] = 1
    concatenated_metadata = concatenated_metadata.drop_duplicates()
    concatenated_metadata.to_csv(os.path.join(files_path, "q6_t1d_metadata.csv"), index=False)


## usage example#
# if __name__ == '__main__':
#     for dataset_type in ["test", "train"]:
#         generate_t1d_metadata_files(metadata_file=f"/path/to/t1d_{dataset_type}_dataset_1_full_metadata.csv",
#                                     out_path="/path/to/q6_metadata_files/", dataset_type=dataset_type)
#         generate_cmv_metadata_files(
#             metadata_file="/path/to/metadata_for_q6_airr_ml_bm.csv",
#             out_path="/path/to/q6_metadata_files/", dataset_type=dataset_type)
#     generate_covid_train_and_test("/path/to/covid_metadata.tsv",
#                                   "/path/to/q6_metadata_files/")
#     concatenate_t1d_metadata_files("/path/to/q6_metadata_files/")

