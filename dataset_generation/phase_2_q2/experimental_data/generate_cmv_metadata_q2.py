import itertools
import secrets
import pandas as pd
from dataset_generation.multi_simairr import make_random_seed


def generate_cmv_metadata_files(metadata_file, out_path, q_num, varying_attribute, run_number, train_size, test_size,
                                varying_value, random_state):
    test_metadata, train_metadata = read_and_process_metadata(metadata_file)
    sampled_train_metadata = sample_metadata(train_metadata, train_size, random_state)
    sampled_test_metadata = sample_metadata(test_metadata, test_size, random_state)
    combined_train_and_test_metadata = pd.concat([sampled_train_metadata, sampled_test_metadata])
    combined_train_and_test_metadata = combined_train_and_test_metadata.sample(frac=1, random_state=random_state)
    combined_metadata_fn, test_metadata_fn, train_metadata_fn = generate_cmv_metadata_paths(out_path, q_num,
                                                                                            varying_attribute,
                                                                                            varying_value, run_number)
    sampled_train_metadata.to_csv(train_metadata_fn, index=False)
    sampled_test_metadata.to_csv(test_metadata_fn, index=False)
    combined_train_and_test_metadata.to_csv(combined_metadata_fn, index=False)


def generate_cmv_metadata_paths(out_path, q_num, varying_attribute, varying_value, run_number):
    train_metadata_fn = f"{out_path}/{q_num}_cmv_metadata_files/{q_num}_cmv_{varying_attribute}_{varying_value}_run{run_number}_train_metadata.csv"
    test_metadata_fn = f"{out_path}/{q_num}_cmv_metadata_files/{q_num}_cmv_{varying_attribute}_{varying_value}_run{run_number}_test_metadata.csv"
    combined_metadata_fn = f"{out_path}/{q_num}_cmv_metadata_files/{q_num}_cmv_{varying_attribute}_{varying_value}_run{run_number}_train_and_test_metadata.csv"
    return combined_metadata_fn, test_metadata_fn, train_metadata_fn


def read_and_process_metadata(metadata_file):
    metadata = pd.read_csv(metadata_file)
    metadata = metadata.rename(columns={"filename": "original_filename", "sim_item": "label_positive"})
    metadata["subject_id"] = [secrets.token_hex(16) for i in range(len(metadata))]
    metadata["filename"] = metadata["subject_id"] + ".tsv"
    test_metadata = metadata[(metadata["seq_depth"] <= 160000)]
    train_metadata = metadata[(metadata["seq_depth"] > 160000)]
    return test_metadata, train_metadata


def sample_metadata(metadata, n_examples, random_state):
    relevant_cols = ['original_filename', 'subject_id', 'filename', 'label_positive']
    metadata = metadata[relevant_cols]
    n_pos_examples = n_examples // 2
    n_neg_examples = n_examples - n_pos_examples
    pos_examples = metadata[metadata["label_positive"] == True]
    pos_examples = pos_examples.sample(n=n_pos_examples, random_state=random_state)
    neg_examples = metadata[metadata["label_positive"] == False]
    neg_examples = neg_examples.sample(n=n_neg_examples, random_state=random_state)
    sampled_metadata = pd.concat([pos_examples, neg_examples])
    sampled_metadata = sampled_metadata.sample(frac=1, random_state=random_state)
    return sampled_metadata


## usage example #
# if __name__ == '__main__':
#     params = itertools.product([(100, 200), (200, 200), (400, 200)], [1, 2, 3])
#     previous_seeds = []
#     for (train_size, test_size), run_number in params:
#         seed = make_random_seed(previous_seeds)
#         generate_cmv_metadata_files("/path/to/original_CMV_metadata.csv",
#                                     "/path/to/output_dir_for_writing_training_and_test_metadata_files",
#                                     "q2", "training_dataset_size", run_number, train_size,
#                                     test_size, train_size, seed)
