import logging
import os
import secrets
import random
import pandas as pd

from dataset_generation.kaggle_data.experimental_data.split_t1d_metadata_kaggle import combine_train_test_metadata


def process_emerson_metadata(metadata_csv, out_path, train_size, test_size,
                             random_state, negative_control=False):
    metadata, reps = read_and_process_metadata(metadata_csv)
    handle_negative_control(metadata, negative_control, reps)
    train_metadata = generate_train_metadata(metadata, random_state, train_size)
    test_metadata = generate_test_metadata(metadata, random_state, test_size, train_metadata)
    write_metadata_files_to_disk(out_path, test_metadata, train_metadata)


def write_metadata_files_to_disk(out_path, test_metadata, train_metadata):
    train_metadata_file = os.path.join(out_path, "train_metadata_all_fields.csv")
    test_metadata_file = os.path.join(out_path, "test_metadata_all_fields.csv")
    train_metadata.to_csv(train_metadata_file, index=False)
    test_metadata.to_csv(test_metadata_file, index=False)
    relevant_cols = ['subject_id', 'filename', 'label_positive']
    train_metadata = train_metadata[relevant_cols]
    test_metadata = test_metadata[relevant_cols]
    relevant_train_metadata_file = os.path.join(out_path, "train_metadata.csv")
    relevant_test_metadata_file = os.path.join(out_path, "test_metadata.csv")
    train_metadata.to_csv(relevant_train_metadata_file, index=False)
    test_metadata.to_csv(relevant_test_metadata_file, index=False)
    train_and_test_metadata_file = os.path.join(out_path, "train_and_test_metadata.csv")
    combine_train_test_metadata(relevant_train_metadata_file, relevant_test_metadata_file, train_and_test_metadata_file)


def generate_test_metadata(metadata, random_state, test_size, train_metadata):
    test_metadata = metadata.drop(train_metadata.index)
    desired_size = int(test_size / 2)
    test_metadata_pos = test_metadata[test_metadata['label_positive']]
    test_metadata_neg = test_metadata[~test_metadata['label_positive']]
    test_pos = test_metadata_pos.sample(n=desired_size, random_state=random_state)
    test_neg = test_metadata_neg.sample(n=desired_size, random_state=random_state)
    test_metadata = pd.concat([test_pos, test_neg])
    test_metadata = test_metadata.sample(frac=1)
    return test_metadata


def generate_train_metadata(metadata, random_state, train_size):
    desired_size = int(train_size / 2)
    metadata_pos = metadata[metadata['label_positive']]
    metadata_neg = metadata[~metadata['label_positive']]
    train_pos = metadata_pos.sample(n=desired_size, random_state=random_state)
    train_neg = metadata_neg.sample(n=desired_size, random_state=random_state)
    train_metadata = pd.concat([train_pos, train_neg])
    train_metadata = train_metadata.sample(frac=1)
    return train_metadata


def handle_negative_control(metadata, negative_control, reps):
    if negative_control:
        num_reps = len(reps)
        num_true = num_reps // 2
        num_false = num_reps - num_true
        labels = [True] * num_true + [False] * num_false
        random.shuffle(labels)
        logging.info('using random labels because of negative control argument')
        metadata['label_positive'] = labels


def read_and_process_metadata(metadata_csv):
    metadata = pd.read_csv(metadata_csv)
    metadata = metadata.sample(frac=1)
    assert list(metadata.columns) == ['repertoire_id', 'sex', 'age', 'sim_item', 'filename', 'identifier']
    metadata = metadata.rename(columns={"sim_item": "label_positive", "filename": "original_filename"})
    reps = metadata['original_filename'].tolist()
    proxy_subject_ids = [secrets.token_hex(16) for i in range(len(reps))]
    proxy_fns = [subject_id + ".tsv" for subject_id in proxy_subject_ids]
    metadata['subject_id'] = proxy_subject_ids
    metadata['filename'] = proxy_fns
    return metadata, reps

# usage example #
# if __name__ == '__main__':
# process_emerson_metadata("/path/to/original_CMV_metadata.csv",
#                          "/path/to/output_dir_for_writing_training_and_test_metadata_files",
#                          400, 200, 2709, negative_control=False)
