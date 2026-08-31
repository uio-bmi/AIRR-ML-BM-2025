import pandas as pd
from dataset_generation.kaggle_data.experimental_data.split_t1d_metadata_kaggle import write_relevant_fields_to_csv, \
    combine_train_test_metadata


def split_t1d_metadata_phase_2_q1(metadata_csv, train_metadata_file, test_metadata_file, train_size, test_size,
                                  random_state):
    metadata = pd.read_csv(metadata_csv)
    metadata['label_positive'] = metadata['ML_class'] == 'T1D'
    train_metadata = generate_train_metadata(metadata, random_state, train_size)
    test_metadata = generate_test_metadata(metadata, random_state, test_size)
    train_metadata.to_csv(train_metadata_file, index=False)
    test_metadata.to_csv(test_metadata_file, index=False)


def generate_test_metadata(metadata, random_state, test_size):
    test_dataset_3 = metadata[metadata['dataset'] != 1]
    test_dataset_3_positive = test_dataset_3[test_dataset_3['label_positive']]
    test_dataset_3_negative = test_dataset_3[~test_dataset_3['label_positive']]
    desired_size = int(test_size / 2)
    if desired_size > len(test_dataset_3_negative):
        neg_desired_size = len(test_dataset_3_negative)
        pos_desired_size = test_size - neg_desired_size
    else:
        neg_desired_size = desired_size
        pos_desired_size = desired_size
    test_pos = test_dataset_3_positive.sample(n=pos_desired_size, random_state=random_state)
    test_neg = test_dataset_3_negative.sample(n=neg_desired_size, random_state=random_state)
    test_dataset = pd.concat([test_pos, test_neg])
    test_dataset = test_dataset.sample(frac=1)
    return test_dataset


def generate_train_metadata(metadata, random_state, train_size):
    cohort_1 = metadata[metadata['dataset'] == 1]
    cohort_1_positive = cohort_1[cohort_1['label_positive']]
    cohort_1_negative = cohort_1[~cohort_1['label_positive']]
    desired_size = int(train_size / 2)
    train_pos = cohort_1_positive.sample(n=desired_size, random_state=random_state)
    train_neg = cohort_1_negative.sample(n=desired_size, random_state=random_state)
    train_dataset = pd.concat([train_pos, train_neg])
    train_dataset = train_dataset.sample(frac=1)
    return train_dataset

def generate_metadata_files(base_path, metadata_filename, q_num, varying_attribute, varying_value, run_number,
                            train_size, test_size, random_state):
    split_t1d_metadata_phase_2_q1(
        f"{base_path}/{metadata_filename}",
        f"{base_path}/{q_num}_metadata_files/{q_num}_t1d_{varying_attribute}_{varying_value}_run{run_number}_full_metadata.csv",
        f"{base_path}/{q_num}_metadata_files/{q_num}_t1d_{varying_attribute}_{varying_value}_run{run_number}_test_full_metadata.csv",
        train_size,
        test_size,
        random_state)
    write_relevant_fields_to_csv(
        f"{base_path}/{q_num}_metadata_files/{q_num}_t1d_{varying_attribute}_{varying_value}_run{run_number}_full_metadata.csv",
        f"{base_path}/{q_num}_metadata_files/{q_num}_t1d_{varying_attribute}_{varying_value}_run{run_number}_relevant_metadata.csv")
    write_relevant_fields_to_csv(
        f"{base_path}/{q_num}_metadata_files/{q_num}_t1d_{varying_attribute}_{varying_value}_run{run_number}_test_full_metadata.csv",
        f"{base_path}/{q_num}_metadata_files/{q_num}_t1d_{varying_attribute}_{varying_value}_run{run_number}_test_relevant_metadata.csv")
    combine_train_test_metadata(
        f"{base_path}/{q_num}_metadata_files/{q_num}_t1d_{varying_attribute}_{varying_value}_run{run_number}_relevant_metadata.csv",
        f"{base_path}/{q_num}_metadata_files/{q_num}_t1d_{varying_attribute}_{varying_value}_run{run_number}_test_relevant_metadata.csv",
        f"{base_path}/{q_num}_metadata_files/{q_num}_t1d_{varying_attribute}_{varying_value}_run{run_number}_train_and_test_relevant_metadata.csv")
    combine_train_test_metadata(
        f"{base_path}/{q_num}_metadata_files/{q_num}_t1d_{varying_attribute}_{varying_value}_run{run_number}_full_metadata.csv",
        f"{base_path}/{q_num}_metadata_files/{q_num}_t1d_{varying_attribute}_{varying_value}_run{run_number}_test_full_metadata.csv",
        f"{base_path}/{q_num}_metadata_files/{q_num}_t1d_{varying_attribute}_{varying_value}_run{run_number}_train_and_test_full_metadata.csv")


## usage example #
# if __name__ == '__main__':
#     generate_metadata_files("/path/to/q1_t1d_metadata_files","cohort1_2_3_anonymized.csv", "q1",
#                             "training_dataset", 1, 1, 400, 400, 2709)

