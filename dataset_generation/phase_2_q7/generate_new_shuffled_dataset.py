import os
import secrets
import shutil
import pandas as pd


def generate_new_dataset(old_data_path, new_data_path):
    new_test_data_dir, new_train_data_dir, test_data_dir, test_metadata_file, train_data_dir, train_metadata_file = make_paths(
        new_data_path, old_data_path)
    train_metadata = make_new_metadata(train_metadata_file)
    test_metadata = make_new_metadata(test_metadata_file)
    copy_to_destination(new_train_data_dir, train_data_dir, train_metadata)
    copy_to_destination(new_test_data_dir, test_data_dir, test_metadata)
    train_metadata.to_csv(train_metadata_file, index=False)
    test_metadata.to_csv(test_metadata_file, index=False)
    train_metadata = train_metadata.drop(columns=["old_filename", "old_subject_id", "old_label_positive"])
    test_metadata = test_metadata.drop(columns=["old_filename", "old_subject_id", "old_label_positive"])
    train_metadata.to_csv(os.path.join(new_train_data_dir, "metadata.csv"), index=False)
    test_metadata.to_csv(os.path.join(new_data_path, "test_metadata.csv"), index=False)
    make_simulated_repertoires_dir(new_data_path)


def make_paths(new_data_path, old_data_path):
    train_data_dir = os.path.join(old_data_path, "train")
    test_data_dir = os.path.join(old_data_path, "test")
    train_metadata_file = os.path.join(train_data_dir, "metadata.csv")
    test_metadata_file = os.path.join(old_data_path, "test_metadata.csv")
    new_train_data_dir = os.path.join(new_data_path, "train")
    new_test_data_dir = os.path.join(new_data_path, "test")
    os.makedirs(new_train_data_dir, exist_ok=True)
    os.makedirs(new_test_data_dir, exist_ok=True)
    return new_test_data_dir, new_train_data_dir, test_data_dir, test_metadata_file, train_data_dir, train_metadata_file


def copy_to_destination(destination_dir, source_dir, metadata):
    for i in range(len(metadata)):
        print("Processing file", i)
        old_filename = metadata.loc[i, "old_filename"]
        new_filename = metadata.loc[i, "filename"]
        shutil.copyfile(os.path.join(source_dir, old_filename), os.path.join(destination_dir, new_filename))


def make_simulated_repertoires_dir(new_data_path):
    train_data_dir = os.path.join(new_data_path, "train")
    test_data_dir = os.path.join(new_data_path, "test")
    sim_data_dir = os.path.join(new_data_path, "simulated_repertoires")
    os.makedirs(sim_data_dir, exist_ok=True)
    for root, dirs, files in os.walk(train_data_dir):
        for file in files:
            shutil.copyfile(os.path.join(root, file), os.path.join(sim_data_dir, file))
    for root, dirs, files in os.walk(test_data_dir):
        for file in files:
            shutil.copyfile(os.path.join(root, file), os.path.join(sim_data_dir, file))
    train_metadata_file = os.path.join(train_data_dir, "metadata.csv")
    test_metadata_file = os.path.join(new_data_path, "test_metadata.csv")
    train_metadata = pd.read_csv(train_metadata_file)
    test_metadata = pd.read_csv(test_metadata_file)
    combined_metadata = pd.concat([train_metadata, test_metadata])
    combined_metadata = combined_metadata.sample(frac=1)
    combined_metadata.to_csv(os.path.join(sim_data_dir, "metadata.csv"), index=False)


def make_new_metadata(old_metadata_file):
    metadata = pd.read_csv(old_metadata_file)
    metadata = metadata.rename(columns={"filename": "old_filename", 'subject_id': 'old_subject_id',
                                        'label_positive': 'old_label_positive'})
    metadata["subject_id"] = [secrets.token_hex(16) for i in range(len(metadata))]
    metadata["filename"] = metadata["subject_id"] + ".tsv"
    metadata["label_positive"] = metadata["old_label_positive"].sample(frac=1).reset_index(drop=True)
    print("Number of old_label_positive and new label_positive that are same: ",
          (metadata["old_label_positive"] == metadata["label_positive"]).sum())
    print("len of metadata", len(metadata))
    return metadata


## usage example #
# if __name__ == '__main__':
#     generate_new_dataset("/path/to/original_experimental_dataset/",
#                          "/path/to/new_dataset_with_shuffled_labels/")
#     generate_new_dataset(
#         "/path/to/original_synthetic_dataset/",
#         "/path/to/new_synthetic_dataset_with_shuffled_labels/")
