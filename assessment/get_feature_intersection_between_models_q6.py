import glob
import os
import pandas as pd


def get_feature_intersection_between_models(submission: pd.DataFrame, train_datasets: list) -> dict:
    relevant_data_for_venn = []
    relevant_cols = ['junction_aa', 'v_call', 'j_call']
    for train_dataset in train_datasets:
        sub_mask = submission['dataset'] == train_dataset
        relevant_submissions = submission.loc[sub_mask, relevant_cols]
        relevant_data_for_venn.append(relevant_submissions.head(500))
    sets = [
        set(df[relevant_cols].itertuples(index=False, name=None))
        for df in relevant_data_for_venn
    ]
    s0, s1, s2 = sets[0], sets[1], sets[2]
    print("s0", s0)
    print("s1", s1)
    print("s2", s2)
    feature_intersection_dict = {
        f"{train_datasets[0]}__{train_datasets[1]}": len(s0 & s1),
        f"{train_datasets[0]}__{train_datasets[2]}": len(s0 & s2),
        f"{train_datasets[1]}__{train_datasets[2]}": len(s1 & s2),
        "all_three": len(s0 & s1 & s2),
    }
    return feature_intersection_dict


if __name__ == '__main__':
    cmv_datasets = ["train_dataset_35", "train_dataset_37", "train_dataset_39"]
    t1d_datasets = ["train_dataset_30", "train_dataset_32", "train_dataset_34"]
    challenge_submission_files = []
    for i in range(10, 11):
        pattern = f"rank_{i}*.csv"
        challenge_submission_files.extend(glob.glob(pattern))
    challenge_submission_files = list(set(challenge_submission_files))

    cmv_rows = {}
    t1d_rows = {}

    for submissions_path in challenge_submission_files:
        if os.path.exists(submissions_path):
            print("Counting submission file:", submissions_path)
            basename = os.path.basename(submissions_path)
            submission = pd.read_csv(submissions_path, header=0)
            cmv_rows[basename] = get_feature_intersection_between_models(submission, cmv_datasets)
            t1d_rows[basename] = get_feature_intersection_between_models(submission, t1d_datasets)
    cmv_df = pd.DataFrame.from_dict(cmv_rows, orient='index')
    t1d_df = pd.DataFrame.from_dict(t1d_rows, orient='index')
    cmv_df.to_csv("q6_cmv_feature_intersection_counts_phase2_models.csv", sep="\t")
    t1d_df.to_csv("q6_t1d_feature_intersection_counts_phase2_models.csv", sep="\t")

