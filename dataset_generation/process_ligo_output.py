import argparse
import glob
import os
import pandas as pd
from dataset_generation.util import makedir_if_not_exists


def convert_airr_to_olga(airr_file, output_path):
    airr = pd.read_csv(airr_file, sep="\t", header=0, index_col=None)
    airr = airr[["junction_aa", "v_call", "j_call"]]
    airr["v_call"] = airr["v_call"].str.replace(r"\*.*", "", regex=True)
    airr["j_call"] = airr["j_call"].str.replace(r"\*.*", "", regex=True)
    airr = airr.sample(frac=1, random_state=99)
    makedir_if_not_exists(output_path, fail_if_exists=False)
    airr_fn = os.path.join(output_path, os.path.basename(airr_file))
    airr.to_csv(airr_fn, sep="\t", index=False)


def process_airr_files(airr_dir, output_path):
    airr_files = glob.glob(airr_dir + "/*.tsv")
    for airr_file in airr_files:
        convert_airr_to_olga(airr_file, output_path)


def process_ligo_metadata(ligo_metadata_file, output_path):
    ligo_metadata = pd.read_csv(ligo_metadata_file, sep=",", header=0, index_col=None)
    ligo_metadata["subject_id"] = ligo_metadata["repertoire_id"]
    ligo_metadata = ligo_metadata[["subject_id", "filename", "label_positive"]]
    meta_fn = os.path.join(output_path, "metadata.csv")
    makedir_if_not_exists(output_path, fail_if_exists=False)
    ligo_metadata.to_csv(meta_fn, sep=",")


def aggregate_ligo_signal_sequences(ligo_output_path):
    ligo_signal_files = glob.glob(ligo_output_path + "/**/signal*.tsv", recursive=True)
    ligo_signal_files.sort()
    ligo_signal = pd.concat([pd.read_csv(f, sep="\t", header=0, index_col=None) for f in ligo_signal_files])
    ligo_signal = ligo_signal[["sequence_aa", "v_call", "j_call"]]
    ligo_signal["v_call"] = ligo_signal["v_call"].str.replace(r"\*.*", "", regex=True)
    ligo_signal["j_call"] = ligo_signal["j_call"].str.replace(r"\*.*", "", regex=True)
    ligo_signal.to_csv(os.path.join(ligo_output_path, "signal_components", "filtered_implantable_signal_pool.tsv"),
                       sep="\t", index=False, header=False)


def process_ligo_output(ligo_output_path):
    signal_comp_path = os.path.join(ligo_output_path, "signal_components")
    makedir_if_not_exists(signal_comp_path, fail_if_exists=False)
    aggregate_ligo_signal_sequences(ligo_output_path)
    ligo_metadata_file = os.path.join(ligo_output_path, "inst1", "exported_dataset", "airr", "metadata.csv")
    output_path = os.path.join(ligo_output_path, "simulated_repertoires")
    process_ligo_metadata(ligo_metadata_file, output_path)
    airr_dir = os.path.join(ligo_output_path, "inst1", "exported_dataset", "airr", "repertoires")
    process_airr_files(airr_dir, output_path)


def execute():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--ligo_output_path', help='path to specific "data" directory that contains ligo output', required=True)
    args = parser.parse_args()
    process_ligo_output(args.ligo_output_path)

