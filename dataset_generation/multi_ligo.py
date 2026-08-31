import argparse
import logging
import os
from itertools import product
from time import sleep
import numpy as np
import pandas as pd
from dataset_generation.write_pwm import write_motif_pattern_to_jaspar
from dataset_generation.util import parse_user_yaml, makedir_if_not_exists, write_yaml_file


def gen_output_path(super_path, usecase, run, phenotype_burden):
    return os.path.join(super_path, usecase, run, "phenotype_burden_" + str(phenotype_burden), "data")


def extract_motifs(filename, n):
    df = pd.read_csv(filename, index_col=0, header=None, names=['0'])
    sorted_df = df.sort_values(by='0', ascending=False)
    top5000 = sorted_df.head(5000)
    random_rows = top5000.sample(n=n)
    return random_rows.index.values.tolist()


def generate_pwm_files(motifs, output_dir):
    fn_paths = []
    for motif in motifs:
        file_prefix = motif.replace("*", "_") + ".jaspar"
        fn_path = os.path.join(output_dir, file_prefix)
        fn_paths.append(fn_path)
        write_motif_pattern_to_jaspar(motif_pattern=motif, filename=fn_path)
    return fn_paths


def generate_pwm_filepaths_dict(relative_freqs_filepaths_yaml):
    config = parse_user_yaml(yaml_file=relative_freqs_filepaths_yaml)
    config2 = {key: os.path.join(config['files_path'], val) for key, val in config['freq_files'].items()}
    output_dir = os.path.join(config['files_path'], "pwm_files")
    makedir_if_not_exists(output_dir, fail_if_exists=False)
    config3 = {key: extract_motifs(val, 1) for key, val in config2.items()}
    config4 = {key: generate_pwm_files(val, output_dir)[0] for key, val in config3.items()}
    return config4


def process_ligo_yaml(ligo_yaml_file, pws_filepaths: dict, witness_rate):
    ligo_yaml = parse_user_yaml(ligo_yaml_file)
    for key, val in pws_filepaths.items():
        ligo_yaml['definitions']['motifs'][key]['file_path'] = val
    random_seeds = list(set(np.random.randint(1, 999, size=20)))[0:6]
    random_seeds = [int(i) for i in random_seeds]
    for sim_item, signal in zip(['AIRR1', 'AIRR2', 'AIRR3', 'AIRR4', 'AIRR5'],
                                ['signal1', 'signal2', 'signal3', 'signal4', 'signal5']):
        ligo_yaml['definitions']['simulations']['sim1']['sim_items'][sim_item]['signals'][signal] = witness_rate
        ligo_yaml['definitions']['simulations']['sim1']['sim_items'][sim_item]['seed'] = random_seeds.pop()
    ligo_yaml['definitions']['simulations']['sim1']['sim_items']['AIRR6']['seed'] = random_seeds.pop()
    return ligo_yaml


def run_ligo(yaml_file_path, outpath):
    command = 'ligo ' + yaml_file_path + ' ' + outpath
    exit_code = os.system(command)
    if exit_code != 0:
        raise RuntimeError(f"Running LiGO failed:{command}.")


def multi_ligo(super_path, ligo_yaml_file, relative_freqs_filepaths_yaml, witness_rates: list):
    makedir_if_not_exists(os.path.join(super_path, "simulation_scripts"), fail_if_exists=False)
    user_config = parse_user_yaml(yaml_file=relative_freqs_filepaths_yaml)
    replication_space = user_config['reps']
    for run, witness_rate in product(replication_space, witness_rates):
        burden = int(witness_rate * 25000)
        outpath = gen_output_path(super_path, user_config["usecase_name"], run, burden)
        yaml_file_path = os.path.join(super_path, "simulation_scripts",
                                      str(user_config["usecase_name"]) +
                                      "_" + "phenotype_burden_" + str(burden) + "_" + str(run) + ".yaml")
        pws_filepaths = generate_pwm_filepaths_dict(relative_freqs_filepaths_yaml)
        ligo_yaml = process_ligo_yaml(ligo_yaml_file, pws_filepaths, witness_rate)
        write_yaml_file(ligo_yaml, yaml_file_path)
        sleep(10)
        logging.info(f'simulating the following config file {yaml_file_path}')
        run_ligo(yaml_file_path, outpath)


def execute():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--config_file', help='path to YAML specification file describing the desired multi '
                                                    'simulation parameters', required=True)
    args = parser.parse_args()
    user_config = parse_user_yaml(yaml_file=args.config_file)
    super_path = user_config['super_path']
    makedir_if_not_exists(super_path, fail_if_exists=False)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=logging.DEBUG,
                        filename=os.path.join(super_path, "multi_LiGO_log.txt"), filemode='a')
    witness_rates = user_config['witness_rates']
    ligo_yaml_file = user_config['ligo_yaml_file']
    multi_ligo(super_path, ligo_yaml_file, args.config_file, witness_rates)
