import argparse
import logging
import os
from time import sleep
from dataset_generation.multi_simairr import make_random_seed, run_simarirr
from dataset_generation.util import parse_user_yaml, write_yaml_file, makedir_if_not_exists


def expand_dict(original_dict):
    expanded_dicts = []
    for run_key, sub_dict in original_dict.items():
        min_length = min(len(v) for v in sub_dict.values())
        for i in range(min_length):
            new_sub_dict = {k: v[i] for k, v in sub_dict.items()}
            expanded_dicts.append({run_key: new_sub_dict})
    return expanded_dicts


def generate_yaml_path(super_path, usecase_name, rep, burden, reps_config):
    yaml_path = os.path.join(super_path, "simulation_scripts", str(usecase_name) +
                             "_" + "phenotype_burden_" + str(burden) + "_" + str(rep) + ".yaml")
    if "n_repertoires" in reps_config:
        yaml_path = os.path.join(super_path, "simulation_scripts", str(usecase_name) +
                                 "_" + "dataset_size_" + str(reps_config['n_repertoires']) + "_" + str(rep) + ".yaml")
    if "n_sequences" in reps_config:
        yaml_path = os.path.join(super_path, "simulation_scripts", str(usecase_name) +
                                 "_" + "sequencing_depth_" + str(reps_config['n_sequences']) + "_" + str(rep) + ".yaml")
    if "positive_label_rate" in reps_config:
        balance_rate = int(reps_config['positive_label_rate']*100)
        yaml_path = os.path.join(super_path, "simulation_scripts", str(usecase_name) +
                                 "_" + "positive_label_rate_" + str(balance_rate) + "_" + str(rep) + ".yaml")
    return yaml_path


def generate_output_path(super_path, usecase_name, rep, burden, reps_config):
    out_path = os.path.join(super_path, usecase_name, rep, "phenotype_burden_" + str(burden), "data")
    if "n_repertoires" in reps_config:
        out_path = os.path.join(super_path, usecase_name, rep, "dataset_size_" + str(reps_config['n_repertoires']),
                                "data")
    if "n_sequences" in reps_config:
        out_path = os.path.join(super_path, usecase_name, rep, "sequencing_depth_" + str(reps_config['n_sequences']),
                                "data")
    if "positive_label_rate" in reps_config:
        out_path = os.path.join(super_path, usecase_name, rep, "positive_label_rate_" +
                                str(int(reps_config['positive_label_rate']*100)), "data")
    return out_path


def run_multiple_configs(user_config):
    config = parse_user_yaml(user_config)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=logging.DEBUG,
                        filename=os.path.join(config['super_path'], "multi_simairr_log.txt"), filemode='a')
    reps = config['reps']
    rem_list = ['usecase_name', 'super_path', 'reps']
    for key in rem_list:
        del config[key]
    params_to_update = expand_dict(reps)
    previous_seeds = []
    makedir_if_not_exists(os.path.join(super_path, "simulation_scripts"), fail_if_exists=False)
    run_each_config(config, params_to_update, previous_seeds)


def run_each_config(config, params_to_update, previous_seeds):
    super_path = config['super_path']
    usecase_name = config['usecase_name']
    for new_params in params_to_update:
        rep = list(new_params.keys())[0]
        new_params = new_params[rep]
        if 'phenotype_burden' in new_params:
            burden = new_params['phenotype_burden']
        else:
            burden = config['phenotype_burden']
        new_params['output_path'] = generate_output_path(super_path, usecase_name, rep, burden, new_params)
        new_params['seed'] = make_random_seed(previous_seeds)
        yaml_path = generate_yaml_path(super_path, usecase_name, rep, burden, new_params)
        run_config(config, new_params, yaml_path)


def run_config(config, new_params, yaml_path):
    config_copy = config.copy()
    config_copy.update(new_params)
    write_yaml_file(config_copy, yaml_path)
    sleep(10)
    logging.info(f'simulating the following config file {yaml_path}')
    run_simarirr(yaml_path)


def execute():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--config_file', help='path to YAML specification file describing the desired multi '
                                                    'simulation parameters', required=True)
    args = parser.parse_args()
    run_multiple_configs(args.config_file)