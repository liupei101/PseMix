"""
Entry filename: main.py

Code is forked from https://github.com/hugochan/IDGL as it provides a flexible way 
to configure hyper-parameters and evaluate model performance. Great thanks to the authors.
"""
import argparse
import yaml
import numpy as np

from model import ClfHandler
from utils.func import args_grid, print_config


def main(handler, config):
    model = handler(config)
    if config['test']:
        metrics = model.exec_test()
    else:
        metrics = model.exec()
    print('[INFO] Metrics:', metrics)

def multi_run_main(handler, config):
    hyperparams = []
    for k, v in config.items():
        if isinstance(v, list):
            hyperparams.append(k)

    configs = args_grid(config)
    for cnf in configs:
        print('\n')
        for k in hyperparams:
            cnf['save_path'] += '-{}_{}'.format(k, cnf[k])
        model = handler(cnf)
        if cnf['test']:
            print(cnf['test_save_path'])
            metrics = model.exec_test()
        else:
            print(cnf['save_path'])
            metrics = model.exec()
        print('[INFO] Metrics:', metrics)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-f', required=True, type=str, default='config/cfg_clf_mix.yml', help='path to the config file')
    parser.add_argument('--handler', '-d', required=True, type=str, default='clf', help='model handler (clf or others)')
    parser.add_argument('--multi_run', action='store_true', help='if execute multi-runs')
    args = vars(parser.parse_args())
    return args

def get_config(config_path="config/config.yml"):
    with open(config_path, "r") as setting:
        config = yaml.load(setting, Loader=yaml.FullLoader)
    return config

if __name__ == '__main__':
    cfg = get_args()
    config = get_config(cfg['config'])
    print_config(config)
    if cfg['handler'] == 'clf':
        handler = ClfHandler
    else:
        handler = None
    if cfg['multi_run']:
        multi_run_main(handler, config)
    else:
        main(handler, config)
