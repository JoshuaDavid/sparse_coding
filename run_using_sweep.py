from argparser import parse_args
import json
import torch
import torchopt
import numpy as np
from autoencoders.ensemble import FunctionalEnsemble
from autoencoders.sae_ensemble import FunctionalSAE
import itertools
from big_sweep import sweep
from collections import defaultdict

def sweep_l1_vals_and_dict_ratios(cfg):
    devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    
    l1_exp_range = range(cfg.l1_exp_low, cfg.l1_exp_high)
    l1_range = [cfg.l1_exp_base**exp for exp in l1_exp_range]
    dict_ratio_exp_range = range(cfg.dict_ratio_exp_low, cfg.dict_ratio_exp_high)
    dict_ratios = [cfg.dict_ratio_exp_base**exp for exp in dict_ratio_exp_range]
    dict_sizes = [int(cfg.activation_dim * ratio) for ratio in dict_ratios]

    runs = enumerate(itertools.product(l1_range, dict_sizes))

    models_by_device_and_dictsize = [defaultdict(list) for device in devices]
    for i, (l1_alpha, dict_size) in runs:
        device = i % len(devices)
        models_by_device_and_dictsize[device][dict_size].append(FunctionalSAE.init(
            cfg.activation_width,
            dict_size,
            l1_alpha,
            dtype=cfg.dtype
        ))

    ensembles = [
        (
            FunctionalEnsemble(
                models,
                FunctionalSAE,
                torchopt.adam,
                dict(lr=cfg.lr),
                device=device
            ),
            dict(
                batch_size=cfg.batch_size,
                device=device,
                dict_size=dict_size
            ),
            f'run_sweep_{i}'
        )
        for device, models_by_dict_size in zip(devices, models_by_device_and_dictsize)
        for dict_size, models in models_by_dict_size.items()
    ]

    return (
        # [(ensemble, arg, name)]
        ensembles,
        # dict_size is same for every ensemble
        ['dict_size'],
        # l1_alpha varies between models in the ensemble
        ['l1_alpha'],
        {
            'dict_size': dict_sizes,
            'l1_alpha': l1_range
        }
    )

if __name__ == '__main__':
    cfg = parse_args()

    # Shamelessly copied from big_sweep_experiments.py
    cfg.dtype = torch.float32
    cfg.lr = 3e-4
    cfg.use_synthetic_dataset = False
    # cfg.batch_size = 2048
    # cfg.gen_batch_size = 4096
    # cfg.n_ground_truth_components = 1024
    # cfg.activation_width = 512
    # cfg.noise_magnitude_scale = 0.001
    # cfg.feature_prob_decay = 0.99
    # cfg.feature_num_nonzero = 10

    if not hasattr(cfg, 'dataset_folder'):
        cfg.dataset_folder = cfg.datasets_folder
    if not hasattr(cfg, 'output_folder'):
        cfg.output_folder = cfg.outputs_folder

    print(json.dumps({
        k:v
        for k,v in cfg.items()
        if type(v) in [str, int, float, bool, dict, list, tuple, type(None)]
    } , indent=' '))

    sweep(sweep_l1_vals_and_dict_ratios, cfg);


