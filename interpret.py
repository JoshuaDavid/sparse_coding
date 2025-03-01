import argparse
import asyncio
from datetime import datetime
from functools import partial
import importlib
import json
import multiprocessing as mp
import os
import pickle
import requests
import sys
from typing import Any, Dict, Union, List, Callable, Optional, Tuple

from baukit import Trace
from datasets import load_dataset, ReadInstruction
from einops import rearrange
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA, PCA, NMF
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformers import GPT2Tokenizer, AutoTokenizer

from autoencoders.learned_dict import LearnedDict
from autoencoders.pca import BatchedPCA
from argparser import parse_args
from comparisons import NoCredentialsError
from utils import dotdict, make_tensor_name, upload_to_aws, get_activation_size, check_use_baukit
from nanoGPT_model import GPT
from activation_dataset import setup_data


# set OPENAI_API_KEY environment variable from secrets.json['openai_key']
# needs to be done before importing openai interp bits
with open("secrets.json") as f:
    secrets = json.load(f)
    os.environ["OPENAI_API_KEY"] = secrets["openai_key"]

mp.set_start_method("spawn", force=True)

from neuron_explainer.activations.activation_records import calculate_max_activation
from neuron_explainer.activations.activations import ActivationRecordSliceParams, ActivationRecord, NeuronRecord, NeuronId
from neuron_explainer.explanations.calibrated_simulator import UncalibratedNeuronSimulator
from neuron_explainer.explanations.explainer import TokenActivationPairExplainer
from neuron_explainer.explanations.prompt_builder import PromptFormat
from neuron_explainer.explanations.scoring import simulate_and_score, aggregate_scored_sequence_simulations
from neuron_explainer.explanations.simulator import ExplanationNeuronSimulator
from neuron_explainer.fast_dataclasses import loads


EXPLAINER_MODEL_NAME = "gpt-4" # "gpt-3.5-turbo"
SIMULATOR_MODEL_NAME = "text-davinci-003"

OPENAI_MAX_FRAGMENTS = 50000
OPENAI_FRAGMENT_LEN = 64
OPENAI_EXAMPLES_PER_SPLIT = 5
N_SPLITS = 4
TOTAL_EXAMPLES = OPENAI_EXAMPLES_PER_SPLIT * N_SPLITS
REPLACEMENT_CHAR = "�"
MAX_CONCURRENT = None


# Replaces the load_neuron function in neuron_explainer.activations.activations because couldn't get blobfile to work
def load_neuron(
    layer_index: Union[str, int], neuron_index: Union[str, int],
    dataset_path: str = "https://openaipublic.blob.core.windows.net/neuron-explainer/data/collated-activations",
) -> NeuronRecord:
    """Load the NeuronRecord for the specified neuron."""
    url = os.path.join(dataset_path, str(layer_index), f"{neuron_index}.json")
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Neuron record not found at {url}.")
    neuron_record = loads(response.content)

    if not isinstance(neuron_record, NeuronRecord):
        raise ValueError(
            f"Stored data incompatible with current version of NeuronRecord dataclass."
        )
    return neuron_record

def make_activation_dataset(cfg, model, total_activation_size: int = 512 * 1024 * 1024):
    if cfg.model_name in ["gpt2", "nanoGPT"]:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    elif cfg.model_name == "EleutherAI/pythia-70m-deduped":
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")
    else:
        raise NotImplementedError

    cfg.n_chunks = 1
    dataset_name = cfg.dataset_name.split("/")[-
    1] + "-" + cfg.model_name.split("/")[-1] + "-" + str(cfg.layer)
    cfg.dataset_folder = os.path.join(cfg.datasets_folder, dataset_name)
    if not os.path.exists(cfg.dataset_folder) or len(os.listdir(cfg.dataset_folder)) == 0:
        cfg.n_chunks = 1
        setup_data(
            tokenizer, 
            model,
            model_name=cfg.model_name,
            dataset_name=cfg.dataset_name,
            dataset_folder=cfg.dataset_folder,
            layer=cfg.layer,
            layer_loc=cfg.layer_loc,
            n_chunks=cfg.n_chunks,
            device=cfg.device
        )
    chunk_loc = os.path.join(cfg.dataset_folder, f"0.pkl")

    elem_size = 4
    activation_dim = get_activation_size(cfg.model_name, cfg.layer_loc)
    n_activations = total_activation_size // (elem_size * activation_dim)

    dataset = DataLoader(pickle.load(open(chunk_loc, "rb")), batch_size=n_activations, shuffle=True)
    return dataset, n_activations


def activation_ICA(dataset, n_activations):
    """
    Takes a tensor of activations and returns the ICA of the activations
    """
    ica = FastICA()
    print(f"Fitting ICA on {n_activations} activations")
    ica_start = datetime.now()
    ica.fit(next(iter(dataset))[0].cpu().numpy()) # 1GB of activations takes about 15m
    print(f"ICA fit in {datetime.now() - ica_start}")
    return ica


def activation_PCA(dataset, n_activations):
    pca = PCA()
    print(f"Fitting PCA on {n_activations} activations")
    pca_start = datetime.now()
    pca.fit(next(iter(dataset))[0].cpu().numpy()) # 1GB of activations takes about 40s
    print(f"PCA fit in {datetime.now() - pca_start}")
    return pca


def activation_NMF(dataset, n_activations):
    nmf = NMF()
    print(f"Fitting NMF on {n_activations} activations")
    nmf_start = datetime.now()
    data = next(iter(dataset))[0].cpu().numpy() # 1GB of activations takes an unknown but long time
    # NMF doesn't support negative values, so shift the data to be positive
    data -= data.min()
    nmf.fit(data)
    print(f"NMF fit in {datetime.now() - nmf_start}")
    return nmf


def make_feature_activation_dataset(
        model_name: str,
        model: HookedTransformer, 
        layer: int,
        layer_loc: str,
        activation_fn: Callable,
        feat_dim: int,
        device: str = "cpu",
        n_fragments = OPENAI_MAX_FRAGMENTS,
        random_fragment = True, # used for debugging
    ):
    """
    Takes a specified point of a model, and a dataset. 
    Returns a dataset which contains the activations of the model at that point, 
    for each fragment in the dataset, transformed into the feature space
    """
    use_baukit = check_use_baukit(model_name)

    sentence_dataset = load_dataset("openwebtext", split="train", streaming=True)

    if model_name == "nanoGPT":
        tokenizer_model = HookedTransformer.from_pretrained("gpt2")
    else:
        tokenizer_model = model
    
    tensor_name = make_tensor_name(layer, layer_loc, model_name)
    # make list of sentence, tokenization pairs
    
    iter_dataset = iter(sentence_dataset)

    # Make dataframe with columns for each feature, and rows for each sentence fragment
    # each row should also have the full sentence, the current tokens and the previous tokens

    n_thrown = 0
    n_added = 0
    batch_size = min(20, n_fragments)

    fragment_token_ids_list = []
    fragment_token_strs_list = []

    activation_means_table = np.zeros((n_fragments, feat_dim), dtype=np.float16)
    activation_maxes_table = np.zeros((n_fragments, feat_dim), dtype=np.float16)
    activation_data_table = np.zeros((n_fragments, feat_dim * OPENAI_FRAGMENT_LEN), dtype=np.float16)
    with torch.no_grad():
        while n_added < n_fragments:
            fragments: List[torch.Tensor] = []
            fragment_strs: List[str] = []
            while len(fragments) < batch_size:
                print(f"Added {n_added} fragments, thrown {n_thrown} fragments\t\t\t\t\t\t", end="\r")
                sentence = next(iter_dataset)
                # split the sentence into fragments
                sentence_tokens = tokenizer_model.to_tokens(sentence["text"], prepend_bos=False)
                n_tokens = sentence_tokens.shape[1]
                # get a random fragment from the sentence - only taking one fragment per sentence so examples aren't correlated]
                if random_fragment:
                    token_start = np.random.randint(0, n_tokens - OPENAI_FRAGMENT_LEN)
                else:
                    token_start = 0
                fragment_tokens = sentence_tokens[:, token_start:token_start + OPENAI_FRAGMENT_LEN]
                token_strs = tokenizer_model.to_str_tokens(fragment_tokens[0])
                if REPLACEMENT_CHAR in token_strs:
                    n_thrown += 1
                    continue

                fragment_strs.append(token_strs)
                fragments.append(fragment_tokens)
            
            tokens = torch.cat(fragments, dim=0)
            assert tokens.shape == (batch_size, OPENAI_FRAGMENT_LEN), tokens.shape

            if use_baukit:
                with Trace(model, tensor_name) as ret:
                    _ = model(tokens)
                    mlp_activation_data = ret.output.to(device)
                    mlp_activation_data = nn.functional.gelu(mlp_activation_data)
            else:
                _, cache = model.run_with_cache(tokens)
                mlp_activation_data = cache[tensor_name].to(device)

            for i in range(batch_size):
                fragment_tokens = tokens[i:i+1, :]
                activation_data = mlp_activation_data[i:i+1, :].squeeze(0)
                token_ids = fragment_tokens[0].tolist()
                    
                feature_activation_data = activation_fn(activation_data)
                feature_activation_means = torch.mean(feature_activation_data, dim=0)
                feature_activation_maxes = torch.max(feature_activation_data, dim=0)[0]

                activation_means_table[n_added, :] = feature_activation_means.cpu().numpy()[:feat_dim]
                activation_maxes_table[n_added, :] = feature_activation_maxes.cpu().numpy()[:feat_dim]

                feature_activation_data = feature_activation_data.cpu().numpy()[:, :feat_dim]

                activation_data_table[n_added, :] = feature_activation_data.flatten()
    
                fragment_token_ids_list.append(token_ids)
                fragment_token_strs_list.append(fragment_strs[i])
                
                n_added += 1

                if n_added >= n_fragments:
                    break
            
    print(f"Added {n_added} fragments, thrown {n_thrown} fragments")
    # Now we build the dataframe from the numpy arrays and the lists
    print(f"Making dataframe from {n_added} fragments")
    df = pd.DataFrame()
    df["fragment_token_ids"] = fragment_token_ids_list
    df["fragment_token_strs"] = fragment_token_strs_list
    means_column_names = [f"feature_{i}_mean" for i in range(feat_dim)]
    maxes_column_names = [f"feature_{i}_max" for i in range(feat_dim)]
    activations_column_names = [f"feature_{i}_activation_{j}" for j in range(OPENAI_FRAGMENT_LEN) for i in range(feat_dim)] # nested for loops are read left to right
    
    assert feature_activation_data.shape == (OPENAI_FRAGMENT_LEN, feat_dim)
    df = pd.concat([df, pd.DataFrame(activation_means_table, columns=means_column_names)], axis=1)
    df = pd.concat([df, pd.DataFrame(activation_maxes_table, columns=maxes_column_names)], axis=1)
    df = pd.concat([df, pd.DataFrame(activation_data_table, columns=activations_column_names)], axis=1)
    print(f"Threw away {n_thrown} fragments, made {len(df)} fragments")
    return df

async def main(cfg: dotdict) -> None:
    # Load model
    if cfg.model_name in ["gpt2", "EleutherAI/pythia-70m-deduped"]:
        model = HookedTransformer.from_pretrained(cfg.model_name, device=cfg.device)
        use_baukit = False
        if cfg.model_name == "gpt2":
            resid_width = 768
        elif cfg.model_name == "EleutherAI/pythia-70m-deduped":
            resid_width = 512
    elif cfg.model_name == "nanoGPT":
        model_dict = torch.load(open(cfg.model_path, "rb"), map_location="cpu")["model"]
        model_dict = {k.replace("_orig_mod.", ""): v for k, v in model_dict.items()}
        cfg_loc = cfg.model_path[:-3] + "cfg"  # cfg loc is same as model_loc but with .pt replaced with cfg.py
        cfg_loc = cfg_loc.replace("/", ".")
        model_cfg = importlib.import_module(cfg_loc).model_cfg
        model = GPT(model_cfg).to(cfg.device)
        model.load_state_dict(model_dict)
        use_baukit = True
        resid_width = 32
    else:
        raise ValueError("Model name not recognised")
    if cfg.layer_loc == "mlp":
        activation_width = resid_width * 4
    else:
        activation_width = resid_width
    
    # Load feature dict
    if cfg.activation_transform in ["feature_dict", "feature_no_bias", "neuron_basis_bias", "random_bias"]:
        assert cfg.load_interpret_autoencoder is not None
        autoencoder: LearnedDict = torch.load(cfg.load_interpret_autoencoder)
        autoencoder.to_device(cfg.device)

    if cfg.activation_transform in ["feature_dict", "feature_no_bias"]:
        feature_size = autoencoder.n_feats
    elif cfg.activation_transform in ["pca_learned"]:
        feature_size = activation_width * 2
    else:
        feature_size = activation_width

    if cfg.df_n_feats and cfg.df_n_feats < feature_size:
        feat_dim = cfg.df_n_feats
    else:
        feat_dim = feature_size
    
    if cfg.activation_transform in ["ica", "pca", "nmf"]:
        activation_dataset, n_activations = make_activation_dataset(cfg, model)

    activations_name = f"{cfg.model_name.split('/')[-1]}_layer{cfg.layer}_{cfg.layer_loc}"

    print(f"Using activation transform {cfg.activation_transform}")
    if cfg.activation_transform == "neuron_basis":
        activation_fn = lambda x: x
    elif cfg.activation_transform == "neuron_relu":
        activation_fn = torch.relu
    elif cfg.activation_transform == "neuron_basis_bias":
        bias = autoencoder.encoder_bias
        if not cfg.tied_ae:
            norms = torch.norm(autoencoder.encoder, 2, dim=-1)
            bias = bias / norms
        activation_fn = lambda x: torch.relu(x + bias)

    elif cfg.activation_transform == "ica":
        ica_path = os.path.join("auto_interp_results", activations_name, "ica_1gb.pkl")
        if os.path.exists(ica_path):
            print("Loading ICA")
            ica = pickle.load(open(ica_path, "rb"))
        else:
            ica = activation_ICA(activation_dataset, n_activations)
            os.makedirs(os.path.dirname(ica_path), exist_ok=True)
            pickle.dump(ica, open(ica_path, "wb"))
        
        activation_fn = lambda x: torch.tensor(ica.transform(x.cpu()))

    elif cfg.activation_transform == "pca":
        pca_path = os.path.join("auto_interp_results", activations_name, "pca_1gb.pkl")
        if os.path.exists(pca_path):
            print("Loading PCA")
            pca = pickle.load(open(pca_path, "rb"))
        else:
            pca = activation_PCA(activation_dataset, n_activations)
            os.makedirs(os.path.dirname(pca_path), exist_ok=True)
            pickle.dump(pca, open(pca_path, "wb"))
        
        activation_fn = lambda x: torch.tensor(pca.transform(x.cpu()))

    elif cfg.activation_transform == "pca_learned":
        pca_path = os.path.join("auto_interp_results", activations_name, "pca_dict.pt")
        if os.path.exists(pca_path):
            print("Loading PCA")
            pca = torch.load(pca_path)
        else:
            dataset = torch.load(f"/mnt/ssd-cluster/single_chunks/l{cfg.layer}_{cfg.layer_loc}/0.pt").to(cfg.device)
            pca = BatchedPCA(dataset.shape[1], cfg.device)
            print("Training PCA")
            batch_size = 5000
            for i in tqdm(range(0, len(dataset), batch_size)):
                j = min(i + batch_size, len(dataset))
                batch = dataset[i:j]
                pca.train_batch(batch)
            
            torch.save(pca, pca_path)

        pca_top_k = pca.to_topk_dict(cfg.top_k_pca)        
        activation_fn = pca_top_k.encode

    elif cfg.activation_transform == "nmf":
        nmf_path = os.path.join("auto_interp_results", activations_name, "nmf_1gb.pkl")
        if os.path.exists(nmf_path):
            print("Loading NMF")
            nmf = pickle.load(open(nmf_path, "rb"))
        else:
            nmf = activation_NMF(activation_dataset, n_activations)
            os.makedirs(os.path.dirname(nmf_path), exist_ok=True)
            pickle.dump(nmf, open(nmf_path, "wb"))

        activation_fn = lambda x: torch.tensor(nmf.transform(x.cpu()))

    elif cfg.activation_transform == "feature_dict":
        activation_fn = autoencoder.encode

    elif cfg.activation_transform == "feature_no_bias":
        activation_fn = partial(autoencoder.encode, bias=False)
        
    elif cfg.activation_transform == "random":
        random_path = os.path.join("auto_interp_results", activations_name, "random_dirs.pkl")
        if os.path.exists(random_path):
            print("Loading random directions")
            random_direction_matrix = pickle.load(open(random_path, "rb"))
        else:
            random_direction_matrix = torch.randn(activation_width, activation_width)
            random_direction_matrix = random_direction_matrix / torch.norm(random_direction_matrix, dim=1, keepdim=True)
            os.makedirs(os.path.dirname(random_path), exist_ok=True)
            pickle.dump(random_direction_matrix, open(random_path, "wb"))
        
        activation_fn = lambda x: torch.relu(x @ random_direction_matrix.to(cfg.device))

    elif cfg.activation_transform == "random_bias":
        random_path = os.path.join("auto_interp_results", activations_name, "random_dirs.pkl")
        if os.path.exists(random_path):
            print("Loading random directions")
            random_direction_matrix = pickle.load(open(random_path, "rb"))
        else:
            random_direction_matrix = torch.randn(activation_width, activation_width)
            os.makedirs(os.path.dirname(random_path), exist_ok=True)
            pickle.dump(random_direction_matrix, open(random_path, "wb"))
        
        bias = autoencoder.encoder_bias
        if not cfg.tied_ae:
            norms = torch.norm(autoencoder.encoder, 2, dim=-1)
            bias = bias / norms

        activation_fn = lambda x: torch.relu(x @ random_direction_matrix.to(cfg.device) + bias)

    else:
        raise ValueError(f"Activation transform {cfg.activation_transform} not recognised")

    if cfg.activation_transform == "feature_dict":
        transform_name = cfg.load_interpret_autoencoder.split("/")[-1][:-3]
    elif cfg.activation_transform == "pca_learned":
        transform_name = f"pca_top_k_{cfg.top_k_pca}"
    else:
        transform_name = cfg.activation_transform

    if cfg.sort_mode == "mean":
        transform_name += "_mean"

    if cfg.interp_name:
        transform_folder = os.path.join("auto_interp_results", activations_name, cfg.interp_name)
    else:
        transform_folder = os.path.join("auto_interp_results", activations_name, transform_name)
    df_loc = os.path.join(transform_folder, f"activation_df.hdf")

    if not (cfg.load_activation_dataset and os.path.exists(df_loc)) or cfg.refresh_data:
        base_df = make_feature_activation_dataset(
            cfg.model_name,
            model,
            layer=cfg.layer,
            layer_loc=cfg.layer_loc,
            activation_fn=activation_fn,
            device=cfg.device,
            feat_dim=feature_size
        )
        # save the dataset, saving each column separately so that we can retrive just the columns we want later
        print(f"Saving dataset to {df_loc}")
        os.makedirs(transform_folder, exist_ok=True)
        base_df.to_hdf(df_loc, key="df", mode="w")
    else:
        start_time = datetime.now()
        base_df = pd.read_hdf(df_loc)
        print(f"Loaded dataset in {datetime.now() - start_time}")


    # save the autoencoder being investigated
    os.makedirs(transform_folder, exist_ok=True)
    if cfg.activation_transform == "feature_dict":
        torch.save(autoencoder, os.path.join(transform_folder, "autoencoder.pt"))
        
    for feat_n in range(0, cfg.n_feats_explain):
        if os.path.exists(os.path.join(transform_folder, f"feature_{feat_n}")):
            print(f"Feature {feat_n} already exists, skipping")
            continue

        activation_col_names = [f"feature_{feat_n}_activation_{i}" for i in range(OPENAI_FRAGMENT_LEN)]
        read_fields = ["fragment_token_strs", f"feature_{feat_n}_mean", f"feature_{feat_n}_max", *activation_col_names]
        # check that the dataset has the required columns
        if not all([field in base_df.columns for field in read_fields]):
            print(f"Dataset does not have all required columns for feature {feat_n}, skipping")
            continue
        df = base_df[read_fields].copy()
        if cfg.sort_mode == "mean":
            sorted_df = df.sort_values(by=f"feature_{feat_n}_mean", ascending=False)
        else:
            sorted_df = df.sort_values(by=f"feature_{feat_n}_max", ascending=False)
        sorted_df = sorted_df.head(TOTAL_EXAMPLES)
        top_activation_records = []
        for i, row in sorted_df.iterrows():
            top_activation_records.append(ActivationRecord(row["fragment_token_strs"], [row[f"feature_{feat_n}_activation_{j}"] for j in range(OPENAI_FRAGMENT_LEN)]))
        
        random_activation_records: List[ActivationRecord] = []
        # Adding random fragments
        # random_df = df.sample(n=TOTAL_EXAMPLES)
        # for i, row in random_df.iterrows():
        #     random_activation_records.append(ActivationRecord(row["fragment_token_strs"], [row[f"feature_{feat_n}_activation_{j}"] for j in range(OPENAI_FRAGMENT_LEN)]))
        
        # making sure that the have some variation in each of the features, though need to be careful that this doesn't bias the results
        random_ordering = torch.randperm(len(df)).tolist()
        skip_feature = False
        while len(random_activation_records) < TOTAL_EXAMPLES:
            try:
                i = random_ordering.pop()
            except IndexError:  
                skip_feature = True
                break
            # if there are no activations for this fragment, skip it
            if df.iloc[i][f"feature_{feat_n}_mean"] == 0:
                continue
            random_activation_records.append(ActivationRecord(df.iloc[i]["fragment_token_strs"], [df.iloc[i][f"feature_{feat_n}_activation_{j}"] for j in range(OPENAI_FRAGMENT_LEN)]))
        if skip_feature:
            print(f"Skipping feature {feat_n} due to lack of activating examples")
            continue

        neuron_id = NeuronId(layer_index=2, neuron_index=feat_n)

        neuron_record = NeuronRecord(neuron_id=neuron_id, random_sample=random_activation_records, most_positive_activation_records=top_activation_records)
        slice_params = ActivationRecordSliceParams(n_examples_per_split=OPENAI_EXAMPLES_PER_SPLIT)
        train_activation_records = neuron_record.train_activation_records(slice_params)
        valid_activation_records = neuron_record.valid_activation_records(slice_params)

        explainer = TokenActivationPairExplainer(
            model_name=EXPLAINER_MODEL_NAME,
            prompt_format=PromptFormat.HARMONY_V4,
            max_concurrent=MAX_CONCURRENT,
        )
        explanations = await explainer.generate_explanations(
            all_activation_records=train_activation_records,
            max_activation=calculate_max_activation(train_activation_records),
            num_samples=1
        )
        assert len(explanations) == 1
        explanation = explanations[0]
        print(f"Feature {feat_n}, {explanation=}")

        # Simulate and score the explanation.
        format = PromptFormat.HARMONY_V4 if SIMULATOR_MODEL_NAME == "gpt-3.5-turbo" else PromptFormat.INSTRUCTION_FOLLOWING
        simulator = UncalibratedNeuronSimulator(
            ExplanationNeuronSimulator(
                SIMULATOR_MODEL_NAME,
                explanation,
                max_concurrent=MAX_CONCURRENT,
                prompt_format=format,
            )
        )
        scored_simulation = await simulate_and_score(simulator, valid_activation_records)
        score = scored_simulation.get_preferred_score()
        assert len(scored_simulation.scored_sequence_simulations) == 10
        top_only_score = aggregate_scored_sequence_simulations(scored_simulation.scored_sequence_simulations[:5]).get_preferred_score()
        random_only_score = aggregate_scored_sequence_simulations(scored_simulation.scored_sequence_simulations[5:]).get_preferred_score()
        print(f"Feature {feat_n}, score={score:.2f}, top_only_score={top_only_score:.2f}, random_only_score={random_only_score:.2f}")

        feature_name = f"feature_{feat_n}"
        feature_folder = os.path.join(transform_folder, feature_name)
        os.makedirs(feature_folder, exist_ok=True)
        pickle.dump(scored_simulation, open(os.path.join(feature_folder, "scored_simulation.pkl"), "wb"))
        pickle.dump(neuron_record, open(os.path.join(feature_folder, "neuron_record.pkl"), "wb"))
        # write a file with the explanation and the score
        with open(os.path.join(feature_folder, "explanation.txt"), "w") as f:
            f.write(f"{explanation}\nScore: {score:.2f}\nExplainer model: {EXPLAINER_MODEL_NAME}\nSimulator model: {SIMULATOR_MODEL_NAME}\n")
            f.write(f"Top only score: {top_only_score:.2f}\n")
            f.write(f"Random only score: {random_only_score:.2f}\n")
                
    
    if cfg.upload_to_aws:
        upload_to_aws(transform_folder)


def get_score(lines: List[str], mode: str):
    if mode == "top":
        return float(lines[-3].split(" ")[-1])
    elif mode == "random":
        return float(lines[-2].split(" ")[-1])
    elif mode == "top_random":
        score_line = [line for line in lines if "Score: " in line][0]
        return float(score_line.split(" ")[1])
    else:
        raise ValueError(f"Unknown mode: {mode}")

def run_folder(cfg: dotdict):
    base_folder = cfg.load_interpret_autoencoder
    all_encoders = os.listdir(cfg.load_interpret_autoencoder)
    all_encoders = [x for x in all_encoders if (x.endswith(".pt") or x.endswith(".pkl"))]
    print(f"Found {len(all_encoders)} encoders in {cfg.load_interpret_autoencoder}")
    for i, encoder in enumerate(all_encoders):
        print(f"Running encoder {i} of {len(all_encoders)}: {encoder}")
        cfg.load_interpret_autoencoder = os.path.join(base_folder, encoder)
        asyncio.run(main(cfg))
    

def make_tag_name(hparams: Dict) -> str:
    tag = ""
    if "tied" in hparams.keys():
        tag += f"tied_{hparams['tied']}"
    if "dict_size" in hparams.keys():
        tag += f"dict_size_{hparams['dict_size']}"
    if "l1_alpha" in hparams.keys():
        tag += f"l1_alpha_{hparams['l1_alpha']:.2}"
    if "bias_decay" in hparams.keys():
        tag += "0.0" if hparams["bias_decay"] == 0 else f"{hparams['bias_decay']:.1}"
    return tag


def run_from_grouped(cfg: dotdict, results_loc: str):
    """
    Run autointerpretation across a file of learned dicts as outputted by big_sweep.py or similar.
    Expects results_loc to a .pt file containing a list of tuples of (learned_dict, hparams_dict)
    """
    # First, read in the results file
    results = torch.load(results_loc)
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(os.path.join("auto_interp_results", time_str), exist_ok=True)
    # Now split the results out into separate files 
    for learned_dict, hparams_dict in results:
        filename = make_tag_name(hparams_dict) + ".pt"
        torch.save(learned_dict, os.path.join("auto_interp_results", time_str, filename))
    
    cfg.load_interpret_autoencoder = os.path.join("auto_interp_results", time_str)
    run_folder(cfg)

def read_scores(results_folder: str, score_mode: str = "top") -> Dict[str, Tuple[List[int], List[float]]]:
    scores: Dict[str, Tuple[List[int], List[float]]] = {}
    transforms = os.listdir(results_folder)
    transforms = [transform for transform in transforms if os.path.isdir(os.path.join(results_folder, transform))]
    if "sparse_coding" in transforms:
        transforms.remove("sparse_coding")
        transforms = ["sparse_coding"] + transforms
  
    for transform in transforms:
        transform_scores = []
        transform_ndxs = []
        # list all the features by looking for folders
        feat_folders = [x for x in os.listdir(os.path.join(results_folder, transform)) if x.startswith("feature_")]
        print(f"{transform=}, {len(feat_folders)=}")
        for feature_folder in feat_folders:
            feature_ndx = int(feature_folder.split("_")[1])
            folder = os.path.join(results_folder, transform, feature_folder)
            if not os.path.exists(folder):
                continue
            explanation_text = open(os.path.join(folder, "explanation.txt")).read()
            # score should be on the second line but if explanation had newlines could be on the third or below
            # score = float(explanation_text.split("\n")[1].split(" ")[1])
            lines = explanation_text.split("\n")
            score = get_score(lines, score_mode)

            print(f"{feature_ndx=}, {transform=}, {score=}")
            transform_scores.append(score)
            transform_ndxs.append(feature_ndx)
        
        scores[transform] = (transform_ndxs, transform_scores)
        
    return scores


def read_results(activation_name: str, score_mode: str, exclude_mean: bool = True) -> None:
    results_folder = os.path.join("auto_interp_results", activation_name)

    scores = read_scores(results_folder, score_mode) # Dict[str, Tuple[List[int], List[float]]], where the tuple is (feature_ndxs, scores)
    transforms = scores.keys()

    plt.clf() # clear the plot                
    
    # plot the scores as a violin plot
    colors = ["red", "blue", "green", "orange", "purple", "pink", "black", "brown", "cyan", "magenta", "grey"]

    # fix yrange from -0.2 to 0.6
    plt.ylim(-0.2, 0.6)
    # add horizontal grid lines every 0.1
    plt.yticks(np.arange(-0.2, 0.6, 0.1))
    plt.grid(axis="y", color="grey", linestyle="-", linewidth=0.5, alpha=0.3)
    # first we need to get the scores into a list of lists
    scores_list = [scores[transform][1] for transform in transforms]
    # remove any transforms that have no scores
    scores_list = [scores for scores in scores_list if len(scores) > 0]
    violin_parts = plt.violinplot(scores_list, showmeans=False, showextrema=False)
    for i, pc in enumerate(violin_parts['bodies']):
        pc.set_facecolor(colors[i % len(colors)])
        pc.set_edgecolor(colors[i % len(colors)])
        pc.set_alpha(0.3)

    # add x labels
    plt.xticks(np.arange(1, len(transforms) + 1), transforms, rotation=90)

    # add standard errors around the means but don't plot the means
    cis = [1.96 * np.std(scores[transform][1]) / np.sqrt(len(scores[transform][1])) for transform in transforms]
    for i, transform in enumerate(transforms):
        plt.errorbar(i+1, np.mean(scores[transform][1]), yerr=cis[i], fmt="o", color=colors[i % len(colors)], elinewidth=2, capsize=20)

    plt.title(f"{activation_name} {score_mode}")
    plt.xlabel("Transform")
    plt.ylabel("GPT-4-based interpretability score")
    plt.xticks(rotation=90)



    # and a thicker line at 0
    plt.axhline(y=0, linestyle="-", color="black", linewidth=1)

    plt.tight_layout()
    save_path = os.path.join(results_folder, f"{score_mode}_means_and_violin.png")
    print(f"Saving means and violin graph to {save_path}")
    plt.savefig(save_path)  


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "read_results":
        # parse --layer and --model_name from command line using custom parser
        argparser = argparse.ArgumentParser()
        argparser.add_argument("--layer", type=int, default=1)
        argparser.add_argument("--model_name", type=str, default="EleutherAI/pythia-70m-deduped")
        argparser.add_argument("--layer_loc", type=str, default="mlp")
        argparser.add_argument("--score_mode", type=str, default="top_random") # can be "top", "random", "top_random", "all"
        argparser.add_argument("--run_all", type=bool, default=False)
        argparser.add_argument("--exclude_mean", type=bool, default=True)
        cfg = argparser.parse_args(sys.argv[2:])

        if cfg.score_mode == "all":
            score_modes = ["top", "random", "top_random"]
        else:
            score_modes = [cfg.score_mode]      

        if cfg.run_all:
            activation_names = [x for x in os.listdir("auto_interp_results") if os.path.isdir(os.path.join("auto_interp_results", x))]
        else:
            activation_names = [f"{cfg.model_name.split('/')[-1]}_layer{cfg.layer}_{cfg.layer_loc}"]
        
        for activation_name in activation_names:
            for score_mode in score_modes:
                read_results(activation_name, score_mode, cfg.exclude_mean)
            
    elif len(sys.argv) > 1 and sys.argv[1] == "run_group":
        sys.argv.pop(1)
        default_cfg = parse_args()
        run_from_grouped(default_cfg, default_cfg.load_interpret_autoencoder)

    else:
        default_cfg = parse_args()
        default_cfg.chunk_size_gb = 10
        if os.path.isdir(default_cfg.load_interpret_autoencoder):
            run_folder(default_cfg)

        else:
            asyncio.run(main(default_cfg))