import torch
import numpy as np
from dataclasses import dataclass
from circuitsvis.activations import text_neuron_activations
from einops import rearrange

def hyperparams_match(matchers, hyperparams):
    for k in set(hyperparams.keys()).intersection(matchers.keys()):
        if not matchers[k](hyperparams[k]):
            return False
    return True

@dataclass
class LocatedAutoencoder:
    location: str
    autoencoder: object
    hyperparams: dict

def load_autoencoders(desired_hyperparams, hyperparams_by_path):
    located_autoencoders = []
    for dict_path, path_hyperparams in hyperparams_by_path.items():
        # Check if the hyperparams match
        if hyperparams_match(desired_hyperparams, path_hyperparams):
            # If they do, load the dictionaries at the path
            path_contents = torch.load(dict_path)
            for autoencoder, ae_hyperparams in path_contents:
                if hyperparams_match(desired_hyperparams, ae_hyperparams):
                    located_autoencoders.append(LocatedAutoencoder(
                        location=dict_path,
                        autoencoder=autoencoder,
                        hyperparams=dict(**path_hyperparams, **ae_hyperparams)
                    ))
    return located_autoencoders

def get_feature_datapoints(feature_index, dictionary_activations, dataset, k=10, setting="max"):
    best_feature_activations = dictionary_activations[:, feature_index]
    # Sort the features by activation, get the indices
    if setting=="max":
        found_indices = torch.argsort(best_feature_activations, descending=True)[:k]
    elif setting=="uniform":
        min_value = torch.min(best_feature_activations)
        max_value = torch.max(best_feature_activations)

        # Define the number of bins
        num_bins = k

        # Calculate the bin boundaries as linear interpolation between min and max
        bin_boundaries = torch.linspace(min_value, max_value, num_bins + 1)

        # Assign each activation to its respective bin
        bins = torch.bucketize(best_feature_activations, bin_boundaries)

        # Initialize a list to store the sampled indices
        sampled_indices = []

        # Sample from each bin
        for bin_idx in torch.unique(bins):
            # Get the indices corresponding to the current bin
            bin_indices = torch.nonzero(bins == bin_idx, as_tuple=False).squeeze(dim=1)
            
            # Randomly sample from the current bin
            sampled_indices.extend(np.random.choice(bin_indices, size=1, replace=False))

        # Convert the sampled indices to a PyTorch tensor & reverse order
        found_indices = torch.tensor(sampled_indices).long().flip(dims=[0])
    else: # random
        # get nonzero indices
        nonzero_indices = torch.nonzero(best_feature_activations)[:, 0]
        # shuffle
        shuffled_indices = nonzero_indices[torch.randperm(nonzero_indices.shape[0])]
        found_indices = shuffled_indices[:k]
    datapoint_indices =[np.unravel_index(i, (datapoints, token_amount)) for i in found_indices]
    text_list = []
    full_text = []
    token_list = []
    full_token_list = []
    for md, s_ind in datapoint_indices:
        md = int(md)
        s_ind = int(s_ind)
        full_tok = torch.tensor(dataset[md]["input_ids"])
        full_text.append(model.tokenizer.decode(full_tok))
        tok = dataset[md]["input_ids"][:s_ind+1]
        text = model.tokenizer.decode(tok)
        text_list.append(text)
        token_list.append(tok)
        full_token_list.append(full_tok)
    return text_list, full_text, token_list, full_token_list

def get_neuron_activation(token, feature, autoencoder, model, cache_name):
    with torch.no_grad():
        _, cache = model.run_with_cache(token.to(model.cfg.device))
        neuron_act_batch = cache[cache_name]
        b, s, n = neuron_act_batch.shape
        ae_input_batch = rearrange(neuron_act_batch, 'b s n -> (b s) n')
        act = rearrange(
            autoencoder.encode(ae_input_batch),
            '(b s) n -> b s n',
            b=b,
            s=s
        )
    return act[0, :, feature].tolist()

def ablate_text(text, feature, autoencoder, model, cache_name, setting="plot"):
    if isinstance(text, str):
        text = [text]
    display_text_list = []
    activation_list = []
    for t in text:
        # Convert text into tokens
        if isinstance(t, str): # If the text is a list of tokens
            split_text = model.to_str_tokens(t, prepend_bos=False)
            tokens = model.to_tokens(t, prepend_bos=False)
        else: # t equals tokens
            tokens = t
            split_text = model.to_str_tokens(t, prepend_bos=False)
        seq_size = tokens.shape[1]
        if(seq_size == 1): # If the text is a single token, we can't ablate it
            continue
        original = get_neuron_activation(tokens, feature, autoencoder, model, cache_name)[-1]
        changed_activations = torch.zeros(seq_size, device=model.cfg.device).cpu()
        for i in range(seq_size):
            # Remove the i'th token from the input
            ablated_tokens = torch.cat((tokens[:,:i], tokens[:,i+1:]), dim=1)
            changed_activations[i] += get_neuron_activation(ablated_tokens, feature, autoencoder, model, cache_name)[-1]
        changed_activations -= original
        display_text_list += [x.replace('\n', '\\newline') for x in split_text] + ["\n"]
        activation_list += changed_activations.tolist() + [0.0]
    activation_list = torch.tensor(activation_list).reshape(-1,1,1)
    if setting == "plot":
        return text_neuron_activations(tokens=display_text_list, activations=activation_list)
    else:
        return display_text_list, activation_list

def visualize_text(text, feature, autoencoder, model, cache_name, setting="plot"):
    if isinstance(text, str):
        text = [text]
    display_text_list = []
    act_list = []
    for t in text:
        if isinstance(t, str): # If the text is a list of tokens
            split_text = model.to_str_tokens(t, prepend_bos=False)
            token = model.to_tokens(t, prepend_bos=False)
        else: # t are tokens
            token = t
            split_text = model.to_str_tokens(t, prepend_bos=False)
        display_text_list += [x.replace('\n', '\\newline') for x in split_text] + ["\n"]
        act_list += get_neuron_activation(token, feature, autoencoder, model, cache_name) + [0.0]
    act_list = torch.tensor(act_list).reshape(-1,1,1)
    return text_neuron_activations(tokens=display_text_list, activations=act_list)

def ablate_feature_direction(tokens, feature, model, cache_name, autoencoder):
    def ablation_hook(value, hook):
        # Rearrange to fit autoencoder
        int_val = rearrange(value, 'b s h -> (b s) h')

        # Run through the autoencoder
        act = autoencoder.encode(int_val)

        # Subtract value with feature direction*act_of_feature
        if hasattr(autoencoder, 'decoder'):
            feature_direction = torch.outer(act[:, feature], autoencoder.decoder.weight[:, feature])
        elif hasattr(autoencoder, 'encoder') and hasattr(autoencoder, 'encoder_bias'):
            feature_direction = torch.outer(act[:, feature], autoencoder.encoder.T[:, feature])
        else:
            raise Exception("Unrecognized autoencoder type")

        batch, seq_len, hidden_size = value.shape
        feature_direction = rearrange(feature_direction, '(b s) h -> b s h', b=batch, s=seq_len)
        value -= feature_direction
        return value

    return model.run_with_hooks(tokens, 
        fwd_hooks=[(
            cache_name,
            ablation_hook
        )]
    )

def visualize_logit_diff(text, autoencoder, cache_name, features=None, setting="true_tokens", verbose=False):
    features = best_feature

    if features==None:
        features = torch.tensor([best_feature])
    if isinstance(features, int):
        features = torch.tensor([features])
    if isinstance(features, list):
        features = torch.tensor(features)
    if isinstance(text, str):
        text = [text]
    text_list = []
    logit_list = []
    for t in text:
        tokens = model.to_tokens(t, prepend_bos=False)
        with torch.no_grad():
            original_logits = model(tokens).log_softmax(-1).cpu()
            ablated_logits = ablate_feature_direction(tokens, features, model, cache_name, autoencoder).log_softmax(-1).cpu()
        diff_logits = ablated_logits  - original_logits# ablated > original -> negative diff
        tokens = tokens.cpu()
        if setting == "true_tokens":
            split_text = model.to_str_tokens(t, prepend_bos=False)
            gather_tokens = rearrange(tokens[:,1:], "b s -> b s 1") # TODO: verify this is correct
            # Gather the logits for the true tokens
            diff = rearrange(diff_logits[:, :-1].gather(-1,gather_tokens), "b s n -> (b s n)")
        elif setting == "max":
            # Negate the diff_logits to see which tokens have the largest effect on the neuron
            val, ind = (-1*diff_logits).max(-1)
            diff = rearrange(val[:, :-1], "b s -> (b s)")
            diff*= -1 # Negate the values gathered
            split_text = model.to_str_tokens(ind, prepend_bos=False)
            gather_tokens = rearrange(ind[:,1:], "1 s -> 1 s 1")
        split_text = split_text[1:] # Remove the first token since we're not predicting it
        if(verbose):
            text_list += [x.replace('\n', '\\newline') for x in split_text] + ["\n"]
            text_list += [x.replace('\n', '\\newline') for x in split_text] + ["\n"]
            orig = rearrange(original_logits[:, :-1].gather(-1, gather_tokens), "b s n -> (b s n)")
            ablated = rearrange(ablated_logits[:, :-1].gather(-1, gather_tokens), "b s n -> (b s n)")
            logit_list += orig.tolist() + [0.0]
            logit_list += ablated.tolist() + [0.0]
        text_list += [x.replace('\n', '\\newline') for x in split_text] + ["\n"]
        logit_list += diff.tolist() + [0.0]
    logit_list = torch.tensor(logit_list).reshape(-1,1,1)
    if verbose:
        print(f"Max & Min logit-diff: {logit_list.max().item():.2f} & {logit_list.min().item():.2f}")
    return text_neuron_activations(tokens=text_list, activations=logit_list)

if __name__ == '__main__':
    # tests
    assert hyperparams_match({}, {}), "Always match if both desired and actual are empty"
    assert hyperparams_match({}, {"l1_alpha": 0.05}), "Do not match on a key that is only in actual but not desired"
    assert hyperparams_match({"l1_alpha": lambda x: x == 0.05}, {}), "Do not match on a key that is only in desired but not actual"
    assert hyperparams_match({"l1_alpha": lambda x: x == 0.05}, {"l1_alpha": 0.05}), "Same is a match"
    assert not hyperparams_match({"l1_alpha": lambda x: x == 0.05}, {"l1_alpha": 0.10}), "Mismatch is mismatch"
    assert hyperparams_match({"l1_alpha": lambda x: abs(x - 0.05) < 1e-6 }, {"l1_alpha": 0.050000000001}), "Matcher can do fuzzy float matching"
    assert not hyperparams_match({"l1_alpha": lambda x: abs(x - 0.05) < 1e-6 }, {"l1_alpha": 0.051}), "Matcher can do fuzzy float matching"
