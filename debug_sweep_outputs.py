###############################################################################
# This belongs in an ipynb, and specifically probably in feature_interp.ipynb #
# Once this is verified correct, move it over and delete this file.           #
###############################################################################
import torch
from transformer_lens import HookedTransformer
import numpy as np
import matplotlib.pyplot as plt

# Fix up path issues so that we can load the model using torch.load
import sys
import os
sys.path.insert(0, os.path.realpath('..'))

# Load the autoencoders
model_name = "EleutherAI/pythia-70m-deduped"
filename = r'/workspace/sparse_coding/outputs/_0/learned_dicts.pt'
layer = 2
setting = "residual"
if setting == "residual":
    cache_name = f"blocks.{layer}.hook_resid_post"
elif setting == "mlp":
    cache_name = f"blocks.{layer}.mlp.hook_post"
else:
    raise NotImplementedError

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
learned_dicts = torch.load(filename)
learned_dict, hyperparam_values = learned_dicts[-1]

# Confirm that, within the largest autoencoder, the learned features are almost orthogonal
learned_decoder_normed = learned_dict.get_learned_dict()
self_cosine_sim = (learned_decoder_normed @ learned_decoder_normed.T)

# Relying on get_learned_dict() returning directions with magnitude 1
# which means cosine sim is just dot product
# (yes math people I know this is obvious to you)
assert 0.999 < self_cosine_sim.diag().min() < 1.001
assert 0.999 < self_cosine_sim.diag().max() < 1.001

# Grab all values of the upper triangle of dict@dict.T, offset by 1
# That's the cosine similarity of every feature with every other feature
# except itself
dict_size = learned_decoder_normed.shape[0]
pairwise_self_similarities = self_cosine_sim[np.triu_indices(dict_size, k=1)]

# And plot
plt.xlabel(f"Pairwise cosine similarity of directions in dictionary\nwith other (non-self) directions in same dictionary\n(size={dict_size})")
plt.ylabel("Count of pairs")
plt.hist(pairwise_self_similarities, bins=100)
plt.show()

# And now to check whether two learned dictionaries of the same size
# contain any features in common. We're hoping to see a nice bimodal
# histogram here.
smaller_dict, smaller_dict_hyperparams = learned_dicts[2]
larger_dict,  larger_dict_hyperparams  = learned_dicts[4]

smaller_decoder_normed = smaller_dict.get_learned_dict()
larger_decoder_normed  = larger_dict.get_learned_dict()

smaller_dict_size = smaller_decoder_normed.shape[0]
larger_dict_size  = larger_decoder_normed.shape[0]

smaller_mcs = (smaller_decoder_normed @ larger_decoder_normed.T).amax(axis=1)
larger_mcs  = (larger_decoder_normed @ smaller_decoder_normed.T).amax(axis=1)

plt.xlabel(f"MCS with any feature in size-{larger_dict_size} dict")
plt.ylabel(f"Count of features in size-{smaller_dict_size} dict")
plt.hist(smaller_mcs, bins=50)
plt.show()

plt.xlabel(f"MCS with any feature in size-{smaller_dict_size} dict")
plt.ylabel(f"Count of features in size-{larger_dict_size} dict")
plt.hist(larger_mcs, bins=50)
plt.show()

# Make sure that the dataset is actually reasonable
from transformer_lens import HookedTransformer
from activation_dataset import make_sentence_dataset, chunk_and_tokenize
from torch import tensor
import json

model_name = "EleutherAI/pythia-70m-deduped"
dataset_name = 'NeelNanda/pile-10k'
max_lines = 100000
model = HookedTransformer.from_pretrained(model_name)
tokenizer = model.tokenizer
sentence_dataset = make_sentence_dataset(dataset_name, max_lines=max_lines, start_line=0)
tokenized_sentence_dataset, bits_per_byte = chunk_and_tokenize(sentence_dataset, tokenizer, max_length=256)
input_ids = tokenized_sentence_dataset['input_ids']
_, real_cache = model.run_with_cache(input_ids[0])
_, cache_len1 = model.run_with_cache(input_ids[0][:1])
_, cache_len2 = model.run_with_cache(input_ids[0][:2])

# First row is activation on first token alone (i.e. <|endoftext|>)
assert real_cache['blocks.2.hook_resid_post'].shape == (1, 256, 512)
assert cache_len1['blocks.2.hook_resid_post'].shape == (1,   1, 512)
assert cache_len2['blocks.2.hook_resid_post'].shape == (1,   2, 512)
assert torch.allclose(
    real_cache['blocks.2.hook_resid_post'][0,0],
    cache_len1['blocks.2.hook_resid_post'][0,0],
    atol=1e-3
)
# Then on first two (i.e. <|endoftext|>It)
assert torch.allclose(
    real_cache['blocks.2.hook_resid_post'][0,1],
    cache_len2['blocks.2.hook_resid_post'][0,1],
    atol=1e-3
)
# So this means that the dataset looks something like the layer-2 residuals
# for the following inputs
# <|endoftext|>
# <|endoftext|>It
# <|endoftext|>It is
# <|endoftext|>It is done
# <|endoftext|>It is done,
# <|endoftext|>It is done, and

# Check for duplicate inputs:
# There are not many duplicates of length >= 20 tokens, so no need to worry
# about that.
c = Counter(tokenizer.decode(ipt[:i]) for ipt in input_ids for i in range(1,20))
for st, ct in c.most_common(10):
    print(f"{ct:>4d} {json.dumps(st)}")
# 2383 "\n"
# 2078 "."
# 1854 ","
# 1721 " the"
#  963 " of"
#  853 " and"
#  766 "\n\n"
#  763 " to"
#  746 "-"
#  635 " a"
