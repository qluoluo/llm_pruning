import argparse
import json
import os

import torch
from einops import rearrange
from tqdm import tqdm
from transformers import AutoConfig, LlamaConfig, LlamaTokenizerFast


def load_safetensors(filename):
    from safetensors import safe_open

    model = safe_open(filename, framework="pytorch")
    state_dict = {}
    for k in model.keys():
        state_dict[k] = model.get_tensor(k)

    return state_dict

def repeat_kv(kv, num_groups, head_dim):
    d_m = kv.shape[1]
    assert d_m % num_groups == 0
    group_size = d_m // num_groups // head_dim

    kv = kv.view(num_groups, head_dim, d_m)
    kv = kv.repeat(1, group_size, 1) # expand head_dim to group_size x head_dim
    kv = kv.view(d_m, d_m)
    return kv
    

def save_conifg(config, tgt):
    config.num_key_value_heads = config.num_attention_heads
    config.save_pretrained(tgt)


def convert(src, tgt):
    """Repeat KV heads to convert gqa model to normal multi-head attention model"""

    config = AutoConfig.from_pretrained(src, trust_remote_code=True)

    head_dim = config.hidden_size // config.num_attention_heads
    num_key_value_groups = config.num_attention_heads // config.num_key_value_heads

    # load index json file
    index_file = os.path.join(src, "pytorch_model.bin.index.json")
    if os.path.exists(index_file):
        with open(index_file) as fp:
            index_dict = json.load(fp)
            index_dict["weight_map"] = {}
    else:
        index_dict = None

    os.makedirs(tgt, exist_ok=True)
    if True:
        for filename in tqdm(os.listdir(src)):
            if filename.endswith(".bin"):
                states = torch.load(os.path.join(src, filename))
            elif filename.endswith(".safetensors"):
                states = load_safetensors(os.path.join(src, filename))
            else:
                continue
            mha_states = {}
            
            for k, v in states.copy().items():    
                if 'k_proj' in k or 'v_proj' in k:
                    v = repeat_kv(v, config.num_key_value_heads, head_dim)
                mha_states[k] = v
            if index_dict is not None:
                for k in mha_states:
                    index_dict["weight_map"][k] = filename
            print(f"Saving to {os.path.join(tgt, filename)}...", flush=True)
            if filename.endswith(".bin"):
                torch.save(mha_states, os.path.join(tgt, filename))
            elif filename.endswith(".safetensors"):
                from safetensors.torch import save_file
                save_file(mha_states, os.path.join(tgt, filename), metadata={"format": "pt"})
            
            del states
    print("Saving config and tokenizer...")
    # index.json
    if index_dict is not None:
        with open(os.path.join(tgt, "pytorch_model.bin.index.json"), "w") as fp:
            json.dump(index_dict, fp, indent=2)
    # tokenizer
    # tokenizer = LlamaTokenizerFast.from_pretrained(src)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(src, trust_remote_code=True)

    tokenizer.init_kwargs.pop("auto_map", None)
    tokenizer.save_pretrained(tgt)
    # config
    save_conifg(config, tgt)
    print("Done!")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, help="Input folder")
    parser.add_argument("--tgt", type=str, help="Output folder")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    convert(args.src, args.tgt)