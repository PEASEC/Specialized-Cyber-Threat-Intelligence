import argparse
import os
import torch
import numpy as np
from transformers import *

from ADAPET.src.data.Batcher import Batcher
from ADAPET.src.utils.Config import Config
from ADAPET.src.utils.util import device
from ADAPET.src.adapet import adapet
from ADAPET.src.eval.eval_model import test_eval

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', "--exp_dir", required=True)
    args = parser.parse_args()

    test(args.exp_dir)

def test(exp_dir):
    config_file = os.path.join(exp_dir, "config.json")
    config = Config(config_file, mkdir=False)

    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_weight)
    batcher = Batcher(config, tokenizer, config.dataset)
    dataset_reader = batcher.get_dataset_reader()

    model = adapet(config, tokenizer, dataset_reader).to(device)
    model.load_state_dict(torch.load(os.path.join(exp_dir, "best_model.pt")))
    test_eval(config, model, batcher)
