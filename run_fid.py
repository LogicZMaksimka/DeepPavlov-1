import os
import json
from pathlib import Path
import wandb
from deeppavlov import configs, build_model, evaluate_model, train_model

def gen_wandb_config(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)
        config = {**config["chainer"]["pipe"][0],
                  **config["chainer"]["pipe"][1],
                  **config["train"],
                  **config["metadata"]["variables"]}
    return config

if __name__ == "__main__":
    deeppavlov_config = configs.odqa.en_odqa_bpr_fid
    config_path = str(deeppavlov_config)
    config = gen_wandb_config(config_path)

    wandb.login()
    # wandb.init(entity="logiczmaksimka",
    #            project="Fusion-in-decoder",
    #            group="FiD_01",
    #            name="train on contexts sorted by relevance",
    #            job_type="train",
    #            config=config)
    wandb.init(entity="logiczmaksimka",
           project="Fusion-in-decoder",
           group="FiD_test",
           name="BPR+FiD no shuffling trivia eval",
           job_type="eval",
           config=config)
    
    wandb.watch_called = False
    wandb.save(config_path)
    wandb.save(str(Path(__file__).absolute()))

    evaluate_model(deeppavlov_config, download=False)