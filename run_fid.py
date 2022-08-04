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
    deeppavlov_config = configs.squad.natural_questions_fid
    config_path = str(deeppavlov_config)
    config = gen_wandb_config(config_path)

    wandb.login()
    wandb.init(entity="logiczmaksimka",
            project="Fusion-in-decoder",
            group="FiD_01",
            name="preprocessing_fix",
            job_type="train",
            config=config)
    
    wandb.watch_called = False
    wandb.save(config_path)
    wandb.save(Path(__file__).absolute())

    model = train_model(deeppavlov_config, download=False)
    # evaluate_model(deeppavlov_config, download=False)
