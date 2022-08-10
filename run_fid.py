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
    deeppavlov_config = configs.squad.nq_t5
    config_path = str(deeppavlov_config)
    config = gen_wandb_config(config_path)

    wandb.login()
    wandb.init(entity="logiczmaksimka",
            project="T5",
            group="t5_test",
            job_type="train",
            config=config)
    
    wandb.watch_called = False
    wandb.save(config_path)
    wandb.save(str(Path(__file__).absolute()))

    print(train_model(deeppavlov_config, download=False))
