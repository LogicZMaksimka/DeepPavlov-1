import json
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
    config_path = str(configs.squad.natural_questions_fid)
    config = gen_wandb_config(config_path)
    model_path = "/home/savkin/.deeppavlov/models/generative_qa/fid/natural_questions/model.pth.tar"

    wandb.login()
    wandb.init(entity="logiczmaksimka",
            project="Fusion-in-decoder",
            group="FiD_test_00",
            job_type="train",
            config=config)
    wandb.watch_called = False


    # TODO: научиться извлекать модель из библиотеки перед началом обучения
    # chainer = build_model(configs.squad.natural_questions_fid, download=False)
    # preprocessor = chainer[0]
    # model = chainer[1] # FiD class
    # nn_model = model.model.module if hasattr(model.model, "module") else model.model # torch.nn.Model

    # wandb.watch(nn_model, log="all")

    # TODO: научится красиво извлекать логи и передавать из в wandb

    model = train_model(configs.squad.natural_questions_fid, download=False)

    wandb.save(config_path)