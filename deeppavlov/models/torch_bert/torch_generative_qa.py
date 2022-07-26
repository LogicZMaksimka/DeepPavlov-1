# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import json
import math
from logging import getLogger
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import types
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from overrides import overrides
from transformers import T5ForConditionalGeneration, T5Config, AutoConfig, AutoTokenizer, T5Tokenizer
from transformers.data.processors.utils import InputFeatures

from deeppavlov import build_model
from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.estimator import Component
from deeppavlov.core.models.torch_model import TorchModel


from deeppavlov.FiD.src.model import FiDT5


logger = getLogger(__name__)


def softmax_mask(val, mask):
    inf = 1e30
    return -inf * (1 - mask.to(torch.float32)) + val


@register('torch_generative_qa')
class TorchGenerativeQA(TorchModel):
    def __init__(self,
                 pretrained_transformer: str,
                 attention_probs_keep_prob: Optional[float] = None,
                 hidden_keep_prob: Optional[float] = None,
                 optimizer: str = "AdamW",
                 optimizer_parameters: Optional[dict] = None,
                 bert_config_file: Optional[str] = None,
                 learning_rate_drop_patience: int = 20,
                 learning_rate_drop_div: float = 2.0,
                 load_before_drop: bool = True,
                 clip_norm: Optional[float] = None,
                 min_learning_rate: float = 1e-06,
                 **kwargs) -> None:

        if not optimizer_parameters:
            optimizer_parameters = {"lr": 0.01,
                                    "weight_decay": 0.01,
                                    "betas": (0.9, 0.999),
                                    "eps": 1e-6}

        self.attention_probs_keep_prob = attention_probs_keep_prob
        self.hidden_keep_prob = hidden_keep_prob
        self.clip_norm = clip_norm

        self.pretrained_transformer = pretrained_transformer
        self.bert_config_file = bert_config_file
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_transformer, do_lower_case=False)

        super().__init__(optimizer=optimizer,
                         optimizer_parameters=optimizer_parameters,
                         learning_rate_drop_patience=learning_rate_drop_patience,
                         learning_rate_drop_div=learning_rate_drop_div,
                         load_before_drop=load_before_drop,
                         min_learning_rate=min_learning_rate,
                         **kwargs)

    def train_on_batch(self, input_ids_batch, attention_mask_batch, target_ids_batch) -> Dict:
        input_ids_batch = torch.LongTensor(input_ids_batch).to(self.device)
        attention_mask_batch = torch.LongTensor(attention_mask_batch).to(self.device)
        target_ids_batch = torch.LongTensor(target_ids_batch).to(self.device)
        input_ = {
            'input_ids': input_ids_batch,
            'attention_mask': attention_mask_batch,
            'labels': target_ids_batch
        }

        self.optimizer.zero_grad()
        loss = self.model(**input_).loss
        if self.is_data_parallel:
            loss = loss.mean()
        loss.backward()
        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        if self.clip_norm:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.clip_norm)

        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return {'loss': loss.item()}

    @property
    def is_data_parallel(self) -> bool:
        return isinstance(self.model, torch.nn.DataParallel)

    def __call__(self, input_ids_batch, attention_mask_batch, target_ids_batch=None):
        input_ids_batch = torch.LongTensor(input_ids_batch).to(self.device)
        attention_mask_batch = torch.LongTensor(attention_mask_batch).to(self.device)

        input_ = {
            'input_ids': input_ids_batch,
            'attention_mask': attention_mask_batch,
        }

        ppl = 0.0
        model = self.model.module if hasattr(self.model, "module") else self.model
        with torch.no_grad():
            answer_ids_batch = model.generate(**input_)
            if target_ids_batch is not None:
                target_ids_batch = torch.LongTensor(
                    target_ids_batch).to(self.device)
                loss = model(input_ids=input_ids_batch, attention_mask=attention_mask_batch,
                                  labels=target_ids_batch).loss
                ppl = torch.exp(loss)
                ppl = [ppl.detach().cpu().numpy().tolist()]

        answers_batch = []
        for answer_ids in answer_ids_batch:
            answer = self.tokenizer.decode(
                answer_ids, skip_special_tokens=True)
            answers_batch.append(answer)

        if target_ids_batch is None:
            return answers_batch
        else:
            return answers_batch, ppl


    @overrides
    def load(self, fname=None):
        if fname is not None:
            self.load_path = fname

        if self.pretrained_transformer:
            logger.info(f"From pretrained {self.pretrained_transformer}.")
            config = AutoConfig.from_pretrained(self.pretrained_transformer,
                                                output_attentions=False,
                                                output_hidden_states=False)

            self.model = T5ForConditionalGeneration.from_pretrained(
                self.pretrained_transformer, config=config)

        elif self.bert_config_file and Path(self.bert_config_file).is_file():
            self.bert_config = AutoConfig.from_json_file(
                str(expand_path(self.bert_config_file)))

            if self.attention_probs_keep_prob is not None:
                self.bert_config.attention_probs_dropout_prob = 1.0 - self.attention_probs_keep_prob
            if self.hidden_keep_prob is not None:
                self.bert_config.hidden_dropout_prob = 1.0 - self.hidden_keep_prob
            self.model = T5ForConditionalGeneration(config=self.bert_config)
        else:
            raise ConfigError("No pre-trained BERT model is given.")

        if self.device.type == "cuda" and torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)

        self.model.to(self.device)
        

        self.optimizer = getattr(torch.optim, self.optimizer_name)(
            self.model.parameters(), **self.optimizer_parameters)

        if self.lr_scheduler_name is not None:
            self.lr_scheduler = getattr(torch.optim.lr_scheduler, self.lr_scheduler_name)(
                self.optimizer, **self.lr_scheduler_parameters)

        if self.load_path:
            logger.info(f"Load path {self.load_path} is given.")
            if isinstance(self.load_path, Path) and not self.load_path.parent.is_dir():
                raise ConfigError("Provided load path is incorrect!")

            weights_path = Path(self.load_path.resolve())
            weights_path = weights_path.with_suffix(f".pth.tar")
            if weights_path.exists():
                logger.info(f"Load path {weights_path} exists.")
                logger.info(
                    f"Initializing `{self.__class__.__name__}` from saved.")

                # now load the weights, optimizer from saved
                logger.info(f"Loading weights from {weights_path}.")
                checkpoint = torch.load(weights_path, map_location=self.device)
                model_state = checkpoint["model_state_dict"]
                optimizer_state = checkpoint["optimizer_state_dict"]

                # load a multi-gpu model on a single device
                if not self.is_data_parallel and "module." in list(model_state.keys())[0]:
                    tmp_model_state = {}
                    for key, value in model_state.items():
                        tmp_model_state[re.sub("module.", "", key)] = value
                    model_state = tmp_model_state

                strict_load_flag = bool([key for key in checkpoint["model_state_dict"].keys()
                                         if key.endswith("embeddings.position_ids")])
                self.model.load_state_dict(
                    model_state, strict=strict_load_flag)
                self.optimizer.load_state_dict(optimizer_state)
                self.epochs_done = checkpoint.get("epochs_done", 0)
            else:
                logger.info(
                    f"Init from scratch. Load path {weights_path} does not exist.")


@register('torch_generative_qa_fid')
class TorchFiD(TorchModel):
    def __init__(self,
                 pretrained_transformer: str,
                 attention_probs_keep_prob: Optional[float] = None,
                 hidden_keep_prob: Optional[float] = None,
                 optimizer: str = "AdamW",
                 optimizer_parameters: Optional[dict] = None,
                 bert_config_file: Optional[str] = None,
                 learning_rate_drop_patience: int = 20,
                 learning_rate_drop_div: float = 2.0,
                 load_before_drop: bool = True,
                 clip_norm: Optional[float] = None,
                 min_learning_rate: float = 1e-06,
                 generate_max_length: int = 50,
                 **kwargs) -> None:        

        if not optimizer_parameters:
            optimizer_parameters = {"lr": 0.01,
                                    "weight_decay": 0.01,
                                    "betas": (0.9, 0.999),
                                    "eps": 1e-6}
        self.generate_max_length = generate_max_length

        self.attention_probs_keep_prob = attention_probs_keep_prob
        self.hidden_keep_prob = hidden_keep_prob
        self.clip_norm = clip_norm

        self.pretrained_transformer = pretrained_transformer
        self.bert_config_file = bert_config_file
        self.tokenizer = T5Tokenizer.from_pretrained(self.pretrained_transformer, return_dict=False)

        super().__init__(optimizer=optimizer,
                         optimizer_parameters=optimizer_parameters,
                         learning_rate_drop_patience=learning_rate_drop_patience,
                         learning_rate_drop_div=learning_rate_drop_div,
                         load_before_drop=load_before_drop,
                         min_learning_rate=min_learning_rate,
                         **kwargs)

    def train_on_batch(self, input_ids_batch, attention_mask_batch, target_ids_batch) -> Dict:
        input_ids_batch = torch.LongTensor(input_ids_batch).to(self.device)
        attention_mask_batch = torch.LongTensor(attention_mask_batch).to(self.device)
        target_ids_batch = torch.LongTensor(target_ids_batch).to(self.device)
        input_ = {
            'input_ids': input_ids_batch,
            'attention_mask': attention_mask_batch,
            'labels': target_ids_batch
        }

        self.optimizer.zero_grad()

        loss = self.model(**input_)[0]
        if self.is_data_parallel:
            loss = loss.mean()
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        if self.clip_norm:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.clip_norm)

        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return {'loss': loss.item()}

    @property
    def is_data_parallel(self) -> bool:
        return isinstance(self.model, torch.nn.DataParallel) # TODO: не забудь подправить если поменяешь класс для параллельности

    def __call__(self, input_ids_batch, attention_mask_batch):
        input_ids_batch = torch.LongTensor(input_ids_batch).to(self.device)
        attention_mask_batch = torch.LongTensor(attention_mask_batch).to(self.device)
        input_ = {
            'input_ids': input_ids_batch,
            'attention_mask': attention_mask_batch,
        }

        model = self.model.module if hasattr(self.model, "module") else self.model
        with torch.no_grad():
            answer_ids_batch = model.generate(max_length=self.generate_max_length, **input_)

        answers_batch = self.tokenizer.batch_decode(answer_ids_batch, skip_special_tokens=True)        
        return answers_batch
    
    
    def load_default(self) -> None:
        if self.pretrained_transformer:
            self.tokenizer = T5Tokenizer.from_pretrained(self.pretrained_transformer, return_dict=False)
            t5 = T5ForConditionalGeneration.from_pretrained(self.pretrained_transformer)
            # проинициализировали архитектуру модели (веса ещё не загрузили)          
            self.model = FiDT5(t5.config)
            self.model.load_t5(t5.state_dict())
            self.model.to(self.device)

            # говорим какой у нас оптимайзер и планировщик lr
            self.optimizer = getattr(torch.optim, self.optimizer_name)(
                self.model.parameters(), **self.optimizer_parameters)
            if self.lr_scheduler_name is not None:
                self.lr_scheduler = getattr(torch.optim.lr_scheduler, self.lr_scheduler_name)(
                    self.optimizer, **self.lr_scheduler_parameters)

            if self.opt.get("criterion", None):
                self.criterion = getattr(torch.nn, self.opt.get("criterion", None))()
        else:
            raise AttributeError("Model is not defined.")

    @overrides
    def load(self, fname=None):
        if fname is not None:
            self.load_path = fname
        
        # Initialize model from scratch
        self.load_default()
    
        # Load model weights and
        # Выставляем у ранее объявленных типов оптимайзеров и моделей
        # конкретные веса и параметры хранящиеся в чекпоинте
        if self.load_path:
            logger.info(f"Load path {self.load_path} is given.")
            if isinstance(self.load_path, Path) and not self.load_path.parent.is_dir():
                raise ConfigError("Provided load path is incorrect!")

            weights_path = Path(self.load_path.resolve())
            weights_path = weights_path.with_suffix(f".pth.tar")
            if weights_path.exists():
                logger.info(f"Load path {weights_path} exists.")
                logger.info(f"Initializing `{self.__class__.__name__}` from saved.")
                logger.info(f"Loading weights from {weights_path}.")
                
                # загружаем чекпоинт со всей полезной информацией
                checkpoint = torch.load(weights_path, map_location=self.device)

                model_state = checkpoint["model_state_dict"]
                optimizer_state = checkpoint["optimizer_state_dict"]
                # load a multi-gpu model on a single device
                if not self.is_data_parallel and "module." in list(model_state.keys())[0]:
                    tmp_model_state = {}
                    for key, value in model_state.items():
                        tmp_model_state[re.sub("module.", "", key)] = value
                    model_state = tmp_model_state

                strict_load_flag = bool([key for key in checkpoint["model_state_dict"].keys()
                                         if key.endswith("embeddings.position_ids")])

                #TODO: чекни что за приколы с флагом strict

                # выгружаем из чекпоинта веса модели, параметры оптимайзера и количество эпох
                self.model.load_state_dict(checkpoint["model_state_dict"], strict=strict_load_flag)
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                self.epochs_done = checkpoint.get("epochs_done", 0)

            else:
                logger.info(f"Init from scratch. Load path {weights_path} does not exist.")
        
        # Enter parallel mode if possible
        if self.device.type == "cuda" and torch.cuda.device_count() > 1:

            self.model = torch.nn.DataParallel(self.model)
            self.model.to(self.device)

            # TODO: в препроцессинге соответсвенно нужно будет разбивать данные на несколько gpu