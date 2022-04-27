import json
import random
import pytest
from string import ascii_lowercase
from deeppavlov import evaluate_model, configs
from deeppavlov.metrics.sacrebleu import sacrebleu
from transformers import T5Tokenizer


model_name = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)

def tokens_count(str: str):
    return len(tokenizer(str)["input_ids"])


def make_typos(text: str, count: int):
    possible_typos_pos = [i for i, sym in enumerate(text) if sym != ' ']
    rand_positions = random.sample(possible_typos_pos, count)
    return ''.join(sym if i not in rand_positions else random.choice(ascii_lowercase) for i, sym in enumerate(text))


def create_typo_dataset(datset: dict(), out_data_path: str, typos_count: int = 1):
    coqa_with_typos = datset.copy()
    for i, [[question, [contexts]], answer] in enumerate(datset["valid"]):
        coqa_with_typos["valid"][i] = [
            [make_typos(question, typos_count), [contexts]], answer]

    with open(out_data_path, "w") as file:
        json.dump(coqa_with_typos, file)


def eval_model(config):
    model = evaluate_model(config, download=False)
    sacrebleu_score = model["valid"]["sacrebleu"]
    perplexity_score = model["valid"]["ppl"]
    return sacrebleu_score, perplexity_score


def get_dataset(path: str):
    with open(path, 'r') as file:
        dataset = json.load(file)
        return dataset

# Dataset tests
class TestDataset:
    coqa_path = "/home/admin/.deeppavlov/downloads/coqa/coqa_max_tok_50.json"

    def test_dataset_tokens_count(self):
        """Dataset contains questions and contexts with valid number of tokens"""
        coqa_dataset = get_dataset(self.coqa_path)

        for type in ["train", "valid"]:
            for [[question, [contexts]], answer] in coqa_dataset[type]:
                assert tokens_count(contexts) <= 50
                assert tokens_count(contexts) + tokens_count(question) <= 512
                assert tokens_count(
                    f"question: {question} sentence: {contexts}") <= 512


# Metrics tests
class TestMetrics:
    precision = 0.1

    def test_sacrebleu_metrics_single(self):
        """Sacrebleu computed correctly on single pair 'question' - 'answer'"""

        y_true = ["I am thirty six years old"]
        y_pred = ["I have thirty six years"]
        assert 35.0 - self.precision < sacrebleu(y_true, y_pred) < 35.0 + self.precision

        y_true = ['The dog had bit the man.']
        y_pred = ['The dog bit the man.']
        assert 51.15 - self.precision < sacrebleu(y_true, y_pred) < 51.15 + self.precision

        y_true = ['No one was surprised.']
        y_pred = ["It wasn't surprising."]
        assert 12.44 - self.precision < sacrebleu(y_true, y_pred) < 12.44 + self.precision

        y_true = ['The man had bitten the dog.']
        y_pred = ['The man had just bitten him.']
        assert 27.77 - self.precision < sacrebleu(y_true, y_pred) < 27.77 + self.precision


    def test_sacrebleu_metrics_batch(self):
        """Sacrebleu computed correctly on batch of pairs 'question' - 'answer'"""

        y_true = ['The dog had bit the man.',
                'No one was surprised.', 'The man had bitten the dog.']
        y_pred = ['The dog bit the man.', "It wasn't surprising.",
                'The man had just bitten him.']
        assert 28.33 - self.precision < sacrebleu(y_true, y_pred) < 28.33 + self.precision


# Model scores tests
class TestModelScores:
    def test_standart_scores(self):
        """Metrics of the basic model are within acceptable area"""

        sacrebleu_score, perplexity_score = eval_model(configs.squad.coqa_generative_qa)
        assert sacrebleu_score > 40.0
        assert perplexity_score < 2.0


    def test_question_1_typos(self):
        """1 typo in question didn't affect model's sacrebleu too much """

        sacrebleu_score, _ = eval_model(configs.squad.coqa_with_question_typos_1_generative_qa)
        assert sacrebleu_score > 38.0


    def test_question_2_typos(self):
        """2 typos in question didn't affect model's sacrebleu too much """

        sacrebleu_score, _ = eval_model(configs.squad.coqa_with_question_typos_2_generative_qa)
        assert sacrebleu_score > 34.0