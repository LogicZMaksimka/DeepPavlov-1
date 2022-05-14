from deeppavlov import build_model, configs
from transformers import T5Tokenizer
import json
import pytest
import sys

def get_dataset(path: str):
    with open(path, 'r') as file:
        dataset = json.load(file)
        return dataset

def tokens_count(str: str):
    return len(pytest.tokenizer(str)["input_ids"])

# Inference tests
class TestModelsOutput:
    pytest.model_name = "t5-base"
    pytest.tokenizer = T5Tokenizer.from_pretrained(pytest.model_name)
    pytest.gen_model = build_model(configs.squad.coqa_generative_qa_infer, download=True)
    pytest.retrive_gen_model = build_model(configs.squad.coqa_with_bpr_generative_qa_infer, download=True)

    def test_generative_model_output_length(self, coqa_path):
        """Generative model output is no longer than 20 tokens"""
        coqa_dataset = get_dataset(coqa_path)

        for [[question, [contexts]], answer] in coqa_dataset["valid"][:1000]:
            model_answer = pytest.gen_model([question], [[contexts]])
            assert tokens_count(model_answer) <= 20


    def test_retrieve_generative_model_output_length(self, coqa_path):
        """Retrieve + generative model output is no longer than 20 tokens"""
        coqa_dataset = get_dataset(coqa_path)

        for [[question, [contexts]], answer] in coqa_dataset["valid"][:1000]:
            model_answer = pytest.retrive_gen_model([question])
            assert tokens_count(model_answer) <= 20

    def test_generative_model_answers(self):
        question = "What color was Cotton?"
        context = "Once upon a time, in a barn near a farm house, there lived a little white kitten named Cotton."
        assert pytest.gen_model([question], [[context]]) == ['white']

        question = "Where did she live?"
        context = "Once upon a time, in a barn near a farm house, there lived a little white kitten named Cotton."
        assert pytest.gen_model([question], [[context]]) == ['in a barn']

        question = "Did she live alone?"
        context = "But Cotton wasn't alone in her little home above the barn, oh no. She shared her hay bed with her mommy and 5 other sisters. All of her sisters were cute and fluffy, like Cotton."
        assert pytest.gen_model([question], [[context]]) == ['no']


        question = "Who did she live with?"
        context = "But Cotton wasn't alone in her little home above the barn, oh no. She shared her hay bed with her mommy and 5 other sisters. All of her sisters were cute and fluffy, like Cotton."
        assert pytest.gen_model([question], [[context]]) == ['her mommy and 5 other sisters']

    def test_retrieve_generative_model_answers(self):
        question = "What is the capital of Russia?"
        assert pytest.retrive_gen_model([question]) == ['Moscow']

        question = "How many countries are there?"
        assert pytest.retrive_gen_model([question]) == ['120']

        question = "What is natural language processing?"
        assert pytest.retrive_gen_model([question]) == ['a subfield of computer science, information engineering, and artificial intelligence']