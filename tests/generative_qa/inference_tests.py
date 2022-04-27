from deeppavlov import build_model, configs
from transformers import T5Tokenizer
import pytest
import json

def get_dataset(path: str):
    with open(path, 'r') as file:
        dataset = json.load(file)
        return dataset

model_name = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
gen_model = build_model(configs.squad.coqa_generative_qa_infer, download=True)
retrive_gen_model = build_model(configs.squad.coqa_with_bpr_generative_qa_infer, download=True)
coqa_path = "/home/admin/.deeppavlov/downloads/coqa/coqa_max_tok_50.json"
coqa_dataset = get_dataset(coqa_path)

def tokens_count(str: str):
    return len(tokenizer(str)["input_ids"])


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()

    test_fn = item.obj
    docstring = getattr(test_fn, '__doc__')
    if docstring:
        report.nodeid = docstring


# Inference tests
def test_generative_model_output_length():
    """Generative model output is no longer than 20 tokens"""
    for [[question, [contexts]], answer] in coqa_dataset["valid"][:1000]:
        model_answer = gen_model([question], [[contexts]])
        assert tokens_count(model_answer) <= 20


def test_retrieve_generative_model_output_length():
    """Retrieve + generative model output is no longer than 20 tokens"""
    for [[question, [contexts]], answer] in coqa_dataset["valid"][:1000]:
        model_answer = retrive_gen_model([question])
        assert tokens_count(model_answer) <= 20
