from typing import List
from datasets import load_metric
from transformers import AutoTokenizer
from sacrebleu.metrics import BLEU
from deeppavlov.core.common.metrics_registry import register_metric


class SacreBleuT5Base:
    def __init__(self):
        self.metric = load_metric('sacrebleu')
        self.tok = AutoTokenizer.from_pretrained("t5-base")
    
    def add_batch(self, predictions, references):
        predictions_ids = self.tok(predictions)['input_ids']
        references_ids = self.tok(references)['input_ids']
        predictions_decoded = self.tok.batch_decode(predictions_ids, skip_special_tokens=True)
        references_decoded = self.tok.batch_decode(references_ids, skip_special_tokens=True)
        self.metric.add_batch(predictions=predictions_decoded, references=map(lambda x: [x], references_decoded))
    
    def compute(self):
        bleu = self.metric.compute()
        return bleu['score']
    
    def clear(self):
        if self.metric.cache_file_name:
            self.compute()



@register_metric('sacred_bleu')
def sacred_bleu(y_true: List[List[str]], y_predicted: List[str]) -> float:
    sacred_bleu = SacreBleuT5Base()
    sacred_bleu.clear()
    sacred_bleu.add_batch(y_predicted, y_true)
    return sacred_bleu.compute()

    # bleu = BLEU()
    # return bleu.corpus_score(y_predicted, [y_true]).score