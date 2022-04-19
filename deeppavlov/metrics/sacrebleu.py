from typing import List
from sacrebleu.metrics import BLEU
from deeppavlov.core.common.metrics_registry import register_metric


@register_metric('sacrebleu')
def sacrebleu(y_true: List[str], y_predicted: List[str]) -> float:
    bleu = BLEU()
    return bleu.corpus_score(y_predicted, [y_true]).score