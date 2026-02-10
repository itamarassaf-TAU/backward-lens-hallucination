from dataclasses import dataclass
from typing import Iterable, List, Tuple


@dataclass
class ThresholdResult:
    threshold: float
    f1: float
    precision: float
    recall: float
    accuracy: float


class ThresholdOptimizer:
    def __init__(self, num_thresholds: int = 101):
        self.num_thresholds = num_thresholds

    @staticmethod
    def _min_max_normalize(values: Iterable[float]) -> List[float]:
        values = list(values)
        if not values:
            return []
        v_min = min(values)
        v_max = max(values)
        if v_max == v_min:
            return [0.0 for _ in values]
        return [(v - v_min) / (v_max - v_min) for v in values]

    def similarity_factor(
        self,
        cos_scores: Iterable[float],
        tfidf_scores: Iterable[float],
        nli_scores: Iterable[float],
        weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> List[float]:
        cos_norm = self._min_max_normalize(cos_scores)
        tfidf_norm = self._min_max_normalize(tfidf_scores)
        nli_norm = self._min_max_normalize(nli_scores)

        w_cos, w_tfidf, w_nli = weights
        combined = [
            w_cos * cos_norm[i] + w_tfidf * tfidf_norm[i] + w_nli * nli_norm[i]
            for i in range(len(cos_norm))
        ]

        return self._min_max_normalize(combined)

    @staticmethod
    def _binary_metrics(preds: List[int], labels: List[int]) -> Tuple[float, float, float, float]:
        tp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 1)
        fp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 0)
        fn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 1)
        tn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 0)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
        return f1, precision, recall, accuracy

    def grid_search_threshold(self, sim_scores: Iterable[float], labels: Iterable[int]) -> ThresholdResult:
        sim_scores = list(sim_scores)
        labels = list(labels)
        if not sim_scores or len(sim_scores) != len(labels):
            raise ValueError("sim_scores and labels must be same non-zero length.")

        best = ThresholdResult(threshold=0.0, f1=-1.0, precision=0.0, recall=0.0, accuracy=0.0)
        for i in range(self.num_thresholds):
            threshold = i / (self.num_thresholds - 1)
            preds = [1 if s >= threshold else 0 for s in sim_scores]
            f1, precision, recall, accuracy = self._binary_metrics(preds, labels)
            if f1 > best.f1:
                best = ThresholdResult(
                    threshold=threshold,
                    f1=f1,
                    precision=precision,
                    recall=recall,
                    accuracy=accuracy,
                )

        return best

    def grid_search_kl_threshold(self, kl_scores: Iterable[float], labels: Iterable[int]) -> ThresholdResult:
        kl_scores = list(kl_scores)
        labels = list(labels)
        if not kl_scores or len(kl_scores) != len(labels):
            raise ValueError("kl_scores and labels must be same non-zero length.")

        v_min = min(kl_scores)
        v_max = max(kl_scores)
        if v_max == v_min:
            threshold = v_min
            preds = [1 if s >= threshold else 0 for s in kl_scores]
            f1, precision, recall, accuracy = self._binary_metrics(preds, labels)
            return ThresholdResult(threshold=threshold, f1=f1, precision=precision, recall=recall, accuracy=accuracy)

        best = ThresholdResult(threshold=v_min, f1=-1.0, precision=0.0, recall=0.0, accuracy=0.0)
        for i in range(self.num_thresholds):
            threshold = v_min + (v_max - v_min) * (i / (self.num_thresholds - 1))
            preds = [1 if s >= threshold else 0 for s in kl_scores]
            f1, precision, recall, accuracy = self._binary_metrics(preds, labels)
            if f1 > best.f1:
                best = ThresholdResult(
                    threshold=threshold,
                    f1=f1,
                    precision=precision,
                    recall=recall,
                    accuracy=accuracy,
                )

        return best
