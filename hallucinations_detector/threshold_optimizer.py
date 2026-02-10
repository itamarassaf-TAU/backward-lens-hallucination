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
        # Normalize a list to [0, 1].
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
    def _confusion_stats(preds: List[int], labels: List[int]) -> Tuple[int, int, int, int]:
        # Compute TP/FP/FN/TN counts.
        tp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 1)
        fp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 0)
        fn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 1)
        tn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 0)
        return tp, fp, fn, tn

    @staticmethod
    def _binary_metrics(preds: List[int], labels: List[int]) -> Tuple[float, float, float, float, float, float]:
        # Compute standard classification metrics.
        tp, fp, fn, tn = ThresholdOptimizer._confusion_stats(preds, labels)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
        tpr = recall
        tnr = tn / (tn + fp) if (tn + fp) else 0.0
        balanced_accuracy = (tpr + tnr) / 2
        denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
        mcc = ((tp * tn) - (fp * fn)) / denom if denom else 0.0
        return f1, precision, recall, accuracy, balanced_accuracy, mcc

    @staticmethod
    def _rate_loss(preds: List[int], labels: List[int]) -> float:
        # Loss over rates: penalize FPR/FNR and reward TPR/TNR.
        tp, fp, fn, tn = ThresholdOptimizer._confusion_stats(preds, labels)
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        tnr = tn / (tn + fp) if (tn + fp) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        fnr = fn / (fn + tp) if (fn + tp) else 0.0
        return (1 - tpr) ** 2 + fpr ** 2 + (1 - tnr) ** 2 + fnr ** 2

    def grid_search_threshold(self, sim_scores: Iterable[float], labels: Iterable[int]) -> ThresholdResult:
        # Grid-search threshold on a generic score to maximize F1.
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

    def grid_search_kl_threshold(
        self,
        kl_scores: Iterable[float],
        labels: Iterable[int],
        metric: str = "balanced_accuracy",
    ) -> ThresholdResult:
        # Grid-search KL threshold and optimize the chosen metric.
        kl_scores = list(kl_scores)
        labels = list(labels)
        if not kl_scores or len(kl_scores) != len(labels):
            raise ValueError("kl_scores and labels must be same non-zero length.")

        v_min = min(kl_scores)
        v_max = max(kl_scores)
        if v_max == v_min:
            threshold = v_min
            preds = [1 if s >= threshold else 0 for s in kl_scores]
            f1, precision, recall, accuracy, balanced_accuracy, mcc = self._binary_metrics(preds, labels)
            return ThresholdResult(threshold=threshold, f1=f1, precision=precision, recall=recall, accuracy=accuracy)

        best = ThresholdResult(threshold=v_min, f1=-1.0, precision=0.0, recall=0.0, accuracy=0.0)
        best_loss = float("inf")
        for i in range(self.num_thresholds):
            threshold = v_min + (v_max - v_min) * (i / (self.num_thresholds - 1))
            preds = [1 if s >= threshold else 0 for s in kl_scores]
            f1, precision, recall, accuracy, balanced_accuracy, mcc = self._binary_metrics(preds, labels)
            loss = self._rate_loss(preds, labels)
            metric_value = {
                "f1": f1,
                "accuracy": accuracy,
                "balanced_accuracy": balanced_accuracy,
                "mcc": mcc,
                "loss": -loss,
            }.get(metric, f1)
            if metric == "loss":
                if loss < best_loss:
                    best_loss = loss
                    best = ThresholdResult(
                        threshold=threshold,
                        f1=f1,
                        precision=precision,
                        recall=recall,
                        accuracy=accuracy,
                    )
            elif metric_value > best.f1:
                best = ThresholdResult(
                    threshold=threshold,
                    f1=f1,
                    precision=precision,
                    recall=recall,
                    accuracy=accuracy,
                )

        return best
