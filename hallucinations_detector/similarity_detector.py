from typing import Optional

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class SimilarityDetector:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        nli_model_name: str = "facebook/bart-large-mnli",
    ):
        # Sentence-transformer for cosine similarity.
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=self.device)
        self._nli_model_name = nli_model_name
        self._nli_tokenizer = None
        self._nli_model = None

    def cosine_similarity(self, text_a: str, text_b: str) -> float:
        # Embedding-based cosine similarity (semantic).
        embeddings = self.model.encode(
            [text_a, text_b],
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        score = F.cosine_similarity(embeddings[0], embeddings[1], dim=0).item()
        return score

    def tfidf_cosine_similarity(self, text_a: str, text_b: str) -> float:
        # TF-IDF cosine similarity (lexical overlap).
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError as exc:
            raise SystemExit("Missing dependency: scikit-learn. Install with 'pip install scikit-learn'.") from exc

        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf = vectorizer.fit_transform([text_a, text_b])
        score = (tfidf[0] @ tfidf[1].T).A[0][0]
        return float(score)

    def _load_nli(self):
        # Lazy-load NLI model.
        if self._nli_model is not None:
            return

        self._nli_tokenizer = AutoTokenizer.from_pretrained(self._nli_model_name)
        self._nli_model = AutoModelForSequenceClassification.from_pretrained(self._nli_model_name).to(self.device)
        self._nli_model.eval()

    def _entailment_index(self) -> int:
        # Find entailment label index for the NLI model.
        label2id = self._nli_model.config.label2id
        for key, idx in label2id.items():
            if key.lower() == "entailment":
                return idx
        return int(label2id.get("ENTAILMENT", 2))

    def nli_entailment_score(self, premise: str, hypothesis: str) -> float:
        # NLI entailment probability: premise entails hypothesis.
        self._load_nli()

        inputs = self._nli_tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self._nli_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)

        ent_idx = self._entailment_index()
        return probs[0, ent_idx].item()

    @staticmethod
    def _min_max_normalize(values):
        # Normalize a list to [0, 1].
        values = list(values)
        if not values:
            return []
        v_min = min(values)
        v_max = max(values)
        if v_max == v_min:
            return [0.0 for _ in values]
        return [(v - v_min) / (v_max - v_min) for v in values]

    def combined_similarity(
        self,
        cos_scores,
        tfidf_scores,
        nli_scores,
        weights=(1.0, 1.0, 1.0),
    ):
        # Combine normalized metrics with weights, then re-normalize.
        cos_norm = self._min_max_normalize(cos_scores)
        tfidf_norm = self._min_max_normalize(tfidf_scores)
        nli_norm = self._min_max_normalize(nli_scores)

        w_cos, w_tfidf, w_nli = weights
        combined = [
            w_cos * cos_norm[i] + w_tfidf * tfidf_norm[i] + w_nli * nli_norm[i]
            for i in range(len(cos_norm))
        ]
        return self._min_max_normalize(combined)
