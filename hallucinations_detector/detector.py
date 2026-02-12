import io
import os
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.console import Console

# Path hack to include BackwardLens library
BACKWARD_LENS_DIR = Path(__file__).resolve().parents[1] / "BackwardLens"
sys.path.insert(0, str(BACKWARD_LENS_DIR))

import llm_utils

class HallucinationDetector:
    def __init__(self, model_name="gpt2-large", device=None, 
                 max_new_tokens=30, min_new_tokens=1, 
                 temperature=0.8, top_p=0.9, do_sample=False):
        
        self.console = Console()
        
        # Store configuration in a simple object to maintain compatibility
        self.args = type('Args', (), {})() 
        self.args.model_name = model_name
        self.args.device = device
        self.args.max_new_tokens = max_new_tokens
        self.args.min_new_tokens = min_new_tokens
        self.args.temperature = temperature
        self.args.top_p = top_p
        self.args.do_sample = do_sample

        self._model = None
        self._tokenizer = None
        self._model_aux = None
        self._device = None

    def _load_model(self):
        # Lazily load model + tokenizer + Backward Lens helpers once.
        if self._model is not None:
            return self._model, self._tokenizer, self._model_aux, self._device

        args = self.args
        device = torch.device(args.device) if args.device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Load tokenizer and model for the selected LLM.
        self.console.print(f"Loading tokenizer/model for [bold]{args.model_name}[/bold]...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        try:
            os.environ["TOKENIZERS_PARALLELISM"] = "true"
        except Exception:
            pass

        # Load model and wrap with Backward Lens utilities.
        model = AutoModelForCausalLM.from_pretrained(args.model_name).eval().requires_grad_(False).to(device)
        
        # Redirect stdout/stderr to suppress library noise during loading
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            model_aux = llm_utils.model_extra(model=model, device=device)

        self._model = model
        self._tokenizer = tokenizer
        self._model_aux = model_aux
        self._device = device
        return model, tokenizer, model_aux, device

    def calculate_hallucination_score(self, prompt: str):
        # Compute KL disagreement across generated tokens until stop criteria.
        args = self.args
        model, tokenizer, model_aux, device = self._load_model()
        
        # The Lens Layer is typically the middle layer
        lens_layer = model_aux.n_layer // 2

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        generated_ids = input_ids
        prompt_len = input_ids.shape[1]

        kl_scores = []
        max_new_tokens = self.args.max_new_tokens

        # Token generation loop with KL computed at each step.
        for step in range(max_new_tokens):
            with torch.no_grad():
                outputs = model(generated_ids, output_hidden_states=True)

            # Get Hidden States: Final Layer vs Lens Layer (Middle)
            hs_final = outputs.hidden_states[-1][0, -1, :]
            hs_lens = outputs.hidden_states[lens_layer][0, -1, :]

            # Project hidden states to Vocabulary (Logits)
            probs_final = model_aux.hs_to_probs(hs_final)
            probs_lens = model_aux.hs_to_probs(hs_lens)

            # Calculate KL Divergence (Entropy/Disagreement)
            kl = F.kl_div(probs_lens.log(), probs_final, reduction="sum").item()
            kl_scores.append(kl)

            next_logits = outputs.logits[0, -1, :]

            # Sampling Logic (Optional, usually Greedy for this task)
            if args.do_sample:
                if args.temperature > 0:
                    next_logits = next_logits / args.temperature

                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                sorted_probs = torch.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                cutoff = cumulative_probs > args.top_p
                cutoff[..., 0] = False 
                sorted_logits[cutoff] = -float("inf")
                filtered_logits = torch.empty_like(next_logits).scatter_(0, sorted_indices, sorted_logits)

                probs = torch.softmax(filtered_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)

            # Stop Criteria
            partial_text = tokenizer.decode(
                generated_ids[0, prompt_len:], skip_special_tokens=True
            )

            # 1. Stop on punctuation (end of sentence/phrase)
            if any(char in partial_text for char in [".", "\n", ",", ";", "!", "?"]):
                break

            # 2. Stop length limit (approx 10 words) to prevent rambling
            if len(partial_text.strip().split()) >= 10:
                break

        if not kl_scores:
            kl_scores = [0.0]

        # Decode final answer
        answer_text = tokenizer.decode(generated_ids[0, prompt_len:], skip_special_tokens=True)
        if not answer_text:
            answer_text = "<empty>"

        # Summary stats
        kl_array = np.array(kl_scores, dtype=np.float32)
        kl_stats = {
            "mean": float(kl_array.mean()),
            "max": float(kl_array.max()),
            "p90": float(np.percentile(kl_array, 90)),
            "std": float(kl_array.std()),
        }

        return kl_stats, answer_text