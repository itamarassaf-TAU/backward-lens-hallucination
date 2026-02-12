import argparse
import io
import os
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from rich import box
from rich.panel import Panel
from rich.table import Table

from console_utils import setup_console_logger

BACKWARD_LENS_DIR = Path(__file__).resolve().parents[1] / "BackwardLens"
sys.path.insert(0, str(BACKWARD_LENS_DIR))

import llm_utils
import opt_utils


class HallucinationDetector:

    def __init__(self):
        self.console, self.log = setup_console_logger()
        self.args = self.parse_args()
        self._model = None
        self._tokenizer = None
        self._model_aux = None
        self._device = None

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_name", type=str, default="gpt2-large")
        parser.add_argument("--device", type=str, default=None)
        parser.add_argument("--max_new_tokens", type=int, default=30)
        parser.add_argument("--min_new_tokens", type=int, default=1)
        parser.add_argument("--temperature", type=float, default=0.8)
        parser.add_argument("--top_p", type=float, default=0.9)
        parser.add_argument("--do_sample", action="store_true")
        return parser.parse_args()

    def _load_model(self):
        # Lazily load model + tokenizer + Backward Lens helpers once.
        if self._model is not None:
            return self._model, self._tokenizer, self._model_aux, self._device

        args = self.args
        device = torch.device(args.device) if args.device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Load tokenizer and model for the selected LLM.
        self.log.info(f"Loading tokenizer/model for [bold]{args.model_name}[/bold]...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        try:
            os.environ["TOKENIZERS_PARALLELISM"] = "true"
        except Exception:
            pass

        # Load model and wrap with Backward Lens utilities.
        model = AutoModelForCausalLM.from_pretrained(args.model_name).eval().requires_grad_(False).to(device)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            model_aux = llm_utils.model_extra(model=model, device=device)

        self._model = model
        self._tokenizer = tokenizer
        self._model_aux = model_aux
        self._device = device
        return model, tokenizer, model_aux, device

    def run_backward_lens(self, prompt: str, target_new: str = " Paris"):
        # Run Backward Lens once to collect hidden states and VJPs.
        args = self.args
        model, tokenizer, model_aux, device = self._load_model()
        config = model_aux.config
        n_layer = model_aux.n_layer

        config_table = Table(title="Run Configuration", box=box.MINIMAL_DOUBLE_HEAD)
        config_table.add_column("Field", style="cyan", no_wrap=True)
        config_table.add_column("Value", style="white")
        config_table.add_row("Model", args.model_name)
        config_table.add_row("Device", str(device))
        config_table.add_row("Prompt", prompt)
        config_table.add_row("Max new tokens", str(args.max_new_tokens))
        self.console.print(config_table)

        self.console.print(Panel.fit("Extracting Backward Lens (hidden states + VJPs)...", style="bold yellow"))

        # Suppress verbose internal prints from Backward Lens.
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            res = opt_utils.get_nll_opt_model(
                prompt,
                target_new,
                model_name=args.model_name,
                tokenizer=tokenizer,
                opt="SGD",
                device=device,
                lr=1.0,
                params_names_filter=opt_utils.only_mlp_filter,
                wrapp_forward_pass_config="AUTO",
                wrapp_backward_pass_config="AUTO",
            )

        hs_collector = res["hs_collector"]
        grad_collector = res["grad_collector"]

        vjp_count = len(grad_collector[n_layer - 1][config.mlp_ff2]["output"])

        self.console.print(Panel.fit("Backward Lens extraction OK", style="bold green"))

        self.console.print(Panel.fit(f"Collected VJPs for FF2 output positions: {vjp_count}", style="bold blue"))

        return {
            "model_name": args.model_name,
            "device": str(device),
            "prompt": prompt,
            "vjp_count": vjp_count,
            "config": config,
            "n_layer": n_layer,
            "hs_collector": hs_collector,
            "grad_collector": grad_collector,
            "model_aux": model_aux,
        }

    def calculate_hallucination_score(self, prompt: str):
        # Compute KL disagreement across generated tokens until stop criteria.
        args = self.args
        model, tokenizer, model_aux, device = self._load_model()
        lens_layer = model_aux.n_layer // 2

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        generated_ids = input_ids
        prompt_len = input_ids.shape[1]

        kl_scores = []
        max_new_tokens = self.args.max_new_tokens

        # Token generation with KL computed at each step.
        for step in range(max_new_tokens):
            with torch.no_grad():
                outputs = model(generated_ids, output_hidden_states=True)

            hs_final = outputs.hidden_states[-1][0, -1, :]
            hs_lens = outputs.hidden_states[lens_layer][0, -1, :]

            probs_final = model_aux.hs_to_probs(hs_final)
            probs_lens = model_aux.hs_to_probs(hs_lens)

            kl = F.kl_div(probs_lens.log(), probs_final, reduction="sum").item()
            kl_scores.append(kl)

            next_logits = outputs.logits[0, -1, :]

            if args.do_sample:
                # Apply temperature.
                if args.temperature > 0:
                    next_logits = next_logits / args.temperature

                # Top-p (nucleus) filtering.
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                sorted_probs = torch.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                cutoff = cumulative_probs > args.top_p
                cutoff[..., 0] = False  # keep at least one token
                sorted_logits[cutoff] = -float("inf")
                filtered_logits = torch.empty_like(next_logits).scatter_(0, sorted_indices, sorted_logits)

                probs = torch.softmax(filtered_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)

            # Stop on period/newline to keep answers short.
            partial_text = tokenizer.decode(
                generated_ids[0, prompt_len:], skip_special_tokens=True
            )

            # 1. Stop if we see ANY punctuation that ends a phrase
            if any(char in partial_text for char in [".", "\n", ",", ";", "!", "?"]):
                break

            # 2. Stop immediately if we have generated more than 10 words
            # (Allows for "The United States" but kills "The United States is...")
            if len(partial_text.strip().split()) >= 10:
                break

        if not kl_scores:
            kl_scores = [0.0]

        # Decode only the generated answer portion (exclude prompt).
        answer_text = tokenizer.decode(generated_ids[0, prompt_len:], skip_special_tokens=True)
        if not answer_text:
            answer_text = "<empty>"

        # Summary stats for KL over generated tokens.
        kl_array = np.array(kl_scores, dtype=np.float32)
        kl_stats = {
            "mean": float(kl_array.mean()),
            "max": float(kl_array.max()),
            "p90": float(np.percentile(kl_array, 90)),
            "std": float(kl_array.std()),
        }

        return kl_stats, answer_text
