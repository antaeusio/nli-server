"""Entailment scoring with a pinned DeBERTa-v3 zero-shot NLI model."""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from nli_server.server import InstructionsTooLong

MODEL_REPOSITORY = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0-c"
MODEL_REVISION = "b2730f16019076bb0009481121efbe4705e0e378"
MODEL_NAME = "deberta-v3-large-zeroshot-v2.0-c@" + MODEL_REVISION[:7]
MAX_TOKENS = 512
# Leave at least half of the context window for the premise.
MAX_HYPOTHESIS_TOKENS = 256
# Pairs per forward pass; bounds attention memory for many rules.
BATCH_SIZE = 8


class NLIScorer:
    def __init__(self, model_dir, threads):
        torch.set_num_threads(threads)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_dir, local_files_only=True, use_safetensors=True
        ).eval()
        labels = {label.lower(): index for index, label in self.model.config.id2label.items()}
        self.entailment = labels["entailment"]

    def score(self, premise, hypotheses):
        """Return the entailment probability of each hypothesis given the premise."""
        for hypothesis in hypotheses:
            if len(self.tokenizer(hypothesis)["input_ids"]) > MAX_HYPOTHESIS_TOKENS:
                raise InstructionsTooLong()
        probabilities = []
        for start in range(0, len(hypotheses), BATCH_SIZE):
            batch = hypotheses[start : start + BATCH_SIZE]
            # Long states are truncated from the premise, never the hypothesis.
            inputs = self.tokenizer(
                [premise] * len(batch),
                batch,
                truncation="only_first",
                max_length=MAX_TOKENS,
                padding=True,
                return_tensors="pt",
            )
            with torch.inference_mode():
                logits = self.model(**inputs).logits
            probabilities.extend(torch.softmax(logits, dim=-1)[:, self.entailment].tolist())
        return probabilities
