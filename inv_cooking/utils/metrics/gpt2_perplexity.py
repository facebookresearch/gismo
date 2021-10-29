from enum import Enum
from typing import List

import torch
import torch.nn.functional as functional
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from inv_cooking.datasets.vocabulary import Vocabulary
from inv_cooking.utils.visualisation.recipe_utils import recipe_to_text, format_recipe


class LanguageModelType(Enum):
    small = 0
    medium = 1
    large = 2

    def to_model(self) -> str:
        if self == LanguageModelType.small:
            return "gpt2"
        elif self == LanguageModelType.medium:
            return "gpt2-medium"
        else:
            return "gpt2-large"


class PretrainedLanguageModel:
    """
    Language model built around GPT-2 allowing to:
    - play with GPT-2 generation capability
    - evaluate the perplexity of a generated recipe
    """

    def __init__(self, model_type: LanguageModelType = LanguageModelType.medium):
        super().__init__()
        self.device = torch.device("cpu")
        self.model_type = model_type
        hugging_face_model = model_type.to_model()
        self.tokenizer = GPT2Tokenizer.from_pretrained(hugging_face_model)
        self.model = GPT2LMHeadModel.from_pretrained(hugging_face_model)
        self.model.eval()

    def to(self, device):
        self.device = device
        self.model.cuda(self.device)

    def auto_complete(self, text: str, steps: int = 1) -> str:
        """
        Auto complete a sentence: the initial text provided
        cannot be empty (need an initial prompt).
        """
        for _ in range(steps):
            indexed_tokens = self.tokenizer.encode(text)
            tokens_tensor = torch.tensor([indexed_tokens], device=self.device)
            with torch.no_grad():
                outputs = self.model(tokens_tensor)
                predictions = outputs[0]
                predicted_index = torch.argmax(predictions[0, -1, :]).item()
            text = self.tokenizer.decode(indexed_tokens + [predicted_index])
        return text

    def measure_perplexity_slow(self, generated_recipe: str) -> torch.Tensor:
        """
        Measure the perplexity of a recipe by asking GPT to
        generate text after an initial prompt "recipe instructions:"
        and measure the perplexity of the ground truth recipe

        :return a tensor with one element containing the perplexity
        """
        gpt_prompt = self.tokenizer.encode("recipe instructions:")
        generated_recipe = self.tokenizer.encode(generated_recipe)

        prob_sentence = []
        for i in range(len(generated_recipe)):
            tokens_tensor = torch.tensor([gpt_prompt + generated_recipe[:i]])
            tokens_tensor = tokens_tensor.to(self.device)
            with torch.no_grad():
                outputs = self.model(tokens_tensor)
                predictions = outputs[0]
                probs = torch.nn.functional.softmax(predictions, dim=-1)
                prob_target = probs[0, -1, generated_recipe[i]]
                prob_sentence.append(prob_target.cpu())
        prob_sentence = torch.stack(prob_sentence)
        perplexity = torch.exp(-torch.mean(torch.log(prob_sentence)))
        return perplexity.cpu()

    def measure_perplexity_parallel(self, generated_recipe: str) -> torch.Tensor:
        """
        Measure the perplexity of a recipe by asking GPT to
        generate text after an initial prompt "recipe instructions:"
        and measure the perplexity of the ground truth recipe

        :return a tensor with one element containing the perplexity
        """
        gpt_prompt = self.tokenizer.encode("recipe instructions: ")
        generated_recipe = self.tokenizer.encode(generated_recipe)

        input_tokens = gpt_prompt + generated_recipe[:-1]
        tokens_tensor = torch.tensor([input_tokens])
        tokens_tensor = tokens_tensor.to(self.device)
        targets = torch.LongTensor(generated_recipe[1:]).to(self.device)

        with torch.no_grad():
            outputs = self.model(tokens_tensor)
            predictions = outputs[0]
            logits = predictions[0, len(gpt_prompt) :, :]
            ce = functional.cross_entropy(logits, targets)
            perplexity = torch.exp(ce)
        return perplexity.cpu()

    def measure_perplexity_batch(self, generated_recipes: List[str]) -> torch.Tensor:
        """
        Measure the perplexity of a BATCH of recipe by asking GPT to
        generate text after an initial prompt "recipe instructions:"
        and measure the perplexity of the ground truth recipe

        :return a tensor with one element for each provided recipe,
                each entry containing the perplexity of the given recipe
        """
        gpt_prompt = self.tokenizer.encode("recipe instructions: ")
        generated_recipes = [self.tokenizer.encode(r) for r in generated_recipes]

        input_tokens = [gpt_prompt + r[:-1] for r in generated_recipes]
        targets = [r[1:] for r in generated_recipes]

        # Complete the input batch with padding at the end
        max_len = max(len(x) for x in input_tokens)
        for x in input_tokens:
            if len(x) < max_len:
                x.extend([0] * (max_len - len(x)))

        # Complete the target batch with padding at the end
        # and use -100 as ignored target for the cross entropy
        max_len = max(len(t) for t in targets)
        for t in targets:
            if len(t) < max_len:
                t.extend([-100] * (max_len - len(t)))

        # Compute the logits of GPT for the whole batch
        tokens_tensor = torch.tensor(input_tokens).cuda()
        target_tensor = torch.LongTensor(targets).cuda()
        with torch.no_grad():
            outputs = self.model(tokens_tensor)
            logits = outputs[0]
            logits = logits[:, len(gpt_prompt) :, :]

            # Flatten the target and logits to compute the cross entropy
            batch_size, seq_len, vocab_len = logits.shape
            logits = logits.reshape((-1, vocab_len))
            target_tensor = target_tensor.reshape((-1,))
            ce = functional.cross_entropy(
                logits, target_tensor, ignore_index=-100, reduction="none"
            )

            # Move back the cross entropy to the batch shape and divide by
            # the non-zero entries to have one mean cross entropy value for
            # each of the recipes
            ce = ce.reshape((batch_size, seq_len))
            mask = ce != 0.0
            ce = torch.sum(ce, dim=-1) / torch.sum(mask, dim=-1)

        # Return the perplexity, one for each recipe
        return torch.exp(ce).cpu()


class LanguageModelPerplexity:
    def __init__(
        self,
        vocab_instructions: Vocabulary,
        model_type: LanguageModelType = LanguageModelType.small,
    ):
        super().__init__()
        self.vocab_instructions = vocab_instructions
        self.pretrained_language_model = PretrainedLanguageModel(model_type=model_type)

    def compute(self, recipe_outs: torch.Tensor) -> torch.Tensor:
        if recipe_outs.device != self.pretrained_language_model.device:
            self.pretrained_language_model.to(recipe_outs.device)

        all_perplexity = []
        for recipe_out in recipe_outs:
            text = recipe_to_text(recipe_out, self.vocab_instructions)
            text = format_recipe(text)
            ppl = self.pretrained_language_model.measure_perplexity_parallel(text)
            all_perplexity.append(ppl)
        return torch.stack(all_perplexity)

    def compute_batch(self, recipe_outs: torch.Tensor) -> torch.Tensor:
        if recipe_outs.device != self.pretrained_language_model.device:
            self.pretrained_language_model.to(recipe_outs.device)

        text_recipes = [
            format_recipe(recipe_to_text(r, self.vocab_instructions)) for r in recipe_outs
        ]
        return self.pretrained_language_model.measure_perplexity_batch(text_recipes)

