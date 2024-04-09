"""Logits Processor for generation."""
import inspect
from threading import Thread

import numpy as np

from .utils import log_softmax, softmax, topk

__all__ = [
    "LogitsProcessor",
    "LogitsWarper",
    "LogitsProcessorList",
    "RepetitionPenaltyLogitsProcessor",
    "LogitNormalization",
    "TemperatureLogitsWarper",
    "TopKLogitsWarper",
    "TopPLogitsWarper",
]


class LogitsProcessor:
    """Abstract base class for all logit processors that can be applied during generation."""

    def __call__(self, input_ids, scores):
        """Torch method for processing logits."""
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )


class LogitsWarper:
    """Abstract base class for all logit warpers that can be applied during generation
    with multinomial sampling."""

    def __call__(self, input_ids, scores):
        """Torch method for warping logits."""
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )


class LogitsProcessorList(list):
    """
    This class can be used to create a list of [`LogitsProcessor`] or [`LogitsWarper`] to subsequently
    process a `scores` input tensor. This class inherits from list and adds a specific *__call__* method
    to apply each [`LogitsProcessor`] or [`LogitsWarper`] to the inputs.
    """

    def __call__(self, input_ids, scores, is_finished=None, **kwargs):
        all_threads = []
        for i in range(0, input_ids.shape[0]):
            if is_finished and is_finished[i]:
                continue
            thread = Thread(target=self.process, args=(i, input_ids, scores), kwargs=kwargs)
            all_threads.append(thread)
            thread.start()
        for thread in all_threads:
            thread.join()
        return scores

    def process(self, i, input_ids, scores, **kwargs):
        """apply process"""
        input_ids = input_ids[i : i + 1]
        scores_i = scores[i : i + 1]
        for processor in self:
            function_args = inspect.signature(processor.__call__).parameters
            if len(function_args) > 2:
                if not all(arg in kwargs for arg in list(function_args.keys())[2:]):
                    raise ValueError(
                        f"Make sure that all the required parameters: {list(function_args.keys())} for "
                        f"{processor.__class__} are passed to the logits processor."
                    )
                scores_i = processor(input_ids, scores_i, **kwargs)
            else:
                scores_i = processor(input_ids, scores_i)
        scores[i] = scores_i


class TemperatureLogitsWarper(LogitsWarper):
    r"""
    [`LogitsWarper`] for temperature (exponential scaling output probability distribution).

    Args:
        temperature (`float`):
            The value used to module the logits distribution.
    """

    def __init__(self, temperature: float):
        temperature = float(temperature)
        if temperature <= 0:
            raise ValueError(f"`temperature` has to be a strictly positive float, but is {temperature}")

        self.temperature = temperature

    def __call__(self, input_ids, scores):
        scores = scores / self.temperature
        return scores


class RepetitionPenaltyLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] enforcing an exponential penalty on repeated sequences.

    Args:
        repetition_penalty (`float`):
            The parameter for repetition penalty. 1.0 means no penalty. See [this
            paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
    """

    def __init__(self, repetition_penalty: float):
        repetition_penalty = float(repetition_penalty)
        if repetition_penalty <= 0:
            raise ValueError(f"`penalty` has to be a strictly positive float, but is {repetition_penalty}")

        self.penalty = repetition_penalty

    def __call__(self, input_ids, scores):
        score = np.take_along_axis(scores, input_ids, axis=1)

        # if score < 0 then repetition penalty has to be multiplied to reduce the previous token probability
        negative_index = score < 0
        positive_index = ~negative_index
        score[negative_index] = score[negative_index] * self.penalty
        score[positive_index] = score[positive_index] / self.penalty

        np.put_along_axis(scores, input_ids, score, axis=1)
        return scores


class TopPLogitsWarper(LogitsWarper):
    """
    [`LogitsWarper`] that performs top-p, i.e. restricting to top tokens summing to prob_cut_off <= prob_cut_off.

    Args:
        top_p (`float`):
            If set to < 1, only the smallest set of most probable tokens with probabilities
            that add up to `top_p` or higher are kept for generation.
        filter_value (`float`, *optional*, defaults to `-50000`):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.
        candidate_token_num (`int`, *optional*, defaults to 200):
            Number of candidate tokens to calculate top_p. this can avoid sorting a huge seq,
            save time to speed up generation.
    """

    def __init__(
        self, top_p: float, filter_value: float = -50000, min_tokens_to_keep: int = 1, candidate_token_num: int = 200
    ):
        top_p = float(top_p)
        if top_p < 0 or top_p > 1.0:
            raise ValueError(f"`top_p` has to be a float > 0 and < 1, but is {top_p}")
        if not isinstance(min_tokens_to_keep, int) or (min_tokens_to_keep < 0):
            raise ValueError(f"`min_tokens_to_keep` has to be a non-negative integer, but is {min_tokens_to_keep}")

        self.top_p = top_p
        self.filter_value = float(filter_value)
        self.min_tokens_to_keep = min_tokens_to_keep
        self.candicate_token_num = candidate_token_num

    def __call__(self, input_ids, scores):
        candidate_logits, candidate_indices = topk(scores, self.candicate_token_num)
        cumulative_probs = softmax(candidate_logits)
        cumulative_probs = np.cumsum(cumulative_probs, axis=-1)
        # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
        sorted_indices_to_keep = cumulative_probs < self.top_p
        # add the last token that exceed top_p
        sorted_indices_to_keep = np.concatenate(
            [np.ones(shape=(scores.shape[0], 1)).astype(np.bool_), sorted_indices_to_keep[..., :-1]], axis=-1
        )
        # Keep at least min_tokens_to_keep
        sorted_indices_to_keep[..., : self.min_tokens_to_keep] = 1

        # set remove indices, filter negative value
        indices_to_remove = np.ones_like(scores).astype(np.bool_)
        np.put_along_axis(indices_to_remove, candidate_indices, ~sorted_indices_to_keep, axis=-1)
        scores[indices_to_remove] = self.filter_value

        return scores


class TopKLogitsWarper(LogitsWarper):
    r"""
    [`LogitsWarper`] that performs top-k, i.e. restricting to the k highest probability elements.

    Args:
        top_k (`int`):
            The number of highest probability vocabulary tokens to keep for top-k-filtering.
        filter_value (`float`, *optional*, defaults to `-float("Inf")`):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.
    """

    def __init__(self, top_k: int, filter_value: float = -50000, min_tokens_to_keep: int = 1):
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"`top_k` has to be a strictly positive integer, but is {top_k}")

        self.top_k = max(top_k, min_tokens_to_keep)
        self.filter_value = float(filter_value)

    def __call__(self, input_ids, scores: np.ndarray):
        top_k = min(self.top_k, scores.shape[-1])  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = scores < topk(scores, top_k)[0][..., -1, None]
        scores[indices_to_remove] = self.filter_value
        return scores


class LogitNormalization(LogitsProcessor, LogitsWarper):
    r"""
    [`LogitsWarper`] and [`LogitsProcessor`] for normalizing the scores using log-softmax. It's important to normalize
    the scores during beam search, after applying the logits processors or warpers, since the search algorithm used in
    this library doesn't do it (it only does it before, but they may need re-normalization) but it still supposes that
    the scores are normalized when comparing the hypotheses.
    """

    def __call__(self, input_ids, scores):
        scores = log_softmax(scores, axis=-1)
        return scores
