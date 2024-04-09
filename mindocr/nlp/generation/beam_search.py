"""Beam search for text generation."""
from abc import ABC, abstractmethod
from collections import UserDict
from typing import List, Optional, Union

import numpy as np


class BeamScorer(ABC):
    """Abstract base class for all beam scorers"""

    @abstractmethod
    def process(
        self, input_ids, next_scores, next_tokens, next_indices, pad_token_id, eos_token_id, beam_indices, group_index
    ):
        r"""
        Args:
            input_ids:
                Indices of input sequence tokens in the vocabulary.
            next_scores:
                Current scores of the top `2 * num_beams` non-finished beam hypotheses.
            next_tokens:
                `input_ids` of the tokens corresponding to the top `2 * num_beams` non-finished beam hypotheses.
            next_indices:
                Beam indices indicating to which beam hypothesis the `next_tokens` correspond.
            pad_token_id:
                The id of the *padding* token.
            eos_token_id:
                The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
            beam_indices:
                Beam indices indicating to which beam hypothesis each token correspond.
            group_index:
                The index of the group of beams.
        """
        raise NotImplementedError("This is an abstract method.")

    def finalize(self, input_ids, final_beam_scores, max_length, pad_token_id, eos_token_id, beam_indices):
        r"""
        Args:
            input_ids:
                Indices of input sequence tokens in the vocabulary.
            final_beam_scores:
                The final scores of all non-finished beams.
            max_length:
                The max_length of output ids.
            pad_token_id:
                The id of the *padding* token.
            eos_token_id:
                The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
            beam_indices:
                Beam indices indicating to which beam hypothesis each token correspond.
        """
        raise NotImplementedError("This is an abstract method.")


class BeamSearchScorer(BeamScorer):
    r"""
    [`BeamScorer`] implementing standard beam search decoding.

    Args:
        batch_size (`int`):
            Batch Size of `input_ids` for which standard beam search decoding is run in parallel.
        num_beams (`int`):
            Number of beams for beam search.
        length_penalty (`float`, *optional*, defaults to 1.0):
            Exponential penalty to the length that is used with beam-based generation. It is applied as an exponent to
            the sequence length, which in turn is used to divide the score of the sequence. Since the score is the log
            likelihood of the sequence (i.e. negative), `length_penalty` > 0.0 promotes longer sequences, while
            `length_penalty` < 0.0 encourages shorter sequences.
        do_early_stopping (`bool` or `str`, *optional*, defaults to `False`):
            Controls the stopping condition for beam-based methods, like beam-search. It accepts the following values:
            `True`, where the generation stops as soon as there are `num_beams` complete candidates; `False`, where an
            heuristic is applied and the generation stops when is it very unlikely to find better candidates;
            `"never"`, where the beam search procedure only stops when there cannot be better candidates (canonical
            beam search algorithm).
        num_beam_hyps_to_keep (`int`, *optional*, defaults to 1):
            The number of beam hypotheses that shall be returned upon calling
            [`~transformer.BeamSearchScorer.finalize`].
        num_beam_groups (`int`):
            Number of groups to divide `num_beams` into in order to ensure diversity among different groups of beams.
            See [this paper](https://arxiv.org/pdf/1610.02424.pdf) for more details.
        max_length (`int`, *optional*):
            The maximum length of the sequence to be generated.
    """

    def __init__(
        self,
        batch_size: int,
        num_beams: int,
        length_penalty: Optional[float] = 1.0,
        do_early_stopping: Optional[Union[bool, str]] = False,
        num_beam_hyps_to_keep: Optional[int] = 1,
        num_beam_groups: Optional[int] = 1,
        max_length: Optional[int] = None,
    ):
        self.num_beams = num_beams
        self.length_penalty = length_penalty
        self.do_early_stopping = do_early_stopping
        self.num_beam_hyps_to_keep = num_beam_hyps_to_keep
        self.num_beam_groups = num_beam_groups
        self.group_size = self.num_beams // self.num_beam_groups

        self._is_init = False
        # self._beam_hyps[i*self.num_beam_groups+j] is the beam_hyps of the j-th group in the i-th mini-batch.
        # If group_beam_search is not used, the list consists of `batch_size` beam_hyps.
        self._beam_hyps = [
            BeamHypotheses(
                num_beams=self.group_size,
                length_penalty=self.length_penalty,
                early_stopping=self.do_early_stopping,
                max_length=max_length,
            )
            for _ in range(batch_size * self.num_beam_groups)
        ]
        # self._done[i*self.num_beam_groups+j] indicates whether the generation of the beam_hyps of the j-th group
        # in the i-th mini-batch is complete.
        self._done = np.array([False for _ in range(batch_size * self.num_beam_groups)])

        if not isinstance(num_beams, int) or num_beams <= 1:
            raise ValueError(
                f"`num_beams` has to be an integer strictly greater than 1, but is {num_beams}. For `num_beams` == 1,"
                " one should make use of `greedy_search` instead."
            )

        if not isinstance(num_beam_groups, int) or (num_beam_groups > num_beams) or (num_beams % num_beam_groups != 0):
            raise ValueError(
                "`num_beam_groups` has to be an integer smaller or equal than `num_beams` and `num_beams` has to be"
                f" divisible by `num_beam_groups`, but is {num_beam_groups} with `num_beams` being {num_beams}."
            )

    @property
    def is_done(self) -> bool:
        return self._done.all()

    def process(
        self,
        input_ids,
        next_scores,
        next_tokens,
        next_indices,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        beam_indices=None,
        group_index: Optional[int] = 0,
    ):
        batch_size = len(self._beam_hyps) // self.num_beam_groups

        if not batch_size == (input_ids.shape[0] // self.group_size):
            if self.num_beam_groups > 1:
                raise ValueError(
                    f"A group beam size of {input_ids.shape[0]} is used as the input, but a group beam "
                    f"size of {self.group_size} is expected by the beam scorer."
                )
            raise ValueError(
                f"A beam size of {input_ids.shape[0]} is used as the input, but a beam size of "
                f"{self.group_size} is expected by the beam scorer."
            )

        next_beam_scores = np.zeros((batch_size, self.group_size), dtype=next_scores.dtype)
        next_beam_tokens = np.zeros((batch_size, self.group_size), dtype=next_tokens.dtype)
        next_beam_indices = np.zeros((batch_size, self.group_size), dtype=next_indices.dtype)

        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]

        for batch_idx in range(batch_size):
            batch_group_idx = batch_idx * self.num_beam_groups + group_index
            if self._done[batch_group_idx]:
                if self.num_beams < len(self._beam_hyps[batch_group_idx]):
                    raise ValueError(f"Batch can only be done if at least {self.num_beams} beams have been generated")
                if eos_token_id is None or pad_token_id is None:
                    raise ValueError("Generated beams >= num_beams -> eos_token_id and pad_token have to be defined")
                # pad the batch
                next_beam_scores[batch_idx, :] = 0
                next_beam_tokens[batch_idx, :] = pad_token_id
                next_beam_indices[batch_idx, :] = 0
                continue

            # next tokens for this sentence
            beam_idx = 0
            cur_len = np.min(np.where(input_ids[beam_idx] == pad_token_id))
            for beam_token_rank, (next_token, next_score, next_index) in enumerate(
                zip(next_tokens[batch_idx], next_scores[batch_idx], next_indices[batch_idx])
            ):
                batch_beam_idx = batch_idx * self.group_size + next_index
                # add to generated hypotheses if end of sentence
                if (eos_token_id is not None) and (next_token in eos_token_id):
                    # if beam_token does not belong to top num_beams tokens, it should not be added
                    is_beam_token_worse_than_top_num_beams = beam_token_rank >= self.group_size
                    if is_beam_token_worse_than_top_num_beams:
                        continue
                    if beam_indices is not None:
                        beam_index = beam_indices[batch_beam_idx]
                        beam_index = beam_index + (batch_beam_idx,)
                    else:
                        beam_index = None

                    self._beam_hyps[batch_group_idx].add(
                        input_ids[batch_beam_idx].copy(),
                        next_score,
                        beam_indices=beam_index,
                    )
                else:
                    # add next predicted token since it is not eos_token
                    next_beam_scores[batch_idx, beam_idx] = next_score
                    next_beam_tokens[batch_idx, beam_idx] = next_token
                    next_beam_indices[batch_idx, beam_idx] = batch_beam_idx
                    beam_idx += 1

                # once the beam for next step is full, don't add more tokens to it.
                if beam_idx == self.group_size:
                    break

            if beam_idx < self.group_size:
                raise ValueError(
                    f"At most {self.group_size} tokens in {next_tokens[batch_idx]} can be equal to `eos_token_id:"
                    f" {eos_token_id}`. Make sure {next_tokens[batch_idx]} are corrected."
                )

            # Check if we are done so that we can save a pad step if all(done)
            self._done[batch_group_idx] = self._done[batch_group_idx] or self._beam_hyps[batch_group_idx].is_done(
                next_scores[batch_idx].max(), cur_len
            )

        return UserDict(
            {
                "next_beam_scores": next_beam_scores.flatten(),
                "next_beam_tokens": next_beam_tokens.flatten(),
                "next_beam_indices": next_beam_indices.flatten(),
            }
        )

    def finalize(
        self,
        input_ids,
        final_beam_scores,
        max_length: int,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        beam_indices=None,
    ):
        batch_size = len(self._beam_hyps) // self.num_beam_groups

        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]

        # finalize all open beam hypotheses and add to generated hypotheses
        for batch_group_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_group_idx]:
                continue

            # all open beam hypotheses are added to the beam hypothesis
            # beam hypothesis class automatically keeps the best beams
            for index_per_group in range(self.group_size):
                batch_beam_idx = batch_group_idx * self.group_size + index_per_group
                final_score = final_beam_scores[batch_beam_idx].item()
                final_tokens = input_ids[batch_beam_idx]
                beam_index = beam_indices[batch_beam_idx] if beam_indices is not None else None
                beam_hyp.add(final_tokens, final_score, beam_indices=beam_index)

        # select the best hypotheses
        sent_lengths = np.zeros(batch_size * self.num_beam_hyps_to_keep, dtype=np.int32)
        best = []
        best_indices = []
        best_scores = np.zeros(batch_size * self.num_beam_hyps_to_keep, dtype=np.float32)

        # retrieve best hypotheses
        for i in range(batch_size):
            beam_hyps_in_batch = self._beam_hyps[i * self.num_beam_groups : (i + 1) * self.num_beam_groups]
            candidate_beams = [beam for beam_hyp in beam_hyps_in_batch for beam in beam_hyp.beams]
            sorted_hyps = sorted(candidate_beams, key=lambda x: x[0])
            for j in range(self.num_beam_hyps_to_keep):
                best_hyp_tuple = sorted_hyps.pop()
                best_score = best_hyp_tuple[0]
                best_hyp = best_hyp_tuple[1]
                best_index = best_hyp_tuple[2]
                sent_lengths[self.num_beam_hyps_to_keep * i + j] = len(best_hyp)

                # append hyp to lists
                best.append(best_hyp)

                # append indices to list
                best_indices.append(best_index)

                best_scores[i * self.num_beam_hyps_to_keep + j] = best_score

        # prepare for adding eos
        sent_lengths_max = max(sent_lengths) + 1
        sent_max_len = min(sent_lengths_max, max_length) if max_length is not None else sent_lengths_max
        decoded = np.zeros((batch_size * self.num_beam_hyps_to_keep, sent_max_len), dtype=np.int32)

        if best_indices and best_indices[0] is not None:
            indices = np.zeros((batch_size * self.num_beam_hyps_to_keep, sent_max_len), dtype=np.int32)
        else:
            indices = None

        # shorter batches are padded if needed
        if min(sent_lengths) != max(sent_lengths):
            if pad_token_id is None:
                raise ValueError("`pad_token_id` has to be defined")
            decoded.fill(pad_token_id)

        if indices is not None:
            indices.fill(-1)

        # fill with hypotheses and eos_token_id if the latter fits in
        for i, (hypo, best_idx) in enumerate(zip(best, best_indices)):
            sent_length = min(decoded.shape[-1], sent_lengths[i])
            decoded[i, :sent_length] = hypo[:sent_length]

            if indices is not None:
                indices[i, : len(best_idx)] = best_idx

            if sent_lengths[i] < sent_max_len:
                # inserting only the first eos_token_id
                decoded[i, sent_lengths[i]] = eos_token_id[0]

        return UserDict(
            {
                "sequences": decoded,
                "sequence_scores": best_scores,
                "beam_indices": indices,
            }
        )


class BeamHypotheses:
    """
    Beam hypotheses maintaining n-best list
    """

    def __init__(self, num_beams: int, length_penalty: float, early_stopping: bool, max_length: Optional[int] = None):
        """
        Initialize n-best list of hypotheses.
        """
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.max_length = max_length
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = 1e9

        if not isinstance(self.early_stopping, bool) and self.max_length is None:
            raise ValueError(
                "When `do_early_stopping` is set to a string, `max_length` must be defined. Ensure it is passed to the"
                " BeamScorer class instance at initialization time."
            )

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.beams)

    def add(self, hyp, sum_logprobs: float, beam_indices=None):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / (hyp.shape[-1] ** self.length_penalty)
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp, beam_indices))
            if len(self) > self.num_beams:
                sorted_next_scores = sorted([(s, idx) for idx, (s, _, _) in enumerate(self.beams)])
                del self.beams[sorted_next_scores[0][1]]
                self.worst_score = sorted_next_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs: float, cur_len: int) -> bool:
        """
        If there are enough hypotheses and that none of the hypotheses being generated can become better than the worst
        one in the heap, then we are done with this sentence.
        """

        if len(self) < self.num_beams:
            return False

        # `True`: stop as soon as at least `num_beams` hypotheses are finished
        if self.early_stopping is True:
            return True

        # `False`: heuristic compute the best possible score from `cur_len`, even though it is not entirely accurate
        #  when `length_penalty` is positive.
        if self.early_stopping is False:
            highest_attainable_score = best_sum_logprobs / cur_len**self.length_penalty
            ret = self.worst_score >= highest_attainable_score
            return ret

        # `"never"`: compute the best possible score, depending on the signal of `length_penalty`
        # `length_penalty` > 0.0 -> max denominator is obtained from `max_length`, not from `cur_len` -> min
        # abs(`highest_attainable_score`) is obtained -> `highest_attainable_score` is negative, hence we obtain
        # its max this way
        if self.length_penalty > 0.0:
            highest_attainable_score = best_sum_logprobs / self.max_length**self.length_penalty
        # the opposite logic applies here (max `highest_attainable_score` from `cur_len`)
        else:
            highest_attainable_score = best_sum_logprobs / cur_len**self.length_penalty
        ret = self.worst_score >= highest_attainable_score
        return ret
