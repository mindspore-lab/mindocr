"""For text generation"""
import copy
import logging
import time
from typing import List, Optional, Union

import numpy as np

import mindspore.common.dtype as mstype
from mindspore import ops
from mindspore.common.tensor import Tensor

from mindocr.nlp.generation.beam_search import BeamSearchScorer
from mindocr.nlp.generation.generation_config import GenerationConfig
from mindocr.nlp.generation.logits_process import (
    LogitNormalization,
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from mindocr.nlp.generation.utils import softmax_with_threads, topk

__all__ = ["GeneratorMixin"]
_logger = logging.getLogger(__name__)


class GenerationMode:
    """
    Possible generation modes.
    """

    # Non-beam methods
    GREEDY_SEARCH = "greedy_search"
    SAMPLE = "sample"
    # Beam methods
    BEAM_SEARCH = "beam_search"


class GeneratorMixin:
    """Generator For the nlp models"""

    def __init__(self):
        pass

    # pylint: disable=W0613
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """
        prepare inputs for generation.
        A model class needs to define a `prepare_inputs_for_generation` method
        in order to use `.generate()`

        Raises:
            RuntimeError: Not implemented in model but call `.generate()`
        """
        raise RuntimeError(
            "A model class needs to define a `prepare_inputs_for_generation` method in order to use `.generate()`."
        )

    # pylint: disable=W0613
    def update_model_kwargs_before_generate(self, input_ids, model_kwargs: dict):
        """
        update model kwargs before generate.
        If your model needs to update model kwargs before generate, implement
        this method in your model, else do nothing.
        """
        return

    @staticmethod
    def slice_incremental_inputs(model_inputs: dict, current_index):
        """used for non-first iterations, slice the inputs to length 1."""
        input_ids = model_inputs.pop("input_ids")
        if isinstance(input_ids, Tensor):
            input_ids = input_ids.asnumpy()
        inputs_tmp = []
        for i, index_value in enumerate(current_index):
            current_index_tmp = int(index_value) - i * input_ids.shape[1]  # multi batch
            # use numpy to slice array to avoid compile ascend slice op
            inputs_tmp.append(input_ids[i][current_index_tmp : current_index_tmp + 1])
        inputs_tmp = np.array(inputs_tmp, dtype=np.int32)
        model_inputs["input_ids"] = Tensor(inputs_tmp, mstype.int32)

    @staticmethod
    def process_logits(logits, current_index=None, keep_all=False):
        """Process the logits"""
        logits = logits.reshape(-1, logits.shape[-1])
        if not keep_all and current_index is not None:
            index = current_index.view(
                -1,
            )
            logits = ops.Gather()(logits, index, 0)
        outputs = ops.LogSoftmax(-1)(logits)
        outputs = ops.tensor_pow(np.e, outputs)
        return outputs

    def _get_logits_processor(
        self, generation_config: GenerationConfig, logits_processor: Optional[LogitsProcessorList]
    ):
        """
        This class returns a [`LogitsProcessorList`] list object that contains all relevant [`LogitsProcessor`]
        instances used to modify the scores of the language model head.
        """
        # instantiate processors list
        processors = LogitsProcessorList()

        if generation_config.repetition_penalty is not None and generation_config.repetition_penalty != 1.0:
            processors.append(RepetitionPenaltyLogitsProcessor(repetition_penalty=generation_config.repetition_penalty))
        processors = self._merge_processor_list(processors, logits_processor)
        # `LogitNormalization` should always be the last logit processor, when present
        if generation_config.renormalize_logits is True:
            processors.append(LogitNormalization())
        return processors

    def _merge_processor_list(self, default_list: LogitsProcessorList, custom_list: LogitsProcessorList):
        """merge custom processor list with default list."""
        if not custom_list:
            return default_list
        for default in default_list:
            for custom in custom_list:
                if type(custom) is type(default):
                    object_type = "logits processor"
                    raise ValueError(
                        f"A custom {object_type} of type {type(custom)} with values {custom} has been passed to"
                        f" `.generate()`, but it has already been created with the values {default}."
                        f" {default} has been created by passing the corresponding arguments to generate or"
                        f" by the model's config default values. If you just want to change the default values"
                        f" of {object_type} consider passing them as arguments to `.generate()`"
                        f" instead of using a custom {object_type}."
                    )
        default_list.extend(custom_list)
        return default_list

    def _get_logits_warper(self, generation_config: GenerationConfig):
        """
        This class returns a [`LogitsProcessorList`] list object that contains all relevant [`LogitsWarper`] instances
        used for multinomial sampling.
        """

        # instantiate wrappers list
        wrappers = LogitsProcessorList()

        # all samplers can be found in `generation_utils_samplers.py`
        if generation_config.temperature is not None and generation_config.temperature != 1.0:
            wrappers.append(TemperatureLogitsWarper(generation_config.temperature))
        min_tokens_to_keep = 1
        if generation_config.top_k is not None and generation_config.top_k != 0:
            wrappers.append(TopKLogitsWarper(top_k=generation_config.top_k, min_tokens_to_keep=min_tokens_to_keep))
        if generation_config.top_p is not None and generation_config.top_p < 1.0:
            wrappers.append(TopPLogitsWarper(top_p=generation_config.top_p, min_tokens_to_keep=min_tokens_to_keep))
        # `LogitNormalization` should always be the last logit processor, when present
        if generation_config.renormalize_logits is True:
            wrappers.append(LogitNormalization())
        return wrappers

    @staticmethod
    def _get_generation_mode(generation_config: GenerationConfig):
        """determine the generation mode by config"""
        if generation_config.num_beams == 1:
            if generation_config.do_sample:
                _logger.info("The generation mode will be **SAMPLE**.")
                return GenerationMode.SAMPLE
            _logger.info("The generation mode will be **GREEDY_SEARCH**.")
            return GenerationMode.GREEDY_SEARCH
        _logger.info("The generation mode will be **BEAM_SEARCH**.")
        return GenerationMode.BEAM_SEARCH

    def _prepare_model_inputs_for_decoder(self, input_ids, input_mask):
        """generate the inputs for the decoder"""
        batch_size = input_ids.shape[0]

        encoder_mask = Tensor(input_mask, mstype.float32)

        encoder_output = self.encoder_forward(Tensor(input_ids, mstype.int32), encoder_mask)

        input_ids = np.zeros((batch_size, self.config.max_decode_length))
        _logger.debug("Decoder: pad the origin inputs into shape: %s", input_ids.shape)
        target_mask = np.zeros_like(input_ids)
        target_mask[:, 0] = 1

        # As the decoder is generating from [START] token
        return encoder_output, encoder_mask, input_ids, target_mask

    def _pad_inputs_using_max_length(self, origin_inputs, pad_token_id=0):
        """pad the input_ids to the max_length"""
        pad_length = self.config.seq_length - origin_inputs.shape[-1]
        if pad_length < 0:
            raise ValueError(
                f"origin_inputs size is {origin_inputs.shape}, you should"
                f"increase the seq_length of the model {self.config.seq_length}."
            )
        # Pad original inputs to model_origin_max_length
        input_ids = np.pad(
            origin_inputs,
            ((0, 0), (0, pad_length)),
            "constant",
            constant_values=(0, pad_token_id),
        )
        return input_ids

    def _incremental_infer(self, model_inputs: dict, current_index, valid_length_each_example):
        """model forward for incremental infer."""
        # Claim the first graph
        if self.is_first_iteration:
            self.add_flags_recursive(is_first_iteration=True)
            model_inputs["input_position"] = Tensor(current_index, mstype.int32)
            model_inputs["init_reset"] = Tensor([False], mstype.bool_)  # init_reset (1,) bool False
            model_inputs["batch_valid_length"] = Tensor([valid_length_each_example], mstype.int32)
            # pylint: disable=E1102
            res = self(**model_inputs)
            # first iter done, go to other iters
            self.is_first_iteration = False
            self.add_flags_recursive(is_first_iteration=False)
        else:
            # slice model inputs for incremental infer
            self.slice_incremental_inputs(model_inputs, current_index)
            model_inputs["input_position"] = Tensor(current_index, mstype.int32)
            model_inputs["init_reset"] = Tensor([True], mstype.bool_)  # init_reset (1,) bool True
            model_inputs["batch_valid_length"] = Tensor([valid_length_each_example], mstype.int32)
            # pylint: disable=E1102
            res = self(
                **model_inputs,
            )

        return res

    def _greedy_search(
        self,
        origin_inputs,
        generation_config: GenerationConfig,
        logits_processor: Optional[LogitsProcessorList] = None,
        streamer=None,
        **model_kwargs,
    ):
        r"""
        Generates sequences of token ids for models with a language modeling head using **greedy decoding** and can be
        used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        In most cases, you do not need to call [`~generation.GenerationMixin.greedy_search`] directly. Use generate()
        instead.

        Parameters:
            origin_inputs (`List(str), List(List(str))`):
                The sequence used as a prompt for the generation.
            generation_config (`GenerationConfig`, *optional*):
                The generation configuration to be used as base parametrization for the generation
                call. `**kwargs` passed to generate matching the attributes of `generation_config`
                will override them. If `generation_config` is not provided, the default config
                from the model configuration will be used. Please note that unspecified parameters
                will inherit [`GenerationConfig`]'s default values, whose documentation should be
                checked to parameterize generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            streamer (`TextStreamer, *optional*`):
                The streamer that generator uses.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            A list of the generated token ids
        """
        total_time = time.time()
        prepare_time = time.time()

        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()

        if generation_config.pad_token_id is None:
            generation_config.pad_token_id = 0

        if streamer is not None:
            streamer.put(origin_inputs)

        batch_size = origin_inputs.shape[0]
        is_encoder_decoder = self.config.is_encoder_decoder
        _logger.debug("The input shape is: %s", origin_inputs.shape)
        valid_length_each_example = []
        for i in range(batch_size):
            # As the nonzero returns the index and we need length
            valid_length_each_example.append(
                np.max(np.argwhere(origin_inputs[i] != generation_config.pad_token_id)) + 1
            )
        valid_length_each_example = np.array(valid_length_each_example)
        _logger.debug("Get the valid for each example is: %s", valid_length_each_example)

        # Prepare `max_length` depending on other stopping criteria.
        input_ids_length = np.max(valid_length_each_example)
        if generation_config.max_new_tokens is not None:
            generation_config.max_length = generation_config.max_new_tokens + input_ids_length

        if generation_config.max_length > self.config.seq_length:
            _logger.warning(
                "max_length %s can not exceeds model seq_length %s, set max_length = seq_length.",
                generation_config.max_length,
                self.config.seq_length,
            )
            generation_config.max_length = self.config.seq_length

        _logger.debug("max length is: %s", generation_config.max_length)
        if not is_encoder_decoder and input_ids_length >= generation_config.max_length:
            raise ValueError(
                f"the input_ids length {input_ids_length} exceeds the max length config {generation_config.max_length}."
                f"check your inputs and set max_length larger than your inputs length."
            )

        input_ids = self._pad_inputs_using_max_length(
            origin_inputs=origin_inputs, pad_token_id=generation_config.pad_token_id
        )

        _logger.debug(
            "pad the origin inputs from %s into shape: %s",
            origin_inputs.shape,
            input_ids.shape,
        )

        input_mask = np.zeros_like(input_ids)
        for i in range(valid_length_each_example.shape[0]):
            input_mask[i, : valid_length_each_example[i]] = 1
        encoder_output = None
        encoder_mask = None
        if is_encoder_decoder:
            if generation_config.max_length > self.config.max_decode_length:
                generation_config.max_length = self.config.max_decode_length
            _logger.debug("max decode length is: %s", generation_config.max_length)

            # When do encoder and decoder prediction, the encoder can be cached
            # to speed up the inference
            (
                encoder_output,
                encoder_mask,
                input_ids,
                target_mask,
            ) = self._prepare_model_inputs_for_decoder(input_ids, input_mask)
            valid_length_each_example = [1 for _ in range(batch_size)]
        # A single loop generates one token, loop until reaching target
        # model_origin_max_length or generating eod token
        is_finished = [False] * batch_size

        # update model kwargs once, before go into generate loop.
        self.update_model_kwargs_before_generate(input_ids, model_kwargs)

        # setup is_first_iteration flag for incremental infer
        if generation_config.use_past:
            self.is_first_iteration = True
        need_gather_logits = True

        origin_len = np.sum(valid_length_each_example)
        prepare_time = time.time() - prepare_time
        _logger.debug("forward prepare time: %s s", prepare_time)

        while np.sum(is_finished) != batch_size:
            forward_time = time.time()
            seq_length = input_ids.shape[1]
            current_index = [valid_length_each_example[i] - 1 + i * seq_length for i in range(batch_size)]
            _logger.debug("validate length: %s", valid_length_each_example)
            if is_encoder_decoder:
                inputs = Tensor(input_ids, mstype.int32)
                # pylint: disable=E1102
                res = self(
                    input_ids=None,
                    attention_mask=encoder_mask,
                    encoder_outputs=encoder_output,
                    decoder_input_ids=inputs,
                    decoder_attention_mask=Tensor(target_mask, mstype.float32),
                )
            else:
                model_kwargs["current_index"] = current_index
                # model prepare input dict
                model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
                # incremental generate
                if generation_config.use_past:
                    # when first iteration, gather last logits; others keep all logits.
                    need_gather_logits = self.is_first_iteration
                    # incremental generate
                    res = self._incremental_infer(
                        model_inputs=model_inputs,
                        current_index=current_index,
                        valid_length_each_example=valid_length_each_example,
                    )
                # auto-aggressive generate
                else:
                    res = self(**model_inputs)  # pylint: disable=E1102
            forward_time = time.time() - forward_time

            search_time = time.time()
            # post process logits; skip this phase if post process is done in graph
            if not self.config.is_sample_acceleration:
                # convert to numpy for post process
                logits = res[0] if isinstance(res, tuple) else res
                if isinstance(logits, Tensor):
                    logits = logits.asnumpy()
                logits = np.reshape(logits, (-1, logits.shape[-1]))
                # need gather last seq logits using current_index
                # compare length to determine if need gather; if not, gather should be done in model construct
                if need_gather_logits and logits.shape[0] > len(current_index):
                    logits = logits[current_index]

                # post process logits, without changing logits shape and order
                probs = logits_processor(input_ids, logits, is_finished)
                p_args = np.tile(np.arange(logits.shape[-1]), (batch_size, 1))
            else:
                probs, p_args = res
                if isinstance(probs, Tensor):
                    probs = probs.asnumpy()
                if isinstance(p_args, Tensor):
                    p_args = p_args.asnumpy()
            search_time = time.time() - search_time

            update_time = time.time()

            # Random select a token as final output for this round
            target_list = [[] for _ in range(batch_size)]
            for i in range(batch_size):
                if is_finished[i]:
                    continue

                target_index = np.argmax(probs[i])

                # get target token id
                target = p_args[i][target_index]
                input_ids[i, valid_length_each_example[i]] = target

                if streamer is not None:
                    # assign target element
                    target_list[i] = [target]

                if is_encoder_decoder:
                    target_mask[i][valid_length_each_example[i]] = int(1)

                valid_length_each_example[i] += int(1)
                input_mask[i][valid_length_each_example[i] - 1] = 1

                # Stop judgment
                if (
                    p_args[i][target_index] == generation_config.eos_token_id
                    or valid_length_each_example[i] == generation_config.max_length
                ):
                    is_finished[i] = True
                    continue
            if streamer is not None:
                if batch_size == 1:
                    streamer.put(target_list[0])
                else:
                    streamer.put(target_list)
            update_time = time.time() - update_time
            _logger.debug(
                "forward time: %s s; greedy search time: %s s; update time: %s s; total count: %s s",
                forward_time,
                search_time,
                update_time,
                forward_time + search_time + update_time,
            )

        # Return valid outputs out of padded outputs
        output_ids = []
        for i in range(batch_size):
            output_ids.append(input_ids[i, : int(valid_length_each_example[i])].astype(np.int32))
        _logger.debug("The output is: %s", output_ids)
        if streamer is not None:
            streamer.end()

        generate_len = np.sum(valid_length_each_example) - origin_len
        total_time = time.time() - total_time
        _logger.info(
            "total time: %s s; generated tokens: %s tokens; generate speed: %s tokens/s",
            total_time,
            generate_len,
            generate_len / total_time,
        )

        return output_ids

    def _sample(
        self,
        origin_inputs,
        generation_config: GenerationConfig,
        logits_processor: Optional[LogitsProcessorList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        streamer=None,
        **model_kwargs,
    ):
        r"""
        Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        Parameters:
            origin_inputs (`List(str), List(List(str))`):
                The sequence used as a prompt for the generation.
            generation_config (`GenerationConfig`, *optional*):
                The generation configuration to be used as base parametrization for the generation
                call. `**kwargs` passed to generate matching the attributes of `generation_config`
                will override them. If `generation_config` is not provided, the default config
                from the model configuration will be used. Please note that unspecified parameters
                will inherit [`GenerationConfig`]'s default values, whose documentation should be
                checked to parameterize generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            logits_warper (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
                to warp the prediction score distribution of the language modeling head applied before multinomial
                sampling at each generation step.
            streamer (`TextStreamer, *optional*`):
                The streamer that generator uses.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            A list of the generated token ids
        """
        total_time = time.time()
        prepare_time = time.time()

        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()

        if generation_config.pad_token_id is None:
            generation_config.pad_token_id = 0

        if streamer is not None:
            streamer.put(origin_inputs)

        batch_size = origin_inputs.shape[0]
        is_encoder_decoder = self.config.is_encoder_decoder
        _logger.debug("The input shape is: %s", origin_inputs.shape)
        valid_length_each_example = []
        for i in range(batch_size):
            # As the nonzero returns the index and we need length
            valid_length_each_example.append(
                np.max(np.argwhere(origin_inputs[i] != generation_config.pad_token_id)) + 1
            )
        valid_length_each_example = np.array(valid_length_each_example)
        _logger.debug("Get the valid for each example is: %s", valid_length_each_example)

        # Prepare `max_length` depending on other stopping criteria.
        input_ids_length = np.max(valid_length_each_example)
        if generation_config.max_new_tokens is not None:
            generation_config.max_length = generation_config.max_new_tokens + input_ids_length

        if generation_config.max_length > self.config.seq_length:
            _logger.warning(
                "max_length %s can not exceeds model seq_length %s, set max_length = seq_length.",
                generation_config.max_length,
                self.config.seq_length,
            )
            generation_config.max_length = self.config.seq_length

        _logger.debug("max length is: %s", generation_config.max_length)
        if not is_encoder_decoder and input_ids_length >= generation_config.max_length:
            raise ValueError(
                f"the input_ids length {input_ids_length} exceeds the max length config {generation_config.max_length}."
                f"check your inputs and set max_length larger than your inputs length."
            )

        input_ids = self._pad_inputs_using_max_length(
            origin_inputs=origin_inputs, pad_token_id=generation_config.pad_token_id
        )

        _logger.debug(
            "pad the origin inputs from %s into shape: %s",
            origin_inputs.shape,
            input_ids.shape,
        )

        input_mask = np.zeros_like(input_ids)
        for i in range(valid_length_each_example.shape[0]):
            input_mask[i, : valid_length_each_example[i]] = 1
        encoder_output = None
        encoder_mask = None
        if is_encoder_decoder:
            if generation_config.max_length > self.config.max_decode_length:
                generation_config.max_length = self.config.max_decode_length
            _logger.debug("max decode length is: %s", generation_config.max_length)

            # When do encoder and decoder prediction, the encoder can be cached
            # to speed up the inference
            (
                encoder_output,
                encoder_mask,
                input_ids,
                target_mask,
            ) = self._prepare_model_inputs_for_decoder(input_ids, input_mask)
            valid_length_each_example = [1 for _ in range(batch_size)]
        # A single loop generates one token, loop until reaching target
        # model_origin_max_length or generating eod token
        is_finished = [False] * batch_size

        # update model kwargs once, before go into generate loop.
        self.update_model_kwargs_before_generate(input_ids, model_kwargs)

        # setup is_first_iteration flag for incremental infer
        if generation_config.use_past:
            self.is_first_iteration = True
        need_gather_logits = True

        origin_len = np.sum(valid_length_each_example)
        prepare_time = time.time() - prepare_time
        _logger.debug("forward prepare time: %s s", prepare_time)

        while np.sum(is_finished) != batch_size:
            forward_time = time.time()
            seq_length = input_ids.shape[1]
            current_index = [valid_length_each_example[i] - 1 + i * seq_length for i in range(batch_size)]
            _logger.debug("validate length: %s", valid_length_each_example)
            if is_encoder_decoder:
                inputs = Tensor(input_ids, mstype.int32)
                # pylint: disable=E1102
                res = self(
                    input_ids=None,
                    attention_mask=encoder_mask,
                    encoder_outputs=encoder_output,
                    decoder_input_ids=inputs,
                    decoder_attention_mask=Tensor(target_mask, mstype.float32),
                )
            else:
                model_kwargs["current_index"] = current_index
                # model prepare input dict
                model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
                # incremental generate
                if generation_config.use_past:
                    # when first iteration, gather last logits; others keep all logits.
                    need_gather_logits = self.is_first_iteration
                    # incremental generate
                    res = self._incremental_infer(
                        model_inputs=model_inputs,
                        current_index=current_index,
                        valid_length_each_example=valid_length_each_example,
                    )
                # auto-aggressive generate
                else:
                    res = self(**model_inputs)  # pylint: disable=E1102
            forward_time = time.time() - forward_time

            sample_time = time.time()
            # post process logits; skip this phase if post process is done in graph
            if not self.config.is_sample_acceleration:
                # convert to numpy for post process
                logits = res[0] if isinstance(res, tuple) else res
                if isinstance(logits, Tensor):
                    logits = logits.asnumpy()
                logits = np.reshape(logits, (-1, logits.shape[-1]))
                # need gather last seq logits using current_index
                # compare length to determine if need gather; if not, gather should be done in model construct
                if need_gather_logits and logits.shape[0] > len(current_index):
                    logits = logits[current_index]

                # post process logits, without changing logits shape and order
                probs = logits_processor(input_ids, logits, is_finished)
                probs = logits_warper(input_ids, probs, is_finished)
                p_args = np.tile(np.arange(logits.shape[-1]), (batch_size, 1))
            else:
                probs, p_args = res
                if isinstance(probs, Tensor):
                    probs = probs.asnumpy()
                if isinstance(p_args, Tensor):
                    p_args = p_args.asnumpy()
            sample_time = time.time() - sample_time

            update_time = time.time()
            p_norms = softmax_with_threads(probs, is_finished)

            # Random select a token as final output for this round
            target_list = [[] for _ in range(batch_size)]
            for i in range(batch_size):
                if is_finished[i]:
                    continue

                p_norm = p_norms[i]
                target_index = np.random.choice(len(probs[i]), p=p_norm)

                # get target token id
                target = p_args[i][target_index]
                input_ids[i, valid_length_each_example[i]] = target

                if streamer is not None:
                    # assign target element
                    target_list[i] = [target]

                if is_encoder_decoder:
                    target_mask[i][valid_length_each_example[i]] = int(1)

                valid_length_each_example[i] += int(1)
                input_mask[i][valid_length_each_example[i] - 1] = 1

                # Stop judgment
                if (
                    p_args[i][target_index] == generation_config.eos_token_id
                    or valid_length_each_example[i] == generation_config.max_length
                ):
                    is_finished[i] = True
                    continue
            if streamer is not None:
                if batch_size == 1:
                    streamer.put(target_list[0])
                else:
                    streamer.put(target_list)
            update_time = time.time() - update_time
            _logger.debug(
                "forward time: %s s; sample time: %s s; update time: %s s; total count: %s s",
                forward_time,
                sample_time,
                update_time,
                forward_time + sample_time + update_time,
            )

        # Return valid outputs out of padded outputs
        output_ids = []
        for i in range(batch_size):
            output_ids.append(input_ids[i, : int(valid_length_each_example[i])].astype(np.int32))
        _logger.debug("The output is: %s", output_ids)

        if streamer is not None:
            streamer.end()

        generate_len = np.sum(valid_length_each_example) - origin_len
        total_time = time.time() - total_time
        _logger.info(
            "total time: %s s; generated tokens: %s tokens; generate speed: %s tokens/s",
            total_time,
            generate_len,
            generate_len / total_time,
        )

        return output_ids

    def _beam_search(
        self,
        origin_inputs,
        beam_scorer: BeamSearchScorer,
        generation_config: GenerationConfig,
        logits_processor: Optional[LogitsProcessorList] = None,
        streamer=None,
        **model_kwargs,
    ):
        r"""
        Generates sequences of token ids for models with a language modeling head using **beam search decoding** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        Parameters:
            origin_inputs (`List(str), List(List(str))`):
                The sequence used as a prompt for the generation.
            beam_scorer (`BeamScorer`):
                An derived instance of [`BeamScorer`] that defines how beam hypotheses are constructed, stored and
                sorted during generation. For more information, the documentation of [`BeamScorer`] should be read.
            generation_config (`GenerationConfig`, *optional*):
                The generation configuration to be used as base parametrization for the generation
                call. `**kwargs` passed to generate matching the attributes of `generation_config`
                will override them. If `generation_config` is not provided, the default config
                from the model configuration will be used. Please note that unspecified parameters
                will inherit [`GenerationConfig`]'s default values, whose documentation should be
                checked to parameterize generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            streamer (`TextStreamer, *optional*`):
                The streamer that generator uses.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            A list of the generated token ids
        """
        if streamer is not None:
            raise ValueError("Streamer does not support in beam search method yet!")
        if generation_config.use_past:
            raise ValueError("Beam search does not support incremental inference yet! Please set use_past to False.")
        if self.config.is_sample_acceleration:
            raise ValueError(
                "Beam search does not support sample acceleration yet! Please set is_sample_acceleration to False."
            )

        total_time = time.time()
        prepare_time = time.time()
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()

        if generation_config.pad_token_id is None:
            generation_config.pad_token_id = 0

        batch_size = len(beam_scorer._beam_hyps)  # pylint: disable=W0212
        num_beams = beam_scorer.num_beams
        batch_beam_size = origin_inputs.shape[0]
        _logger.debug("The input shape is: %s", origin_inputs.shape)
        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        is_encoder_decoder = self.config.is_encoder_decoder

        # get the valid length of each example
        valid_length_each_example = []
        for i in range(batch_beam_size):
            # As the nonzero returns the index and we need length
            valid_length_each_example.append(
                np.max(np.argwhere(origin_inputs[i] != generation_config.pad_token_id)) + 1
            )
        valid_length_each_example = np.array(valid_length_each_example)
        _logger.debug("Get the valid for each example is: %s", valid_length_each_example)
        if not is_encoder_decoder and np.max(valid_length_each_example) > generation_config.max_length:
            raise ValueError(
                "The max_length set is smaller than the length in the input_ids."
                f"You shout set max_length to {np.max(valid_length_each_example)}"
            )

        target_length = (
            self.config.seq_length
            if generation_config.max_length > self.config.seq_length
            else generation_config.max_length
        )
        _logger.debug("max target_length is: %s", target_length)
        input_ids = self._pad_inputs_using_max_length(
            origin_inputs=origin_inputs, pad_token_id=generation_config.pad_token_id
        )

        _logger.debug(
            "pad the origin inputs from %s into shape: %s",
            origin_inputs.shape,
            input_ids.shape,
        )

        beam_scores = np.zeros((batch_size, num_beams), dtype=np.float64)
        beam_scores[:, 1:] = -1e9

        input_mask = np.zeros_like(input_ids)
        for i in range(valid_length_each_example.shape[0]):
            input_mask[i, : valid_length_each_example[i]] = 1
        encoder_output = None
        encoder_mask = None
        if is_encoder_decoder:
            if target_length > self.config.max_decode_length:
                target_length = self.config.max_decode_length
            _logger.debug("target_length is: %s", target_length)

            # When do encoder and decoder prediction, the encoder can be cached
            # to speed up the inference
            (
                encoder_output,
                encoder_mask,
                input_ids,
                target_mask,
            ) = self._prepare_model_inputs_for_decoder(input_ids, input_mask)
            valid_length_each_example = np.ones((batch_beam_size, 1)).astype(np.int32)

        # update model kwargs once, before go into generate loop.
        self.update_model_kwargs_before_generate(input_ids, model_kwargs)

        # setup is_first_iteration flag for incremental infer
        if generation_config.use_past:
            self.is_first_iteration = True
        need_gather_logits = True

        is_first_token = True

        origin_len = np.sum(valid_length_each_example) / num_beams
        prepare_time = time.time() - prepare_time
        _logger.debug("forward prepare time: %s s", prepare_time)

        while True:
            forward_time = time.time()
            seq_length = input_ids.shape[1]
            current_index = [valid_length_each_example[i] - 1 + i * seq_length for i in range(batch_beam_size)]
            _logger.debug("validate length: %s", valid_length_each_example)
            if is_encoder_decoder:
                inputs = Tensor(input_ids, mstype.int32)
                # pylint: disable=E1102
                res = self(
                    input_ids=None,
                    attention_mask=encoder_mask,
                    encoder_outputs=encoder_output,
                    decoder_input_ids=inputs,
                    decoder_attention_mask=Tensor(target_mask, mstype.float32),
                )
            else:
                model_kwargs["current_index"] = current_index
                # model prepare input dict
                model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
                # incremental generate
                if generation_config.use_past:
                    _logger.warning(
                        "Beam search currently not support incremental, auto-aggressive generate will be performed."
                    )
                # auto-aggressive generate
                res = self(**model_inputs)
            forward_time = time.time() - forward_time

            search_time = time.time()
            # post process logits
            # convert to numpy for post process
            logits = res[0] if isinstance(res, tuple) else res
            if isinstance(logits, Tensor):
                logits = logits.asnumpy().astype(np.float32)
            logits = np.reshape(logits, (-1, logits.shape[-1]))  # (batch_size * num_beams * seq_length, vocab_size)
            # need gather last seq logits using current_index
            # compare length to determine if need gather; if not, gather should be done in model construct
            if need_gather_logits and logits.shape[0] > len(current_index):
                logits = logits[current_index]  # (total_batch_size, vocab_size)
            logits_processor.append(LogitNormalization())

            # post process logits, without changing logits shape and order
            next_token_scores = logits_processor(input_ids, logits)  # (batch_size * num_beams, vocab_size)

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = np.reshape(next_token_scores, (batch_size, -1))  # (batch_size, num_beams * vocab_size)

            if is_first_token:
                next_token_scores = next_token_scores[:, :vocab_size]
                is_first_token = False

            # sample 2 next tokens for each beam, so we have at least 1 non eos token per beam
            next_token_scores, next_tokens = topk(next_token_scores, 2 * num_beams, axis=1, largest=True, sort=True)

            next_indices = np.floor_divide(next_tokens, vocab_size)
            next_tokens = next_tokens % vocab_size

            beam_outputs = beam_scorer.process(
                input_ids,  # (batch_size * num_beams, seq_length)
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
            )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]
            search_time = time.time() - search_time

            update_time = time.time()
            # reorder model inputs
            old_input_ids = input_ids.copy()
            for i in range(batch_beam_size):
                input_ids[i] = old_input_ids[beam_idx[i], :]

            # add new tokens to input_ids
            for i in range(batch_beam_size):
                input_ids[i, valid_length_each_example[i]] = beam_next_tokens[i]
                if is_encoder_decoder:
                    target_mask[i][valid_length_each_example[i]] = int(1)

                input_mask[i][valid_length_each_example[i]] = 1
                valid_length_each_example[i] += int(1)

            update_time = time.time() - update_time
            _logger.debug(
                "forward time: %s s; beam search time: %s s; update time: %s s; total count: %s s",
                forward_time,
                search_time,
                update_time,
                forward_time + search_time + update_time,
            )

            if beam_scorer.is_done or np.min(valid_length_each_example) >= generation_config.max_length:
                break

        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            pad_token_id=generation_config.pad_token_id,
            eos_token_id=generation_config.eos_token_id,
            max_length=generation_config.max_length,
        )

        generate_len = np.sum(valid_length_each_example) / num_beams - origin_len
        total_time = time.time() - total_time
        _logger.info(
            "total time: %s s; generated tokens: %s tokens; generate speed: %s tokens/s",
            total_time,
            generate_len,
            generate_len / total_time,
        )

        return sequence_outputs["sequences"]

    def generate(
        self,
        input_ids: Optional[Union[List[int], List[List[int]]]],
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        streamer=None,
        seed: Optional[int] = None,
        **kwargs,
    ):
        origin_phase = self.phase
        self.set_train(False)
        try:
            input_ids = np.array(input_ids)
        except ValueError as e:
            raise ValueError(
                str(e) + " Please check your inputs of model.generate(),"
                " and make sure the inputs are padded to same length."
            )
        input_ids = np.reshape(input_ids, (-1, np.shape(input_ids)[-1]))
        batch_size = input_ids.shape[0]
        seed = 0 if seed is None else seed
        np.random.seed(seed)

        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()

        # Handle `generation_config` and kwargs that might update it
        # priority: `generation_config` argument > `model.generation_config` (default config)
        if generation_config is None:
            # legacy: users may modify the model configuration to control generation
            # model attribute accordingly, if it was created from the model config
            generation_config = GenerationConfig.from_model_config(self.config)
        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**kwargs)  # All unused kwargs must be model kwargs

        if generation_config.num_beams > 1:
            _logger.warning(
                "When num_beams is set to a value greater than 1, do_sample will be set to False, "
                "due to the current beam search does not support sampling."
            )
            generation_config.do_sample = False
        if not generation_config.do_sample:
            _logger.warning(
                "When do_sample is set to False, top_k will be set to 1 and top_p will be set to 0, "
                "making them inactive."
            )
            generation_config.top_p = 1.0
            generation_config.top_k = 0
        _logger.info("Generation Config is: %s", generation_config)

        logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            logits_processor=logits_processor,
        )

        # determine generation mode
        generation_mode = self._get_generation_mode(generation_config)

        if streamer is not None and (generation_config.num_beams > 1):
            raise ValueError("`streamer` cannot be used with beam search yet. Make sure that `num_beams` is set to 1.")

        if generation_mode == GenerationMode.GREEDY_SEARCH:
            # run greedy search
            output_ids = self._greedy_search(
                origin_inputs=input_ids,
                generation_config=generation_config,
                logits_processor=logits_processor,
                streamer=streamer,
                **model_kwargs,
            )

        elif generation_mode == GenerationMode.SAMPLE:
            # prepare logits warper
            logits_warper = self._get_logits_warper(generation_config)

            # run sample
            output_ids = self._sample(
                origin_inputs=input_ids,
                generation_config=generation_config,
                logits_processor=logits_processor,
                logits_warper=logits_warper,
                streamer=streamer,
                **model_kwargs,
            )

        elif generation_mode == GenerationMode.BEAM_SEARCH:
            # prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size, num_beams=generation_config.num_beams, max_length=generation_config.max_length
            )
            # interleave input_ids with `num_beams` additional sequences per batch
            input_ids = np.repeat(input_ids, generation_config.num_beams, 0)

            # run beam search
            output_ids = self._beam_search(
                origin_inputs=input_ids,
                beam_scorer=beam_scorer,
                generation_config=generation_config,
                logits_processor=logits_processor,
                streamer=streamer,
                **model_kwargs,
            )

        # set to original phase
        self.set_train(origin_phase == "train")
        return output_ids
