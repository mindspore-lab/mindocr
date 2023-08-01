class ConversionMixin:
    @classmethod
    def support_conversion(cls, config: PretrainedConfig) -> bool:
        """check wether the model support conversion"""
        try:
            # try to get the name-mapping info
            _ = cls._get_name_mappings(config)
        except NotImplementedError:
            return False
        finally:
            return True

    @classmethod
    def convert(cls, weight_file: str, config: PretrainedConfig, cache_dir: str) -> None:
        """the entry of converting config and converting model file

        Args:
            input_dir (str | None): the input dir which contains `pytorch_model.bin` and `config.json` file
            config (PretrainedConfig): the PretrainedConfig instance of model
        """
        # FIXME(wj-Mcat): add compatibility with downstream models
        name_mappings = cls._get_name_mappings(config)

        state_dict = load_torch(weight_file)

        # 3. convert state_dict
        all_layer_names = set(state_dict.keys())
        for name_mapping in name_mappings:
            if name_mapping.source_name not in state_dict:
                logger.warning(f"key<{name_mapping.source_name}> not in the pytorch weight file.")
                continue

            state_dict[name_mapping.target_name] = name_mapping.run(state_dict, name_mapping.source_name)
            if name_mapping.source_name in all_layer_names:
                all_layer_names.remove(name_mapping.source_name)

        if all_layer_names:
            logger.warning(f"there are {len(all_layer_names)} tensors not initialized:")
            for layer_name in all_layer_names:
                logger.warning(f"--- {layer_name}")

        model_weight_file = os.path.join(cache_dir, PADDLE_WEIGHTS_NAME)
        paddle.save(state_dict, model_weight_file)
        return state_dict

    @classmethod
    def _get_name_mappings(cls, config: PretrainedConfig) -> List[StateDictNameMapping]:
        """get name mapping of PretrainedModel

        Args:
            config (PretrainedConfig): the configuration of name-mapping

        Raises:
            NotImplementedError:

        Returns:
            List[StateDictNameMapping]: the name-mappings of pretrained model
        """
        raise NotImplementedError

    @classmethod
    def get_tensor_parallel_convert_actions(cls, config: PretrainedConfig, loaded_state_dict_keys, ignore_error=False):
        name_action_mappings = cls._get_tensor_parallel_mappings(config)
        state_keys_map = cls._resolve_prefix_keys(name_action_mappings.keys(), loaded_state_dict_keys, ignore_error)
        for k, v in state_keys_map.items():
            name_action_mappings[v] = name_action_mappings.pop(k)
        return name_action_mappings

    @classmethod
    def convert_tensor_parallel(
        cls, weight_file: str, config: PretrainedConfig, state_dict=None, ignore_error=False
    ) -> None:
        """the entry of converting config and converting model file

        Args:
            weight_file (str | None): the weight file path of `model_state.pdparams` file
            config (PretrainedConfig): the PretrainedConfig instance of model
        """
        name_action_mappings = cls._get_tensor_parallel_mappings(config)
        if state_dict is None:
            with device_guard("cpu"):
                state_dict = paddle.load(weight_file, return_numpy=False)
            logger.info("starting convert orignal state_dict to tensor parallel state_dict.")

        state_keys_map = cls._resolve_prefix_keys(name_action_mappings.keys(), state_dict.keys(), ignore_error)

        for k, v in state_keys_map.items():
            name_action_mappings[v] = name_action_mappings.pop(k)

        for name, action in name_action_mappings.items():
            if name not in state_dict:
                if not ignore_error:
                    logger.warning(f"key<{name}> not in the model state weight file.")
                continue
            tensor = state_dict.pop(name)
            new_tensor = action(tensor)
            with device_guard("cpu"):
                state_dict[name] = paddle.Tensor(new_tensor, zero_copy=True)

        return state_dict

    @classmethod
    def merge_tensor_parallel(cls, state_dict, config) -> None:
        """the entry of converting config and converting model file

        Args:
            input_dir (str | None): the input dir which contains `pytorch_model.bin` and `config.json` file
            config (PretrainedConfig): the PretrainedConfig instance of model
        """
        name_action_mappings = cls._get_tensor_parallel_mappings(config, is_split=False)
        state_keys_map = cls._resolve_prefix_keys(name_action_mappings.keys(), state_dict.keys())

        for k, v in state_keys_map.items():
            name_action_mappings[v] = name_action_mappings.pop(k)

        state_dict_to_save = {}

        hcg = paddle.distributed.fleet.get_hybrid_communicate_group()
        mp_group = hcg.get_model_parallel_group()
        is_dst = paddle.distributed.get_rank(mp_group) == 0

        for key in state_dict.keys():
            tensor = state_dict[key]
            if key in name_action_mappings:
                ret = distributed_gather(tensor, group=mp_group, offload=True)
                action = name_action_mappings.pop(key)
                tensor = action(ret) if is_dst else None
            else:
                tensor = tensor.numpy() if is_dst else None

            # keep state dict use paddle.tensor
            if isinstance(tensor, np.ndarray):
                with device_guard("cpu"):
                    tensor = paddle.Tensor(tensor, zero_copy=True)

            state_dict_to_save[key] = tensor

        if len(name_action_mappings) > 0:
            for x in name_action_mappings.keys():
                logger.warning(f"key <{x}> need to merge tensor parallel but we can't find in model state.")

        return state_dict_to_save

    @classmethod
    def _get_tensor_parallel_mappings(cls, config: PretrainedConfig, is_split=True) -> List[StateDictNameMapping]:
        """get name mapping of PretrainedModel

        Args:
            config (PretrainedConfig): the configuration of name-mapping

        Raises:
            NotImplementedError:

        Returns:
            List[StateDictNameMapping]: the name-mappings for tensor_parallel
        """
        raise NotImplementedError

    @staticmethod
    def _resolve_prefix_keys(state_keys_base, state_keys_real, ignore_error=False):
        # state_keys_map base to real
        state_keys_map = {}

        state_keys_base = set(state_keys_base)
        state_keys_real = set(state_keys_real)

        for key in state_keys_base:
            for x in state_keys_real:
                if x.endswith(key):
                    state_keys_map[key] = x
                    break
            if key not in state_keys_map:
                if not ignore_error:
                    logger.error(f"tensor parallel conversion: could not find name {key} in loaded state dict!")
            else:
                state_keys_real.remove(state_keys_map[key])

        return state_keys_map