import mindspore as ms

__all__ = ["VQAReTokenLayoutLMPostProcess"]


class VQAReTokenLayoutLMPostProcess:
    """Convert between text-label and text-index"""

    def __init__(self, **kwargs):
        super(VQAReTokenLayoutLMPostProcess, self).__init__()

    def __call__(self, preds, **kwargs):
        label = kwargs["labels"]
        pred_relations = preds["pred_relations"]
        if isinstance(preds["pred_relations"], ms.Tensor):
            pred_relations = pred_relations.numpy()
        pred_relations = self.decode_pred(pred_relations)

        if label is not None:
            return self._metric(pred_relations)
        else:
            return self._infer(pred_relations, **kwargs)

    def _metric(self, pred_relations):
        return pred_relations

    def _infer(self, pred_relations, **kwargs):
        ser_results = kwargs["ser_results"]
        entity_idx_dict_batch = kwargs["entity_idx_dict_batch"]

        # merge relations and ocr info
        results = []
        for pred_relation, ser_result, entity_idx_dict in zip(pred_relations, ser_results, entity_idx_dict_batch):
            result = []
            used_tail_id = []
            for relation in pred_relation:
                if relation["tail_id"] in used_tail_id:
                    continue
                used_tail_id.append(relation["tail_id"])
                ocr_info_head = ser_result[entity_idx_dict[relation["head_id"]]]
                ocr_info_tail = ser_result[entity_idx_dict[relation["tail_id"]]]
                result.append((ocr_info_head, ocr_info_tail))
            results.append(result)
        return results

    def decode_pred(self, pred_relations):
        pred_relations_new = []
        for pred_relation in pred_relations:
            pred_relation_new = []
            pred_relation = pred_relation[1: pred_relation[0, 0, 0] + 1]
            for relation in pred_relation:
                relation_new = dict()
                relation_new["head_id"] = relation[0, 0]
                relation_new["head"] = tuple(relation[1])
                relation_new["head_type"] = relation[2, 0]
                relation_new["tail_id"] = relation[3, 0]
                relation_new["tail"] = tuple(relation[4])
                relation_new["tail_type"] = relation[5, 0]
                relation_new["type"] = relation[6, 0]
                pred_relation_new.append(relation_new)
            pred_relations_new.append(pred_relation_new)
        return pred_relations_new
