import json
import os
from glob import glob

import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score

from mindspore import get_context, nn

from mindocr.utils.kie_utils import Synchronizer
from mindocr.utils.misc import AllReduce

__all__ = ["VQASerTokenMetric", "VQAReTokenMetric"]


class VQASerTokenMetric(nn.Metric):
    """
    Metric method for token classification.
    """

    def __init__(self, device_num: int = 1, **kwargs):
        super().__init__()
        self.clear()
        self.device_num = device_num
        self.synchronizer = None if device_num <= 1 else Synchronizer(device_num)
        self.metric_names = ["precision", "recall", "hmean"]
        if "save_dir" in kwargs:
            self.save_dir = kwargs["save_dir"]

    def update(self, output_batch, gt):
        preds, gt = output_batch
        self.pred_list.extend(preds)
        self.gt_list.extend(gt)

    def eval(self):
        gt_list = self.gt_list
        pred_list = self.pred_list

        if self.synchronizer:
            eval_dir = os.path.join(self.save_dir, "eval_tmp")
            os.makedirs(eval_dir, exist_ok=True)

            device_id = get_context("device_id")
            eval_path = os.path.join(eval_dir, f"eval_result_{device_id}.txt")
            with open(eval_path, "w") as fp:
                json.dump({"gt_list": gt_list, "pred_list": pred_list}, fp)
            self.synchronizer()

            eval_files = glob(eval_dir + "/*")
            gt_list = []
            pred_list = []
            for e_file in eval_files:
                with open(e_file, "r") as fp:
                    eval_info = json.load(fp)
                    gt_list += eval_info["gt_list"]
                    pred_list += eval_info["pred_list"]

        metrics = {
            "precision": precision_score(gt_list, pred_list),
            "recall": recall_score(gt_list, pred_list),
            "hmean": f1_score(gt_list, pred_list),
        }
        return metrics

    def clear(self):
        self.pred_list = []
        self.gt_list = []


class VQAReTokenMetric(nn.Metric):
    """
    Metric method for Token RE task.
    """

    def __init__(self, device_num: int = 1, **kwargs):
        super().__init__()
        self.clear()
        self.device_num = device_num
        self.all_reduce = AllReduce(reduce="sum") if device_num > 1 else None
        self.metric_names = ["precision", "recall", "hmean"]

    def update(self, preds, gt):
        pred_relations = preds
        relations = gt[1]
        entities = gt[0]
        self.pred_relations_list.extend(pred_relations)
        self.relations_list.extend(relations)
        self.entities_list.extend(entities)

    def eval(self):
        gt_relations = []
        for b in range(len(self.relations_list)):
            rel_sent = []
            relation_list = self.relations_list[b]
            entitie_list = self.entities_list[b]
            head_len = relation_list[0, 0]
            if head_len > 0:
                entitie_start_list = entitie_list[1 : entitie_list[0, 0] + 1, 0]
                entitie_end_list = entitie_list[1 : entitie_list[0, 1] + 1, 1]
                entitie_label_list = entitie_list[1 : entitie_list[0, 2] + 1, 2]
                for head, tail in zip(relation_list[1 : head_len + 1, 0], relation_list[1 : head_len + 1, 1]):
                    rel = {}
                    rel["head_id"] = head
                    rel["head"] = (entitie_start_list[head], entitie_end_list[head])
                    rel["head_type"] = entitie_label_list[head]

                    rel["tail_id"] = tail
                    rel["tail"] = (entitie_start_list[tail], entitie_end_list[tail])
                    rel["tail_type"] = entitie_label_list[tail]

                    rel["type"] = 1
                    rel_sent.append(rel)
            gt_relations.append(rel_sent)
        re_metrics = self.re_score(self.pred_relations_list, gt_relations, mode="boundaries")
        metrics = {
            "precision": re_metrics["ALL"]["p"],
            "recall": re_metrics["ALL"]["r"],
            "hmean": re_metrics["ALL"]["f1"],
        }
        return metrics

    def clear(self):
        self.pred_relations_list = []
        self.relations_list = []
        self.entities_list = []

    def re_score(self, pred_relations, gt_relations, mode="strict"):
        """Evaluate RE predictions

        Args:
            pred_relations (list) :  list of list of predicted relations (several relations in each sentence)
            gt_relations (list) :    list of list of ground truth relations

                rel = { "head": (start_idx (inclusive), end_idx (exclusive)),
                        "tail": (start_idx (inclusive), end_idx (exclusive)),
                        "head_type": ent_type,
                        "tail_type": ent_type,
                        "type": rel_type}
            mode (str) :            in 'strict' or 'boundaries'"""

        assert mode in ["strict", "boundaries"]

        relation_types = [v for v in [0, 1] if not v == 0]
        scores = {rel: {"tp": 0, "fp": 0, "fn": 0} for rel in relation_types + ["ALL"]}
        # Count TP, FP and FN per type
        for pred_sent, gt_sent in zip(pred_relations, gt_relations):
            for rel_type in relation_types:
                # strict mode takes argument types into account
                if mode == "strict":
                    pred_rels = {
                        (rel["head"], rel["head_type"], rel["tail"], rel["tail_type"])
                        for rel in pred_sent
                        if rel["type"] == rel_type
                    }
                    gt_rels = {
                        (rel["head"], rel["head_type"], rel["tail"], rel["tail_type"])
                        for rel in gt_sent
                        if rel["type"] == rel_type
                    }

                # boundaries mode only takes argument spans into account
                elif mode == "boundaries":
                    pred_rels = {(rel["head"], rel["tail"]) for rel in pred_sent if rel["type"] == rel_type}
                    gt_rels = set()
                    for rel in gt_sent:
                        if rel["type"] == rel_type:
                            rel_head_start = int(rel["head"][0])
                            rel_head_end = int(rel["head"][1])
                            rel_tail_start = int(rel["tail"][0])
                            rel_tail_end = int(rel["tail"][1])
                            value = ((rel_head_start, rel_head_end), (rel_tail_start, rel_tail_end))
                            gt_rels.add(value)

                if self.all_reduce:
                    scores[rel_type]["tp"] += self.all_reduce(len(pred_rels & gt_rels))
                    scores[rel_type]["fp"] += self.all_reduce(len(pred_rels - gt_rels))
                    scores[rel_type]["fn"] += self.all_reduce(len(gt_rels - pred_rels))
                else:
                    scores[rel_type]["tp"] += len(pred_rels & gt_rels)
                    scores[rel_type]["fp"] += len(pred_rels - gt_rels)
                    scores[rel_type]["fn"] += len(gt_rels - pred_rels)

        # Compute per entity Precision / Recall / F1
        for rel_type in scores.keys():
            if scores[rel_type]["tp"]:
                scores[rel_type]["p"] = scores[rel_type]["tp"] / (scores[rel_type]["fp"] + scores[rel_type]["tp"])
                scores[rel_type]["r"] = scores[rel_type]["tp"] / (scores[rel_type]["fn"] + scores[rel_type]["tp"])
            else:
                scores[rel_type]["p"], scores[rel_type]["r"] = 0, 0

            if not scores[rel_type]["p"] + scores[rel_type]["r"] == 0:
                scores[rel_type]["f1"] = (
                    2 * scores[rel_type]["p"] * scores[rel_type]["r"] / (scores[rel_type]["p"] + scores[rel_type]["r"])
                )
            else:
                scores[rel_type]["f1"] = 0

        # Compute micro F1 Scores
        tp = sum([scores[rel_type]["tp"] for rel_type in relation_types])
        fp = sum([scores[rel_type]["fp"] for rel_type in relation_types])
        fn = sum([scores[rel_type]["fn"] for rel_type in relation_types])

        if tp:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * precision * recall / (precision + recall)

        else:
            precision, recall, f1 = 0, 0, 0

        scores["ALL"]["p"] = precision
        scores["ALL"]["r"] = recall
        scores["ALL"]["f1"] = f1
        scores["ALL"]["tp"] = tp
        scores["ALL"]["fp"] = fp
        scores["ALL"]["fn"] = fn

        # Compute Macro F1 Scores
        scores["ALL"]["Macro_f1"] = np.mean([scores[ent_type]["f1"] for ent_type in relation_types])
        scores["ALL"]["Macro_p"] = np.mean([scores[ent_type]["p"] for ent_type in relation_types])
        scores["ALL"]["Macro_r"] = np.mean([scores[ent_type]["r"] for ent_type in relation_types])

        return scores
