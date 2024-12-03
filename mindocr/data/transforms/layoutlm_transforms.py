import copy
import json
from collections import defaultdict

import cv2
import numpy as np

from mindspore import nn

from mindocr.models.backbones.layoutlmv3 import LayoutLMv3Tokenizer
from mindocr.models.backbones.layoutxlm import LayoutXLMTokenizer
from mindocr.utils.kie_utils import load_vqa_bio_label_maps


class LayoutResize:
    """
    Resize for Layout
    """

    def __init__(self, size=(640, 640), **kwargs):
        self.size = size

    def resize_image(self, img):
        resize_h, resize_w = self.size
        ori_h, ori_w = img.shape[:2]  # (h, w, c)
        ratio_h = float(resize_h) / ori_h
        ratio_w = float(resize_w) / ori_w
        img = cv2.resize(img, (int(resize_w), int(resize_h)))
        return img, [ratio_h, ratio_w]

    def __call__(self, data):
        img = data["image"]
        if "polys" in data:
            text_polys = data["polys"]

        img_resize, [ratio_h, ratio_w] = self.resize_image(img)
        if "polys" in data:
            new_boxes = []
            for box in text_polys:
                new_box = []
                for cord in box:
                    new_box.append([cord[0] * ratio_w, cord[1] * ratio_h])
                new_boxes.append(new_box)
            data["polys"] = np.array(new_boxes, dtype=np.float32)
        data["image"] = img_resize
        return data


class ImageStridePad:
    """
    image stride pad
    """

    def __init__(self, stride=32, max_size=None, **kwargs):
        self.stride = stride
        self.max_size = max_size

    def __call__(self, data):
        img = data["image"]
        image_size = img.shape[-2:]
        if self.max_size is None:
            max_size = np.array(image_size)
        else:
            max_size = np.array(self.max_size)

        max_size = (max_size + (self.stride - 1)) // self.stride * self.stride
        h_pad = int(max_size[0] - image_size[0])
        w_pad = int(max_size[1] - image_size[1])

        padding_size = ((0, 0), (0, h_pad), (0, w_pad))
        img = np.pad(img, padding_size, mode="constant", constant_values=0)
        data["image"] = img
        return data


class VQATokenLabelEncode:
    """
    Label encode for NLP VQA methods
    """

    def __init__(
        self,
        class_path,
        contains_re=False,
        add_special_ids=False,
        algorithm="LayoutXLM",
        use_textline_bbox_info=True,
        order_method=None,
        infer_mode=False,
        ocr_engine=None,
        **kwargs,
    ):
        super(VQATokenLabelEncode, self).__init__()
        tokenizer_dict = {
            "LayoutXLM": {"class": LayoutXLMTokenizer, "pretrained_model": "layoutxlm-base-uncased"},
            "LayoutLMv3": {"class": LayoutLMv3Tokenizer, "pretrained_model": "layoutxlm-base-uncased"},
        }
        self.contains_re = contains_re
        tokenizer_config = tokenizer_dict[algorithm]
        self.tokenizer = tokenizer_config["class"].from_pretrained(tokenizer_config["pretrained_model"])  # to replace
        self.label2id_map, id2label_map = load_vqa_bio_label_maps(class_path)
        self.add_special_ids = add_special_ids
        self.infer_mode = infer_mode
        self.ocr_engine = ocr_engine
        self.use_textline_bbox_info = use_textline_bbox_info
        self.order_method = order_method
        if self.order_method not in [None, "tb-yx"]:
            raise ValueError(
                f"The order_method of VQATokenLabelEncode must be None or 'tb-yx', but got {self.order_method}"
            )

    def split_bbox(self, bbox, text, tokenizer):
        words = text.split()
        token_bboxes = []
        x1, y1, x2, y2 = bbox
        unit_w = (x2 - x1) / len(text)
        for idx, word in enumerate(words):
            curr_w = len(word) * unit_w
            word_bbox = [x1, y1, x1 + curr_w, y2]
            token_bboxes.extend([word_bbox] * len(tokenizer.tokenize(word)))
            x1 += (len(word) + 1) * unit_w
        return token_bboxes

    @staticmethod
    def filter_empty_contents(ocr_info):
        """
        find out the empty texts and remove the links
        """
        new_ocr_info = []
        empty_index = []
        for idx, info in enumerate(ocr_info):
            if len(info["transcription"]) > 0:
                new_ocr_info.append(copy.deepcopy(info))
            else:
                empty_index.append(info["id"])

        for idx, info in enumerate(new_ocr_info):
            new_link = []
            for link in info["linking"]:
                if link[0] in empty_index or link[1] in empty_index:
                    continue
                new_link.append(link)
            new_ocr_info[idx]["linking"] = new_link
        return new_ocr_info

    def order_by_tbyx(self, ocr_info):
        res = sorted(ocr_info, key=lambda r: (r["bbox"][1], r["bbox"][0]))
        for i in range(len(res) - 1):
            for j in range(i, 0, -1):
                if abs(res[j + 1]["bbox"][1] - res[j]["bbox"][1]) < 20 and (res[j + 1]["bbox"][0] < res[j]["bbox"][0]):
                    tmp = copy.deepcopy(res[j])
                    res[j] = copy.deepcopy(res[j + 1])
                    res[j + 1] = copy.deepcopy(tmp)
                else:
                    break
        return res

    def __call__(self, data):
        # load bbox and label info
        ocr_info = self._load_ocr_info(data)

        for idx in range(len(ocr_info)):
            if "bbox" not in ocr_info[idx]:
                ocr_info[idx]["bbox"] = self.trans_poly_to_bbox(ocr_info[idx]["points"])

        if self.order_method == "tb-yx":
            ocr_info = self.order_by_tbyx(ocr_info)

        # for re
        train_re = self.contains_re and not self.infer_mode
        if train_re:
            ocr_info = self.filter_empty_contents(ocr_info)

        height, width, _ = data["image"].shape

        words_list = []
        bbox_list = []
        input_ids_list = []
        token_type_ids_list = []
        segment_offset_id = []
        gt_label_list = []

        entities = []

        if train_re:
            relations = []
            id2label = {}
            entity_id_to_index_map = {}
            empty_entity = set()

        data["ocr_info"] = copy.deepcopy(ocr_info)

        for info in ocr_info:
            text = info["transcription"]
            if len(text) <= 0:
                continue
            if train_re:
                # for re
                if len(text) == 0:
                    empty_entity.add(info["id"])
                    continue
                id2label[info["id"]] = info["label"]
                relations.extend([tuple(sorted(link)) for link in info["linking"]])
            # smooth_box
            info["bbox"] = self.trans_poly_to_bbox(info["points"])

            encode_res = self.tokenizer.encode(
                text, pad_to_max_seq_len=False, return_attention_mask=True, return_token_type_ids=True
            )

            if not self.add_special_ids:
                # TODO: use tok.all_special_ids to remove
                encode_res["input_ids"] = encode_res["input_ids"][1:-1]
                encode_res["token_type_ids"] = encode_res["token_type_ids"][1:-1]
                encode_res["attention_mask"] = encode_res["attention_mask"][1:-1]

            if self.use_textline_bbox_info:
                bbox = [info["bbox"]] * len(encode_res["input_ids"])
            else:
                bbox = self.split_bbox(info["bbox"], info["transcription"], self.tokenizer)
            if len(bbox) <= 0:
                continue
            bbox = self._smooth_box(bbox, height, width)
            if self.add_special_ids:
                bbox.insert(0, [0, 0, 0, 0])
                bbox.append([0, 0, 0, 0])

            # parse label
            if not self.infer_mode:
                label = info["label"]
                gt_label = self._parse_label(label, encode_res)

            # construct entities for re
            if train_re:
                if gt_label[0] != self.label2id_map["O"]:
                    entity_id_to_index_map[info["id"]] = len(entities)
                    label = label.upper()
                    entities.append(
                        {
                            "start": len(input_ids_list),
                            "end": len(input_ids_list) + len(encode_res["input_ids"]),
                            "label": label.upper(),
                        }
                    )
            else:
                entities.append(
                    {
                        "start": len(input_ids_list),
                        "end": len(input_ids_list) + len(encode_res["input_ids"]),
                        "label": "O",
                    }
                )
            input_ids_list.extend(encode_res["input_ids"])
            token_type_ids_list.extend(encode_res["token_type_ids"])
            bbox_list.extend(bbox)
            words_list.append(text)
            segment_offset_id.append(len(input_ids_list))
            if not self.infer_mode:
                gt_label_list.extend(gt_label)

        data["input_ids"] = input_ids_list
        data["token_type_ids"] = token_type_ids_list
        data["bbox"] = bbox_list
        data["attention_mask"] = [1] * len(input_ids_list)
        data["labels"] = gt_label_list
        data["segment_offset_id"] = segment_offset_id
        data["tokenizer_params"] = dict(
            padding_side=self.tokenizer.padding_side,
            pad_token_type_id=self.tokenizer.pad_token_type_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        data["entities"] = entities

        if train_re:
            data["relations"] = relations
            data["id2label"] = id2label
            data["empty_entity"] = empty_entity
            data["entity_id_to_index_map"] = entity_id_to_index_map
        return data

    @staticmethod
    def trans_poly_to_bbox(poly):
        x1 = int(np.min([p[0] for p in poly]))
        x2 = int(np.max([p[0] for p in poly]))
        y1 = int(np.min([p[1] for p in poly]))
        y2 = int(np.max([p[1] for p in poly]))
        return [x1, y1, x2, y2]

    def _load_ocr_info(self, data):
        """read text info from 'label' data"""
        info = data["label"]
        info_dict = json.loads(info)
        return info_dict

    @staticmethod
    def _smooth_box(bboxes, height, width):
        bboxes = np.array(bboxes)
        bboxes[:, 0] = bboxes[:, 0] * 1000 / width
        bboxes[:, 2] = bboxes[:, 2] * 1000 / width
        bboxes[:, 1] = bboxes[:, 1] * 1000 / height
        bboxes[:, 3] = bboxes[:, 3] * 1000 / height
        bboxes = bboxes.astype("int64").tolist()
        return bboxes

    def _parse_label(self, label, encode_res):
        gt_label = []
        if label.lower() in ["other", "others", "ignore"]:
            gt_label.extend([0] * len(encode_res["input_ids"]))
        else:
            gt_label.append(self.label2id_map[("b-" + label).upper()])
            gt_label.extend([self.label2id_map[("i-" + label).upper()]] * (len(encode_res["input_ids"]) - 1))
        return gt_label


class VQATokenPad:
    def __init__(
        self,
        max_seq_len=512,
        pad_to_max_seq_len=True,
        return_attention_mask=True,
        return_token_type_ids=True,
        truncation_strategy="longest_first",
        return_overflowing_tokens=False,
        return_special_tokens_mask=False,
        infer_mode=False,
        **kwargs,
    ):
        self.max_seq_len = max_seq_len
        self.pad_to_max_seq_len = max_seq_len
        self.return_attention_mask = return_attention_mask
        self.return_token_type_ids = return_token_type_ids
        self.truncation_strategy = truncation_strategy
        self.return_overflowing_tokens = return_overflowing_tokens
        self.return_special_tokens_mask = return_special_tokens_mask
        self.pad_token_label_id = nn.CrossEntropyLoss().ignore_index
        self.infer_mode = infer_mode

    def __call__(self, data):
        needs_to_be_padded = self.pad_to_max_seq_len and len(data["input_ids"]) < self.max_seq_len

        if needs_to_be_padded:
            if "tokenizer_params" in data:
                tokenizer_params = data.pop("tokenizer_params")
            else:
                tokenizer_params = dict(padding_side="right", pad_token_type_id=0, pad_token_id=1)

            difference = self.max_seq_len - len(data["input_ids"])
            if tokenizer_params["padding_side"] == "right":
                if self.return_attention_mask:
                    data["attention_mask"] = [1] * len(data["input_ids"]) + [0] * difference
                if self.return_token_type_ids:
                    data["token_type_ids"] = (
                        data["token_type_ids"] + [tokenizer_params["pad_token_type_id"]] * difference
                    )
                if self.return_special_tokens_mask:
                    data["special_tokens_mask"] = data["special_tokens_mask"] + [1] * difference
                data["input_ids"] = data["input_ids"] + [tokenizer_params["pad_token_id"]] * difference
                if not self.infer_mode:
                    data["labels"] = data["labels"] + [self.pad_token_label_id] * difference
                data["bbox"] = data["bbox"] + [[0, 0, 0, 0]] * difference
            elif tokenizer_params["padding_side"] == "left":
                if self.return_attention_mask:
                    data["attention_mask"] = [0] * difference + [1] * len(data["input_ids"])
                if self.return_token_type_ids:
                    data["token_type_ids"] = [tokenizer_params["pad_token_type_id"]] * difference + data[
                        "token_type_ids"
                    ]
                if self.return_special_tokens_mask:
                    data["special_tokens_mask"] = [1] * difference + data["special_tokens_mask"]
                data["input_ids"] = [tokenizer_params["pad_token_id"]] * difference + data["input_ids"]
                if not self.infer_mode:
                    data["labels"] = [self.pad_token_label_id] * difference + data["labels"]
                data["bbox"] = [[0, 0, 0, 0]] * difference + data["bbox"]
        else:
            if self.return_attention_mask:
                data["attention_mask"] = [1] * len(data["input_ids"])

        for key in data:
            if key in ["input_ids", "labels", "token_type_ids", "bbox", "attention_mask"]:
                if self.infer_mode:
                    if key != "labels":
                        length = min(len(data[key]), self.max_seq_len)
                        data[key] = data[key][:length]
                    else:
                        continue
                data[key] = np.array(data[key], dtype="int64")
        return data


class VQASerTokenChunk:
    def __init__(self, max_seq_len=512, infer_mode=False, **kwargs):
        self.max_seq_len = max_seq_len
        self.infer_mode = infer_mode

    def __call__(self, data):
        encoded_inputs_all = []
        seq_len = len(data["input_ids"])
        for index in range(0, seq_len, self.max_seq_len):
            chunk_beg = index
            chunk_end = min(index + self.max_seq_len, seq_len)
            encoded_inputs_example = {}
            for key in data:
                if key in ["label", "input_ids", "labels", "token_type_ids", "bbox", "attention_mask"]:
                    if self.infer_mode and key == "labels":
                        encoded_inputs_example[key] = data[key]
                    else:
                        encoded_inputs_example[key] = data[key][chunk_beg:chunk_end]
                else:
                    encoded_inputs_example[key] = data[key]

            encoded_inputs_all.append(encoded_inputs_example)
        if len(encoded_inputs_all) == 0:
            return None
        return encoded_inputs_all[0]


class VQAReTokenRelation:
    def __init__(self, **kwargs):
        pass

    def __call__(self, data):
        """
        build relations
        """
        entities = data["entities"]
        relations = data["relations"]
        id2label = data.pop("id2label")
        empty_entity = data.pop("empty_entity")
        entity_id_to_index_map = data.pop("entity_id_to_index_map")

        relations = list(set(relations))
        relations = [rel for rel in relations if rel[0] not in empty_entity and rel[1] not in empty_entity]
        kv_relations = []
        for rel in relations:
            pair = [id2label[rel[0]], id2label[rel[1]]]
            if pair == ["question", "answer"]:
                kv_relations.append({"head": entity_id_to_index_map[rel[0]], "tail": entity_id_to_index_map[rel[1]]})
            elif pair == ["answer", "question"]:
                kv_relations.append({"head": entity_id_to_index_map[rel[1]], "tail": entity_id_to_index_map[rel[0]]})
            else:
                continue
        relations = sorted(
            [
                {
                    "head": rel["head"],
                    "tail": rel["tail"],
                    "start_index": self.get_relation_span(rel, entities)[0],
                    "end_index": self.get_relation_span(rel, entities)[1],
                }
                for rel in kv_relations
            ],
            key=lambda x: x["head"],
        )

        data["relations"] = relations
        return data

    def get_relation_span(self, rel, entities):
        bound = []
        for entity_index in [rel["head"], rel["tail"]]:
            bound.append(entities[entity_index]["start"])
            bound.append(entities[entity_index]["end"])
        return min(bound), max(bound)


class VQAReTokenChunk:
    def __init__(self, max_seq_len=512, entities_labels=None, infer_mode=False, **kwargs):
        self.max_seq_len = max_seq_len
        self.entities_labels = {"HEADER": 0, "QUESTION": 1, "ANSWER": 2} if entities_labels is None else entities_labels
        self.infer_mode = infer_mode

    def __call__(self, data):
        # prepare data
        entities = data.pop("entities")
        relations = data.pop("relations")
        encoded_inputs_all = []
        for index in range(0, len(data["input_ids"]), self.max_seq_len):
            item = {}
            for key in data:
                if key in ["label", "input_ids", "labels", "token_type_ids", "bbox", "attention_mask"]:
                    if self.infer_mode and key == "labels":
                        item[key] = data[key]
                    else:
                        item[key] = data[key][index : index + self.max_seq_len]
                else:
                    item[key] = data[key]
            # select entity in current chunk
            entities_in_this_span = []
            global_to_local_map = {}  #
            for entity_id, entity in enumerate(entities):
                if (
                    index <= entity["start"] < index + self.max_seq_len
                    and index <= entity["end"] < index + self.max_seq_len
                ):
                    entity["start"] = entity["start"] - index
                    entity["end"] = entity["end"] - index
                    global_to_local_map[entity_id] = len(entities_in_this_span)
                    entities_in_this_span.append(entity)

            # select relations in current chunk
            relations_in_this_span = []
            for relation in relations:
                if (
                    index <= relation["start_index"] < index + self.max_seq_len
                    and index <= relation["end_index"] < index + self.max_seq_len
                ):
                    relations_in_this_span.append(
                        {
                            "head": global_to_local_map[relation["head"]],
                            "tail": global_to_local_map[relation["tail"]],
                            "start_index": relation["start_index"] - index,
                            "end_index": relation["end_index"] - index,
                        }
                    )
            item.update(
                {
                    "entities": self.reformat(entities_in_this_span),
                    "relations": self.reformat(relations_in_this_span),
                }
            )
            if len(item["entities"]) > 0:
                item["entities"]["label"] = [self.entities_labels[x] for x in item["entities"]["label"]]
                encoded_inputs_all.append(item)
        if len(encoded_inputs_all) == 0:
            return None
        return encoded_inputs_all[0]

    def reformat(self, data):
        new_data = defaultdict(list)
        for item in data:
            for k, v in item.items():
                new_data[k].append(v)
        return new_data


class TensorizeEntitiesRelations:
    def __init__(self, max_seq_len=512, infer_mode=False, max_relation_len=None, **kwargs):
        self.max_seq_len = max_seq_len
        self.infer_mode = infer_mode
        self.max_relation_len = max_relation_len

    def build_relation(self, relations, entities):
        max_seq_len, _ = entities.shape
        num_max_relation = (max_seq_len - 1) * (max_seq_len - 1) // 4
        if self.max_relation_len is not None:
            num_max_relation = min(num_max_relation, self.max_relation_len)

        q_id = np.full(
            shape=[
                num_max_relation,
            ],
            fill_value=-1,
            dtype=np.int32,
        )

        q_labels = np.zeros(
            shape=[
                num_max_relation,
            ],
            dtype=np.int32,
        )

        a_id = np.full(
            shape=[
                num_max_relation,
            ],
            fill_value=-1,
            dtype=np.int32,
        )

        a_labels = np.zeros(
            shape=[
                num_max_relation,
            ],
            dtype=np.int32,
        )

        realation_label = np.full(
            shape=[
                num_max_relation,
            ],
            fill_value=-100,
            dtype=np.int32,
        )

        if entities[0, 0] <= 2:
            entitie_new = -np.ones_like(entities)
            entitie_new[0, :] = 2
            entitie_new[1:3, 0] = 0  # start
            entitie_new[1:3, 1] = 1  # end
            entitie_new[1:3, 2] = 0  # label
            entities = entitie_new
        entitie_label = entities[1 : entities[0, 2] + 1, 2]
        all_possible_relations1 = np.arange(0, entities[0, 2], dtype=entities.dtype)
        all_possible_relations1 = all_possible_relations1[entitie_label == 1]
        all_possible_relations2 = np.arange(0, entities[0, 2], dtype=entities.dtype)
        all_possible_relations2 = all_possible_relations2[entitie_label == 2]

        a, q = np.meshgrid(all_possible_relations2, all_possible_relations1)
        all_possible_relations = np.stack([q, a], axis=2).reshape((-1, 2))

        if len(all_possible_relations) == 0:
            all_possible_relations = np.full_like(all_possible_relations, fill_value=-1, dtype=entities.dtype)
            all_possible_relations[0, 0] = 0
            all_possible_relations[0, 1] = 1

        relation_head = relations[1 : relations[0, 0] + 1, 0]
        relation_tail = relations[1 : relations[0, 1] + 1, 1]
        positive_relations = np.stack([relation_head, relation_tail], axis=1)

        all_possible_relations_repeat = np.expand_dims(all_possible_relations, axis=1).repeat(
            len(positive_relations), axis=1
        )
        positive_relations_repeat = np.expand_dims(positive_relations, axis=0).repeat(
            len(all_possible_relations), axis=0
        )
        mask = np.all(all_possible_relations_repeat == positive_relations_repeat, axis=2)
        negative_mask = np.any(mask, axis=1) == False  # noqa
        negative_relations = all_possible_relations[negative_mask]

        positive_mask = np.any(mask, axis=0) == True  # noqa
        positive_relations = positive_relations[positive_mask]
        if negative_mask.sum() > 0:
            reordered_relations = np.concatenate([positive_relations, negative_relations])

        else:
            reordered_relations = positive_relations
        reordered_relations = reordered_relations[:num_max_relation, :]
        num_recorded_relations = reordered_relations.shape[0]
        q_index = reordered_relations[:, 0]
        entites_id_list = entities[1 : entities[0, 0] + 1, 0]
        recorded_q_id = entites_id_list[q_index]
        recorded_q_labels = entitie_label[q_index]
        q_id[:num_recorded_relations] = recorded_q_id
        q_labels[:num_recorded_relations] = recorded_q_labels

        a_index = reordered_relations[:, 1]
        recorded_a_id = entites_id_list[a_index]
        recorded_a_labels = entitie_label[a_index]
        a_id[:num_recorded_relations] = recorded_a_id
        a_labels[:num_recorded_relations] = recorded_a_labels

        qa_label = np.zeros((reordered_relations.shape[0],), dtype=reordered_relations.dtype)
        qa_label[: positive_relations.shape[0]] = 1
        realation_label[: qa_label.shape[0]] = qa_label
        return q_id, q_labels, a_id, a_labels, realation_label

    def __call__(self, data):
        entities = data["entities"]
        relations = data["relations"]

        entities_new = np.full(shape=[self.max_seq_len + 1, 3], fill_value=-1, dtype="int64")
        entities_new[0, 0] = len(entities["start"])
        entities_new[0, 1] = len(entities["end"])
        entities_new[0, 2] = len(entities["label"])
        entities_new[1 : len(entities["start"]) + 1, 0] = np.array(entities["start"])
        entities_new[1 : len(entities["end"]) + 1, 1] = np.array(entities["end"])
        entities_new[1 : len(entities["label"]) + 1, 2] = np.array(entities["label"])

        relations_new = np.full(shape=[self.max_seq_len * self.max_seq_len + 1, 2], fill_value=-1, dtype="int64")
        relations_new[0, 0] = len(relations["head"])
        relations_new[0, 1] = len(relations["tail"])
        relations_new[1 : len(relations["head"]) + 1, 0] = np.array(relations["head"])
        relations_new[1 : len(relations["tail"]) + 1, 1] = np.array(relations["tail"])

        q_id, q_labels, a_id, a_labels, realation_label = self.build_relation(relations_new, entities_new)
        data["question"] = q_id
        data["question_label"] = q_labels
        data["answer"] = a_id
        data["answer_label"] = a_labels
        data["relation_label"] = realation_label
        return data
