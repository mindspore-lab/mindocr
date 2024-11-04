import copy
import glob
import math
import os
import pickle
import re

import numpy as np
from shapely.geometry import MultiPoint, Polygon


def remove_empty_bboxes(bboxes):
    """
    Remove empty bboxes.
    """
    return np.array([bbox for bbox in bboxes if np.sum(bbox) != 0.0])


def xywh2xyxy(bboxes):
    """
    Convert bboxes from xywh to xyxy format.
    """
    if len(bboxes.shape) == 1:
        new_bboxes = np.empty_like(bboxes)
        new_bboxes[0] = bboxes[0] - bboxes[2] / 2
        new_bboxes[1] = bboxes[1] - bboxes[3] / 2
        new_bboxes[2] = bboxes[0] + bboxes[2] / 2
        new_bboxes[3] = bboxes[1] + bboxes[3] / 2
        return new_bboxes
    if len(bboxes.shape) == 2:
        new_bboxes = np.empty_like(bboxes)
        new_bboxes[:, 0] = bboxes[:, 0] - bboxes[:, 2] / 2
        new_bboxes[:, 1] = bboxes[:, 1] - bboxes[:, 3] / 2
        new_bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2] / 2
        new_bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3] / 2
        return new_bboxes
    raise ValueError


def xyxy2xywh(bboxes):
    """
    Convert bboxes from xyxy to xywh format.
    """
    if len(bboxes.shape) == 1:
        new_bboxes = np.empty_like(bboxes)
        new_bboxes[0] = bboxes[0] + (bboxes[2] - bboxes[0]) / 2
        new_bboxes[1] = bboxes[1] + (bboxes[3] - bboxes[1]) / 2
        new_bboxes[2] = bboxes[2] - bboxes[0]
        new_bboxes[3] = bboxes[3] - bboxes[1]
        return new_bboxes
    if len(bboxes.shape) == 2:
        new_bboxes = np.empty_like(bboxes)
        new_bboxes[:, 0] = bboxes[:, 0] + (bboxes[:, 2] - bboxes[:, 0]) / 2
        new_bboxes[:, 1] = bboxes[:, 1] + (bboxes[:, 3] - bboxes[:, 1]) / 2
        new_bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
        new_bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
        return new_bboxes
    raise ValueError


def pickle_load(path, prefix="end2end"):
    if os.path.isfile(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    if os.path.isdir(path):
        data = {}
        search_path = os.path.join(path, f"{prefix}_*.pkl")
        for pkl in glob.glob(search_path):
            with open(pkl, "rb") as f:
                data.update(pickle.load(f))
        return data
    raise ValueError(f"Path '{path}' is neither a file nor a directory.")


def convert_coord(xyxy):
    new_bbox = np.zeros([4, 2], dtype=np.float32)
    new_bbox[0, 0], new_bbox[0, 1] = xyxy[0], xyxy[1]
    new_bbox[1, 0], new_bbox[1, 1] = xyxy[2], xyxy[1]
    new_bbox[2, 0], new_bbox[2, 1] = xyxy[2], xyxy[3]
    new_bbox[3, 0], new_bbox[3, 1] = xyxy[0], xyxy[3]
    return new_bbox


def cal_iou(bbox1, bbox2):
    bbox1_poly = Polygon(bbox1).convex_hull
    bbox2_poly = Polygon(bbox2).convex_hull
    union_poly = np.concatenate((bbox1, bbox2))

    if not bbox1_poly.intersects(bbox2_poly):
        return 0.0
    inter_area = bbox1_poly.intersection(bbox2_poly).area
    union_area = MultiPoint(union_poly).convex_hull.area
    if union_area == 0:
        return 0.0
    return inter_area / union_area


def cal_distance(p1, p2):
    delta_x = p1[0] - p2[0]
    delta_y = p1[1] - p2[1]
    return math.sqrt((delta_x**2) + (delta_y**2))


def is_inside(center_point, corner_point):
    """
    Check if center_point inside the bbox(corner_point) or not.
    Args:
        center_point (tuple): Center point (x, y).
        corner_point (tuple): Bounding box corners ((x1, y1), (x2, y2)).
    Returns:
        bool: True if the point is inside the bounding box, False otherwise.
    """
    return (
        corner_point[0][0] <= center_point[0] <= corner_point[1][0]
        and corner_point[0][1] <= center_point[1] <= corner_point[1][1]
    )


def find_no_match(match_list, all_end2end_nums, match_type="end2end"):
    """
    Find out no match end2end bbox in previous match list.
    Args:
        match_list (List[Tuple[int, int]]): List of matching pairs.
        all_end2end_nums (int): Total number of end2end_xywh.
        match_type (str): 'end2end' corresponding to idx 0, 'master' corresponding to idx 1.
    Returns:
        List[int]: List of indices with no match.
    """
    idx_map = {"end2end": 0, "master": 1}
    if match_type not in idx_map:
        raise ValueError("Invalid type. Expected 'end2end' or 'master'.")
    idx = idx_map[match_type]

    matched_bbox_indices = {m[idx] for m in match_list}
    return [n for n in range(all_end2end_nums) if n not in matched_bbox_indices]


def is_abs_lower_than_threshold(this_bbox, target_bbox, threshold=3):
    # Only consider y axis, for grouping in row.
    return abs(this_bbox[1] - target_bbox[1]) < threshold


def sort_line_bbox(g, bg):
    """
    Sorted the bbox in the same line(group)
    compare coord 'x' value, where 'y' value is closed in the same group.
    """
    combined = sorted(zip(g, bg), key=lambda item: item[1][0])
    g_sorted, bg_sorted = zip(*combined)
    return list(g_sorted), list(bg_sorted)


def flatten(sorted_groups, sorted_bbox_groups):
    idxs = []
    bboxes = []
    for group, bbox_group in zip(sorted_groups, sorted_bbox_groups):
        for g, bg in zip(group, bbox_group):
            idxs.append(g)
            bboxes.append(bg)
    return idxs, bboxes


def sort_bbox(end2end_xywh_bboxes, no_match_end2end_indexes):
    """
    This function will group the render end2end bboxes in row.
    """
    groups = []
    bbox_groups = []
    for index, this_bbox in zip(no_match_end2end_indexes, end2end_xywh_bboxes):
        for g, bg in zip(groups, bbox_groups):
            if is_abs_lower_than_threshold(this_bbox, bg[0]):
                g.append(index)
                bg.append(this_bbox)
                break
        else:
            groups.append([index])
            bbox_groups.append([this_bbox])
    # sorted bboxes in a group
    sorted_groups_bbox_pairs = sorted(
        (sort_line_bbox(g, bg) for g, bg in zip(groups, bbox_groups)), key=lambda x: x[1][0][1]
    )
    sorted_groups, sorted_bbox_groups = zip(*sorted_groups_bbox_pairs)
    # flatten, get final result
    end2end_sorted_idx_list, end2end_sorted_bbox_list = flatten(sorted_groups, sorted_bbox_groups)
    return (
        end2end_sorted_idx_list,
        end2end_sorted_bbox_list,
        list(sorted_groups),
        list(sorted_bbox_groups),
    )


def get_bboxes_list(end2end_result, structure_master_result):
    """
    Get bbox(xyxy and xywh) list from end2end and structure master result.
    """
    end2end_xyxy_bboxes = []
    end2end_xywh_bboxes = []
    for item in end2end_result:
        bbox = item["bbox"]
        end2end_xyxy_bboxes.append(bbox)
        end2end_xywh_bboxes.append(xyxy2xywh(bbox))
    end2end_xyxy_bboxes = np.array(end2end_xyxy_bboxes)
    end2end_xywh_bboxes = np.array(end2end_xywh_bboxes)

    structure_master_xyxy_bboxes = remove_empty_bboxes(structure_master_result["bbox"])
    structure_master_xywh_bboxes = xyxy2xywh(structure_master_xyxy_bboxes)

    return (
        end2end_xyxy_bboxes,
        end2end_xywh_bboxes,
        structure_master_xywh_bboxes,
        structure_master_xyxy_bboxes,
    )


def center_rule_match(end2end_xywh_bboxes, structure_master_xyxy_bboxes):
    """
    Judge end2end Bbox's center point is inside structure master Bbox or not, if yes, return the match pairs.
    """
    match_pairs_list = []
    for i, end2end_xywh in enumerate(end2end_xywh_bboxes):
        center_point_end2end = (end2end_xywh[0], end2end_xywh[1])
        for j, master_xyxy in enumerate(structure_master_xyxy_bboxes):
            corner_point_master = ((master_xyxy[0], master_xyxy[1]), (master_xyxy[2], master_xyxy[3]))
            if is_inside(center_point_end2end, corner_point_master):
                match_pairs_list.append([i, j])
    return match_pairs_list


def iou_rule_match(end2end_xyxy_bboxes, end2end_xyxy_indexes, structure_master_xyxy_bboxes):
    """
    Use iou to find matching list, choose max iou value bbox as match pair.
    """
    match_pair_list = []
    for end2end_index, end2end_bbox in zip(end2end_xyxy_indexes, end2end_xyxy_bboxes):
        max_iou = 0
        max_match = None
        end2end_4xy = convert_coord(end2end_bbox)
        for j, master_bbox in enumerate(structure_master_xyxy_bboxes):
            master_4xy = convert_coord(master_bbox)
            iou = cal_iou(end2end_4xy, master_4xy)
            if iou > max_iou:
                max_match = [end2end_index, j]
                max_iou = iou

        if max_match:
            match_pair_list.append(max_match)
    return match_pair_list


def distance_rule_match(end2end_indexes, end2end_bboxes, master_indexes, master_bboxes):
    """
    Match end2end bounding boxes with master bounding boxes based on the minimum distance.
    """
    min_match_list = []
    for j, master_bbox in zip(master_indexes, master_bboxes):
        min_distance = np.inf
        min_match = [0, 0]
        master_point = (master_bbox[0], master_bbox[1])
        for i, end2end_bbox in zip(end2end_indexes, end2end_bboxes):
            end2end_point = (end2end_bbox[0], end2end_bbox[1])
            dist = cal_distance(master_point, end2end_point)
            if dist < min_distance:
                min_match = [i, j]
                min_distance = dist
        min_match_list.append(min_match)
    return min_match_list


def extra_match(no_match_end2end_indexes, master_bbox_nums):
    """
    Create virtual master bboxes and match them with the no match end2end indexes.
    """
    return [[no_match_end2end_indexes[i], i + master_bbox_nums] for i in range(len(no_match_end2end_indexes))]


def get_match_dict(match_list):
    """
    Convert match_list to a dict, where key is master bbox's index, value is end2end bbox index.
    """
    match_dict = {}
    for end2end_index, master_index in match_list:
        match_dict.setdefault(master_index, []).append(end2end_index)
    return match_dict


def reduce_repeat_bb(text_list, break_token):
    """
    convert ['<b>Local</b>', '<b>government</b>', '<b>unit</b>'] to ['<b>Local government unit</b>']
    """
    if all(text.startswith("<b>") and text.endswith("</b>") for text in text_list):
        new_text_list = [text[3:-4] for text in text_list]
        return [f"<b>{break_token.join(new_text_list)}</b>"]
    return text_list


def get_match_text_dict(match_dict, end2end_info, break_token=" "):
    match_text_dict = {}
    for master_index, end2end_index_list in match_dict.items():
        text_list = [end2end_info[end2end_index]["text"] for end2end_index in end2end_index_list]
        text_list = reduce_repeat_bb(text_list, break_token)
        text = break_token.join(text_list)
        match_text_dict[master_index] = text
    return match_text_dict


def merge_span_token(master_token_list):
    """
    Merge the span style token (row span or col span).
    """
    new_master_token_list = []
    pointer = 0
    if master_token_list[-1] != "</tbody>":
        master_token_list.append("</tbody>")
    while pointer < len(master_token_list) and master_token_list[pointer] != "</tbody>":
        try:
            if master_token_list[pointer] == "<td":
                if any(master_token_list[pointer + 1].startswith(attr) for attr in [" colspan=", " rowspan="]):
                    # pattern <td colspan="3">, '<td' + 'colspan=" "' + '>' + '</td>'
                    tmp = "".join(master_token_list[pointer : pointer + 4])
                    pointer += 4
                elif any(master_token_list[pointer + 2].startswith(attr) for attr in [" colspan=", " rowspan="]):
                    # pattern <td rowspan="2" colspan="3">, '<td' + 'rowspan=" "' + 'colspan=" "' + '>' + '</td>'
                    tmp = "".join(master_token_list[pointer : pointer + 5])
                    pointer += 5
                else:
                    tmp = master_token_list[pointer]
                    pointer += 1
                new_master_token_list.append(tmp)
            else:
                new_master_token_list.append(master_token_list[pointer])
                pointer += 1
        except IndexError:
            print("Break in merge due to IndexError...")
            break
    new_master_token_list.append("</tbody>")
    return new_master_token_list


def deal_eb_token(master_token):
    replacements = {
        "<eb></eb>": "<td></td>",
        "<eb1></eb1>": "<td> </td>",
        "<eb2></eb2>": "<td><b> </b></td>",
        "<eb3></eb3>": "<td>\u2028\u2028</td>",
        "<eb4></eb4>": "<td><sup> </sup></td>",
        "<eb5></eb5>": "<td><b></b></td>",
        "<eb6></eb6>": "<td><i> </i></td>",
        "<eb7></eb7>": "<td><b><i></i></b></td>",
        "<eb8></eb8>": "<td><b><i> </i></b></td>",
        "<eb9></eb9>": "<td><i></i></td>",
        "<eb10></eb10>": "<td><b> \u2028 \u2028 </b></td>",
    }
    for old, new in replacements.items():
        master_token = master_token.replace(old, new)
    return master_token


def insert_text_to_token(master_token_list, match_text_dict):
    """
    Insert OCR text result to structure token.
    """
    master_token_list = merge_span_token(master_token_list)
    merged_result_list = []
    text_count = 0
    for master_token in master_token_list:
        if master_token.startswith("<td"):
            if text_count in match_text_dict:
                master_token = master_token.replace("><", f">{match_text_dict[text_count]}<")
            text_count += 1
        master_token = deal_eb_token(master_token)
        merged_result_list.append(master_token)
    return "".join(merged_result_list)


def deal_isolate_span(thead_part):
    """
    Deal with isolate span cases caused by wrong predictions in the structure recognition model.
    """
    # 1. Find out isolate span tokens.
    isolate_pattern = re.compile(
        r'<td></td> (rowspan="\d+" colspan="\d+"|colspan="\d+" rowspan="\d+"|rowspan="\d+"|colspan="\d+")></b></td>'
    )
    isolate_list = isolate_pattern.findall(thead_part)
    # 2. Correct the isolated span tokens.
    corrected_list = [f"<td {span}></td>" for span in isolate_list]
    # 3. Replace original isolated tokens with corrected tokens.
    thead_part = isolate_pattern.sub(lambda _: corrected_list.pop(0), thead_part)
    return thead_part


def deal_duplicate_bb(thead_part):
    """
    Deal with duplicate <b></b> tags within <td></td> tags in the <thead> part of an HTML table.
    """
    # 1. Find all <td></td> tags in <thead></thead>.
    td_pattern = re.compile(r'<td(?: rowspan="\d+")?(?: colspan="\d+")?>.*?</td>')
    td_list = td_pattern.findall(thead_part)

    # 2. Check for multiple <b></b> tags within <td></td> and correct them.
    new_td_list = []
    for td_item in td_list:
        if td_item.count("<b>") > 1 or td_item.count("</b>") > 1:
            # Remove all <b></b> tags and reapply them correctly.
            td_item = td_item.replace("<b>", "").replace("</b>", "")
            td_item = td_item.replace("<td>", "<td><b>").replace("</td>", "</b></td>")
        new_td_list.append(td_item)

    # 3. Replace original <td></td> tags with corrected ones.
    for td_item, new_td_item in zip(td_list, new_td_list):
        thead_part = thead_part.replace(td_item, new_td_item)
    return thead_part


def deal_bb(result_token):
    """
    Find out all tokens in <thead></thead> and insert <b></b> manually.
    """
    # Find out <thead></thead> parts.
    thead_pattern = re.compile(r"<thead>(.*?)</thead>")
    match = thead_pattern.search(result_token)
    if not match:
        return result_token
    thead_part = match.group()
    origin_thead_part = copy.deepcopy(thead_part)

    # Check if "rowspan" or "colspan" occur in <thead></thead> parts.
    span_pattern = re.compile(
        r'<td (?:rowspan="\d+" colspan="\d+"|colspan="\d+" rowspan="\d+"|rowspan="\d+"|colspan="\d+")>'
    )
    span_list = span_pattern.findall(thead_part)
    has_span_in_head = bool(span_list)

    if not has_span_in_head:
        # <thead></thead> not include "rowspan" or "colspan".
        thead_part = (
            thead_part.replace("<td>", "<td><b>")
            .replace("</td>", "</b></td>")
            .replace("<b><b>", "<b>")
            .replace("</b></b>", "</b>")
        )
    else:
        # <thead></thead> include "rowspan" or "colspan".
        # Replace ">" with "><b>" and "</td>" with "</b></td>"
        for sp in span_list:
            thead_part = thead_part.replace(sp, sp.replace(">", "><b>"))
        thead_part = thead_part.replace("</td>", "</b></td>")

        # Remove duplicated <b> and </b> tags
        thead_part = re.sub(r"(<b>)+", "<b>", thead_part)
        thead_part = re.sub(r"(</b>)+", "</b>", thead_part)

        # Handle ordinary cases
        thead_part = thead_part.replace("<td>", "<td><b>").replace("<b><b>", "<b>")

    thead_part = thead_part.replace("<td><b></b></td>", "<td></td>")
    # deal with duplicated <b></b>
    thead_part = deal_duplicate_bb(thead_part)
    thead_part = deal_isolate_span(thead_part)
    # replace original result with new thead part.
    result_token = result_token.replace(origin_thead_part, thead_part)
    return result_token


class Matcher:
    def __init__(self, end2end_file, structure_master_file):
        """
        This class process the end2end results and structure recognition results.
        """
        self.end2end_file = end2end_file
        self.structure_master_file = structure_master_file
        self.end2end_results = pickle_load(end2end_file, prefix="end2end")
        self.structure_master_results = pickle_load(structure_master_file, prefix="structure")

    def match(self):
        """
        Match process:
        pre-process : convert end2end and structure master results to xyxy, xywh ndnarray format.
        1. Use pseBbox is inside masterBbox judge rule
        2. Use iou between pseBbox and masterBbox rule
        3. Use min distance of center point rule
        """
        match_results = {}
        for file_name, end2end_result in self.end2end_results.items():
            if file_name not in self.structure_master_results:
                continue
            structure_master_result = self.structure_master_results[file_name]
            (
                end2end_xyxy_bboxes,
                end2end_xywh_bboxes,
                structure_master_xywh_bboxes,
                structure_master_xyxy_bboxes,
            ) = get_bboxes_list(end2end_result, structure_master_result)

            match_list = self._apply_match_rule(
                end2end_xywh_bboxes, end2end_xyxy_bboxes, structure_master_xywh_bboxes, structure_master_xyxy_bboxes
            )
            no_match_end2end_indexes = find_no_match(match_list, len(end2end_xywh_bboxes), "end2end")

            if no_match_end2end_indexes:
                no_match_end2end_xywh = end2end_xywh_bboxes[no_match_end2end_indexes]
                (end2end_sorted_indexes_list, _, sorted_groups, sorted_bboxes_groups) = sort_bbox(
                    no_match_end2end_xywh, no_match_end2end_indexes
                )
                extra_match_list = extra_match(end2end_sorted_indexes_list, len(structure_master_xywh_bboxes))
                match_list_add_extra_match = copy.deepcopy(match_list)
                match_list_add_extra_match.extend(extra_match_list)
            else:
                match_list_add_extra_match = copy.deepcopy(match_list)
                sorted_groups, sorted_bboxes_groups = [], []

            match_result_dict = {
                "match_list": match_list,
                "match_list_add_extra_match": match_list_add_extra_match,
                "sorted_groups": sorted_groups,
                "sorted_bboxes_groups": sorted_bboxes_groups,
            }
            match_result_dict = self._format(match_result_dict, file_name)
            match_results[file_name] = match_result_dict
        return match_results

    def _apply_match_rule(
        self, end2end_xywh_bboxes, end2end_xyxy_bboxes, structure_master_xywh_bboxes, structure_master_xyxy_bboxes
    ):
        match_list = []
        # Rule 1: Center rule
        match_list.extend(center_rule_match(end2end_xywh_bboxes, structure_master_xyxy_bboxes))
        # Rule 2: IoU rule
        center_no_match_end2end_indexes = find_no_match(match_list, len(end2end_xywh_bboxes), "end2end")
        if center_no_match_end2end_indexes:
            center_no_match_end2end_xyxy = end2end_xyxy_bboxes[center_no_match_end2end_indexes]
            match_list.extend(
                iou_rule_match(
                    center_no_match_end2end_xyxy, center_no_match_end2end_indexes, structure_master_xyxy_bboxes
                )
            )
        # Rule 3: Distance rule
        centerIou_no_match_end2end_indexes = find_no_match(match_list, len(end2end_xywh_bboxes), "end2end")
        centerIou_no_match_master_indexes = find_no_match(match_list, len(structure_master_xywh_bboxes), "master")
        if centerIou_no_match_end2end_indexes and centerIou_no_match_master_indexes:
            centerIou_no_match_end2end_xywh = end2end_xywh_bboxes[centerIou_no_match_end2end_indexes]
            centerIou_no_match_master_xywh = structure_master_xywh_bboxes[centerIou_no_match_master_indexes]
            match_list.extend(
                distance_rule_match(
                    centerIou_no_match_end2end_indexes,
                    centerIou_no_match_end2end_xywh,
                    centerIou_no_match_master_indexes,
                    centerIou_no_match_master_xywh,
                )
            )
        return match_list

    def _format(self, match_result, file_name):
        """
        Extend the master token (insert virtual master token), and format matching result.
        """
        master_info = self.structure_master_results[file_name]
        master_token = master_info["text"]
        sorted_groups = match_result["sorted_groups"]

        # creat virtual master token
        virtual_master_token_list = ["<tr>" + "<td></td>" * len(line_group) + "</tr>" for line_group in sorted_groups]
        # insert virtual master token
        master_token_list = master_token.split(",")
        if master_token_list[-1] == "</tbody>":
            master_token_list[:-1].extend(virtual_master_token_list)
        elif master_token_list[-1] == "<td></td>":
            master_token_list.append("</tr>")
            master_token_list.extend(virtual_master_token_list)
            master_token_list.append("</tbody>")
        else:
            master_token_list.extend(virtual_master_token_list)
            master_token_list.append("</tbody>")

        match_result["matched_master_token_list"] = master_token_list
        return match_result

    def get_merge_result(self, match_results):
        """
        Merge the OCR result into structure token to get final results.
        """
        merged_results = {}
        break_token = " "

        for file_name, match_info in match_results.items():
            end2end_info = self.end2end_results[file_name]
            master_token_list = match_info["matched_master_token_list"]
            match_list = match_info["match_list_add_extra_match"]

            match_dict = get_match_dict(match_list)
            match_text_dict = get_match_text_dict(match_dict, end2end_info, break_token)
            merged_result = insert_text_to_token(master_token_list, match_text_dict)
            merged_result = deal_bb(merged_result)
            merged_results[file_name] = merged_result

        return merged_results


class TableMasterMatcher(Matcher):
    def __init__(self):
        pass

    def __call__(self, structure_res, dt_boxes, rec_res, img_name=1):
        self.end2end_results = {
            img_name: [{"bbox": np.array(dt_box), "text": res[0]} for dt_box, res in zip(dt_boxes, rec_res)]
        }
        pred_structures, pred_bboxes = structure_res
        self.structure_master_results = {img_name: {"text": ",".join(pred_structures[3:-3]), "bbox": pred_bboxes}}

        match_results = self.match()
        merged_results = self.get_merge_result(match_results)
        pred_html = merged_results[img_name]
        return "<html><body><table>" + pred_html + "</table></body></html>"
