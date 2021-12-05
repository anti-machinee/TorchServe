import numpy as np

from src.post_process.database import *


class Decoder:
    def __init__(self):
        pass

    @staticmethod
    def check_in_same_line(box1, box2):
        # box1: xmin, ymin, xmax, ymax
        # box2: xmin, ymin, xmax, ymax
        center_y1 = (box1[1] + box2[3]) / 2
        center_y2 = (box2[1] + box2[3]) / 2
        diff = int(abs(center_y1 - center_y2))
        if diff < min((box1[3] - box1[1], box2[3] - box2[1])) / 5:
            return True
        else:
            return False

    def boxes_to_line(self, boxes, texts):
        # boxes: list of [xmin, ymin, xmax, ymax]
        idx_sort_boxes = sorted(range(len(boxes)), key=lambda x: boxes[x][1])
        boxes = [boxes[i] for i in idx_sort_boxes]
        sort_texts = [texts[i] for i in idx_sort_boxes]

        lines_coor_sort_follow_y = []
        lines_text_sort_follow_y = []
        tmp_line = []
        tmp_text = []
        for i in range(len(boxes)):
            box = boxes[i]
            text = sort_texts[i]
            # check boxes in one line follow y axis
            if len(tmp_line) == 0 or self.check_in_same_line(box, tmp_line[-1]):
                tmp_line.append(box)
                tmp_text.append(text)
            else:
                lines_coor_sort_follow_y.append(tmp_line)
                lines_text_sort_follow_y.append(tmp_text)
                tmp_line = [box]
                tmp_text = [text]
        lines_coor_sort_follow_y.append(tmp_line)
        lines_text_sort_follow_y.append(tmp_text)

        # sort words in each line
        lines_coor_sort_follow_x = []
        lines_text_sort_follow_x = []
        for i in range(len(lines_coor_sort_follow_y)):
            line = lines_coor_sort_follow_y[i]
            text = lines_text_sort_follow_y[i]
            idx_sort_line = sorted(range(len(line)), key=lambda x: line[x][0])
            lines_coor_sort_follow_x.append([line[j] for j in idx_sort_line])
            lines_text_sort_follow_x.append([text[j] for j in idx_sort_line])

        return lines_coor_sort_follow_x, lines_text_sort_follow_x

    def boxes2dict(self, boxes, box_labels, texts, text_scores):
        labels = np.array(box_labels)
        result = dict()
        for k, v in inverted_mapping_detect.items():
            # k: int, v: string
            index_k = np.array(range(len(boxes)))[labels == k]
            if len(index_k) == 0:
                continue
            boxes_k = [boxes[i] for i in index_k]
            texts_k = [texts[i] for i in index_k]
            score_k = np.mean([text_scores[i] for i in index_k])
            _, lines_k = self.boxes_to_line(boxes_k, texts_k)
            tmp_lines = []
            for l in lines_k:
                tmp_lines.extend(l)
            result[v] = dict()
            result[v]['value'] = ' '.join(tmp_lines).upper().replace('+', '/')
            result[v]['score'] = float(score_k)
        return result

    def process(self, data):
        result = self.boxes2dict(data["boxes"], data["box_labels"], data["texts"], data["text_scores"])
        final_result = dict()
        for k, v in result.items():
            if k not in KEYS[data["type"]]:
                continue
            final_result[k] = v
        if data["type"] == 'cccd_b':
            final_result['nc'] = dict()
            final_result['nc']['value'] = 'CỤC TRƯỞNG CỤC CẢNH SÁT ĐKQL CƯ TRÚ VÀ DLQG VỀ DÂN CƯ'
            final_result['nc']['score'] = 1.
        if data["type"] == 'cccd_f':
            if 'qt' in final_result:
                if final_result['qt']['value'] in DAN_TOC:
                    final_result['dt'] = final_result['qt']
                    final_result['class'] = dict()
                    final_result['class']['value'] = 'cmt_f'
                    del final_result['qt']
            if 'dt' in final_result:
                if final_result['dt']['value'] in DAN_TOC:
                    final_result['class'] = dict()
                    final_result['class']['value'] = 'cmt_f'
        if data["type"] == 'passport':
            for key in ['ns', 'gt', 'gtd', 'ngc']:
                if key in final_result:
                    if key == 'gtd':
                        if 'k' in final_result[key]['value'].lower():
                            continue
                    final_result[key]['value'] = final_result[key]['value'].replace(' ', '')
        else:
            if 'ngc' in final_result.keys():
                # print(final_result['ngc'])
                final_result['ngc']['value'] = final_result['ngc']['value'].replace(' ', '/')
        return final_result
