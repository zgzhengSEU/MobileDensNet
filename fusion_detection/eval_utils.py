from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import itertools
from terminaltables import AsciiTable
import numpy as np
from mmengine.fileio import dump
from mmengine.utils import is_str

def resize_bbox_to_original(bboxs, start_x, start_y):
    """
    Given bboxes from density crops, cast back coords to original images
    :param bboxs: bboxs from density crops
    :param start_x: The starting x position in original images
    :param start_y: The starting y position in original images
    :return: scaled annotations with coord matches original one.
    """
    # 4 coord for bbox: start_x, start_y, bbox_width, bbox_height
    # we only update first two column
    modify_bbox = []
    for bbox in bboxs:
        coord = bbox["bbox"]
        # coord.shape : 1*4
        coord[0] += start_x
        coord[1] += start_y
        bbox["bbox"] = coord
        modify_bbox.append(bbox)
    return modify_bbox


def wrap_initial_result(img_initial_fusion_result):
    """
    Given img_initial_fusion_result, wrap it to numpy array
    To perform class-wise nms, we need:
    1. global image id
    2. current bbox coord in global image
    3. current conf score in global image
    4. predicted category
    5. no need to record bbox id
    Leave nms to another function
    :param img_initial_fusion_result: raw annotations from initial data collecter
    :return: numpy array that is available to apply nms operation
    """
    nms_process_array = []
    for anno in img_initial_fusion_result:
        nms_process_array.append([anno[key] for key in ['image_id', 'category_id', 'score']] + anno['bbox'])
    return np.array(nms_process_array)

import torch
def soft_nms_pytorch(dets, sigma=0.5, thresh=0.001, cuda=0):
    """
    Build a pytorch implement of Soft NMS algorithm.
    # Augments
        dets:        boxes coordinate tensor (format:[y1, x1, y2, x2])
        box_scores:  box score tensors
        sigma:       variance of Gaussian function
        thresh:      score thresh
        cuda:        CUDA flag
    # Return
        the index of the selected boxes
    """

    # Indexes concatenate boxes with the last column
    dets = torch.from_numpy(dets)
    N = dets.shape[0]
    if cuda:
        indexes = torch.arange(0, N, dtype=torch.float).cuda().view(N, 1)
    else:
        indexes = torch.arange(0, N, dtype=torch.float).view(N, 1)
    dets = torch.cat((dets, indexes), dim=1)

    # The order of boxes coordinate is [y1,x1,y2,x2]
    x1 = dets[:, 3]
    y1 = dets[:, 4]
    x2 = dets[:, 5] + x1
    y2 = dets[:, 6] + y1
    scores = dets[:, 2]
    
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    for i in range(N):
        # intermediate parameters for later parameters exchange
        tscore = scores[i].clone()
        pos = i + 1

        if i != N - 1:
            maxscore, maxpos = torch.max(scores[pos:], dim=0)
            if tscore < maxscore:
                dets[i], dets[maxpos.item() + i + 1] = dets[maxpos.item() + i + 1].clone(), dets[i].clone()
                scores[i], scores[maxpos.item() + i + 1] = scores[maxpos.item() + i + 1].clone(), scores[i].clone()
                areas[i], areas[maxpos + i + 1] = areas[maxpos + i + 1].clone(), areas[i].clone()

        # IoU calculate
        yy1 = np.maximum(dets[i, 4].to("cpu").numpy(), dets[pos:, 4].to("cpu").numpy())
        xx1 = np.maximum(dets[i, 3].to("cpu").numpy(), dets[pos:, 3].to("cpu").numpy())
        yy2 = np.minimum(dets[i, 6].to("cpu").numpy(), dets[pos:, 6].to("cpu").numpy())
        xx2 = np.minimum(dets[i, 5].to("cpu").numpy(), dets[pos:, 5].to("cpu").numpy())
        
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = torch.tensor(w * h).cuda() if cuda else torch.tensor(w * h)
        ovr = torch.div(inter, (areas[i] + areas[pos:] - inter))

        # Gaussian decay
        weight = torch.exp(-(ovr * ovr) / sigma)
        scores[pos:] = weight * scores[pos:]

    # select the boxes and keep the corresponding indexes
    keep = dets[:, 4][scores > thresh].int()

    return keep

def py_cpu_softnms(dets, Nt=0.5, sigma=0.5, thresh=0.001, method=2):
    """
    py_cpu_softnms
    :param dets:   boexs 坐标矩阵 format [y1, x1, y2, x2]
    :param sc:     每个 boxes 对应的分数
    :param Nt:     iou 交叠门限
    :param sigma:  使用 gaussian 函数的方差
    :param thresh: 最后的分数门限
    :param method: 使用的方法
    :return:       留下的 boxes 的 index
    """

    # indexes concatenate boxes with the last column
    N = dets.shape[0]
    indexes = np.array([np.arange(N)])
    dets = np.concatenate((dets, indexes.T), axis=1)

    # the order of boxes coordinate is [y1,x1,y2,x2]
    x1 = dets[:, 3]
    y1 = dets[:, 4]
    x2 = dets[:, 5] + x1
    y2 = dets[:, 6] + y1
    scores = dets[:, 2]
    
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    for i in range(N):
        # intermediate parameters for later parameters exchange
        tBD = dets[i, :].copy()
        tscore = scores[i].copy()
        tarea = areas[i].copy()
        pos = i + 1

        #
        if i != N-1:
            maxscore = np.max(scores[pos:], axis=0)
            maxpos = np.argmax(scores[pos:], axis=0)
        else:
            maxscore = scores[-1]
            maxpos = 0
        if tscore < maxscore:
            dets[i, :] = dets[maxpos + i + 1, :]
            dets[maxpos + i + 1, :] = tBD
            tBD = dets[i, :]

            scores[i] = scores[maxpos + i + 1]
            scores[maxpos + i + 1] = tscore
            tscore = scores[i]

            areas[i] = areas[maxpos + i + 1]
            areas[maxpos + i + 1] = tarea
            tarea = areas[i]

        # IoU calculate
        xx1 = np.maximum(dets[i, 3], dets[pos:, 3])
        yy1 = np.maximum(dets[i, 4], dets[pos:, 4])
        xx2 = np.minimum(dets[i, 5], dets[pos:, 5])
        yy2 = np.minimum(dets[i, 6], dets[pos:, 6])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[pos:] - inter)

        # Three methods: 1.linear 2.gaussian 3.original NMS
        if method == 1:  # linear
            weight = np.ones(ovr.shape)
            weight[ovr > Nt] = weight[ovr > Nt] - ovr[ovr > Nt]
        elif method == 2:  # gaussian
            weight = np.exp(-(ovr * ovr) / sigma)
        else:  # original NMS
            weight = np.ones(ovr.shape)
            weight[ovr > Nt] = 0

        scores[pos:] = weight * scores[pos:]

    # select the boxes and keep the corresponding indexes
    inds = dets[:, 7][scores > thresh]
    keep = inds.astype(int)

    return keep

def py_soft_nms(dets, method='linear', iou_thr=0.3, sigma=0.5, score_thr=0.001):
    """Pure python implementation of soft NMS as described in the paper
    `Improving Object Detection With One Line of Code`_.

    Args:
        dets (numpy.array): Detection results with shape `(num, 5)`,
            data in second dimension are [x1, y1, x2, y2, score] respectively.
        method (str): Rescore method. Only can be `linear`, `gaussian`
            or 'greedy'.
        iou_thr (float): IOU threshold. Only work when method is `linear`
            or 'greedy'.
        sigma (float): Gaussian function parameter. Only work when method
            is `gaussian`.
        score_thr (float): Boxes that score less than the.

    Returns:
        numpy.array: Retained boxes.

    .. _`Improving Object Detection With One Line of Code`:
        https://arxiv.org/abs/1704.04503
    """
    if method not in ('linear', 'gaussian', 'greedy'):
        raise ValueError('method must be linear, gaussian or greedy')

    x1 = dets[:, 3]
    y1 = dets[:, 4]
    x2 = dets[:, 5] + x1
    y2 = dets[:, 6] + y1

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # expand dets with areas, and the second dimension is
    # x1, y1, x2, y2, score, area
    dets = np.concatenate((dets, areas[:, None]), axis=1)
    
    N = dets.shape[0]
    indexes = np.array([np.arange(N)])
    dets = np.concatenate((dets, indexes.T), axis=1)
    
    retained_box = []
    keep = []
    while dets.size > 0:
        max_idx = np.argmax(dets[:, 2], axis=0)
        dets[[0, max_idx], :] = dets[[max_idx, 0], :]
        retained_box.append(dets[0, :-1])
        keep.append(dets[0, 8].astype(int))
        xx1 = np.maximum(dets[0, 3], dets[1:, 3])
        yy1 = np.maximum(dets[0, 4], dets[1:, 4])
        xx2 = np.minimum(dets[0, 5], dets[1:, 5])
        yy2 = np.minimum(dets[0, 6], dets[1:, 6])

        w = np.maximum(xx2 - xx1 + 1, 0.0)
        h = np.maximum(yy2 - yy1 + 1, 0.0)
        inter = w * h
        iou = inter / (dets[0, 7] + dets[1:, 7] - inter)

        if method == 'linear':
            weight = np.ones_like(iou)
            weight[iou > iou_thr] -= iou[iou > iou_thr]
        elif method == 'gaussian':
            weight = np.exp(-(iou * iou) / sigma)
        else:  # traditional nms
            weight = np.ones_like(iou)
            weight[iou > iou_thr] = 0

        dets[1:, 2] *= weight
        retained_idx = np.where(dets[1:, 2] >= score_thr)[0]
        dets = dets[retained_idx + 1, :]

    return keep

def nms(dets, iou_thresh):
    '''
    Fast NMS implementation from detectron
    dets is a numpy array : num_dets, 4, but x2,y2 is height and width instead of coord
    scores is a  nump array : num_dets,
    '''
    # print("selected thr is: {}".format(thresh))
    x1 = dets[:, 3]
    y1 = dets[:, 4]
    x2 = dets[:, 5] + x1
    y2 = dets[:, 6] + y1
    scores = dets[:, 2]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]  # get boxes with more ious first

    keep = []
    while order.size > 0:
        i = order[0]  # pick maxmum iou box
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)  # maximum width
        h = np.maximum(0.0, yy2 - yy1 + 1)  # maxiumum height
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_thresh)[0]
        order = order[inds + 1]
    return keep


def class_wise_nms(current_nms_target_col, thresh, TopN):
    # NOT recommended to use. Try nms instead
    # image_id is optional
    # if final detection amount > TopN, sort by bbox and only take first TopN
    # print(current_nms_target_col.shape)
    bbox_id = np.array([i for i in range(len(current_nms_target_col))])
    truncate_result = current_nms_target_col.copy()
    current_nms_target_col[:, 0] = bbox_id
    categories = current_nms_target_col[:, 1]
    keep = []
    for category in set(categories):
        mask = current_nms_target_col[:, 1] == category
        mask = [i for i in range(len(mask)) if mask[i]]
        current_nms_target = current_nms_target_col[mask]
        current_nms_target_col = np.delete(current_nms_target_col, mask, axis=0)
        scores = current_nms_target[:, 2]
        order = scores.argsort()[::-1]
        x1 = current_nms_target[:, 3]
        y1 = current_nms_target[:, 4]
        x2 = current_nms_target[:, 5] + x1
        y2 = current_nms_target[:, 6] + y1

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)

        while order.size > 0:
            i = order[0]
            keep.append(int(current_nms_target[i][0]))
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
        # print("Element removed: ", len(current_nms_target)-len(keep))
    if len(keep) > TopN:
        fusion_result = truncate_result[keep, :]
        scores = fusion_result[:, 2]
        keep = scores.argsort()[::-1][:TopN]
    return keep


def results2json(json_results, out_file):
    """
    Generate fused annotations to json files
    :param json_results: list, collect results to dump
    :param out_file: The output path for json file
    :return:
    """
    result_files = dict()
    result_files['bbox'] = '{}.{}.json'.format(out_file, 'bbox')
    result_files['proposal'] = '{}.{}.json'.format(out_file, 'bbox')
    dump(json_results, result_files['bbox'])


def coco_eval(result_files,
              result_types,
              coco,
              max_dets=(100, 300, 1000),
              classwise=False):
    """
    Code from MMdetection
    Evaluate given files, with given task objective
    :param result_files: the detection file to evaluate
    :param result_types: task objective, detection? segmentation? Or something else?
    :param coco: coco object that holds all annotations
    :param max_dets: max amount of detections
    :param classwise: Conduct class-wise evaluation
    :return:
    """
    for res_type in result_types:
        assert res_type in [
            'proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'
        ]
    # borrow from mcnn
    # coco file -> cocoGt_global
    # re-load fusion result ,instead of cocoDt
    if is_str(coco):
        coco = COCO(coco)
    assert isinstance(coco, COCO)

    if result_types == ['proposal_fast']:
        ar = fast_eval_recall(result_files, coco, np.array(max_dets))
        for i, num in enumerate(max_dets):
            print('AR@{}\t= {:.4f}'.format(num, ar[i]))
        return

    for res_type in result_types:
        if isinstance(result_files, str):
            result_file = result_files
        elif isinstance(result_files, dict):
            result_file = result_files[res_type]
        else:
            assert TypeError('result_files must be a str or dict')
        assert result_file.endswith('.json')

        coco_dets = coco.loadRes(result_file)
        img_ids = coco.getImgIds()
        iou_type = 'bbox' if res_type == 'proposal' else res_type
        cocoEval = COCOeval(coco, coco_dets, iou_type)
        cocoEval.params.imgIds = img_ids
        if res_type == 'proposal':
            cocoEval.params.useCats = 0
            cocoEval.params.maxDets = list(max_dets)
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        if classwise:
            # Compute per-category AP
            # from https://github.com/facebookresearch/detectron2/blob/03064eb5bafe4a3e5750cc7a16672daf5afe8435/detectron2/evaluation/coco_evaluation.py#L259-L283 # noqa
            precisions = cocoEval.eval['precision']
            catIds = coco.getCatIds()
            # precision has dims (iou, recall, cls, area range, max dets)
            assert len(catIds) == precisions.shape[2]

            results_per_category = []
            for idx, catId in enumerate(catIds):
                # area range index 0: all area ranges
                # max dets index -1: typically 100 per image
                nm = coco.loadCats(catId)[0]
                precision = precisions[:, :, idx, 0, -1]
                precision = precision[precision > -1]
                ap = np.mean(precision) if precision.size else float('nan')
                results_per_category.append(
                    ('{}'.format(nm['name']),
                     '{:0.3f}'.format(float(ap * 100))))

            N_COLS = min(6, len(results_per_category) * 2)
            results_flatten = list(itertools.chain(*results_per_category))
            headers = ['category', 'AP'] * (N_COLS // 2)
            results_2d = itertools.zip_longest(
                *[results_flatten[i::N_COLS] for i in range(N_COLS)])
            table_data = [headers]
            table_data += [result for result in results_2d]
            table = AsciiTable(table_data)
            print(table.table)


