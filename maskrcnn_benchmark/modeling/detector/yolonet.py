# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

import sys

# sys.path.append('/home/stirfryrabbit/Projects/Research_Project/sheepCount')

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.retinanet import build_retinanet
from maskrcnn_benchmark.modeling.roi_heads.mask_head.mask_head import build_roi_mask_head
#from maskrcnn_benchmark.modeling.roi_heads.sparsemask_head.mask_head import build_sparse_mask_head
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.structures.bounding_box import BoxList

from maskrcnn_benchmark.modeling.rpn.retinanet_infer import make_retinanet_postprocessor
import cv2

import copy

from maskrcnn_benchmark.yolov3.models import *
from maskrcnn_benchmark.yolov3.utils.datasets import *
from maskrcnn_benchmark.yolov3.utils.utils import *

class YoloNet(nn.Module):
    """
    Main class for RetinaNet
    It consists of three main parts:
    - backbone
    - bbox_heads: BBox prediction.
    - Mask_heads:
    """

    def __init__(self, cfg):
        super(YoloNet, self).__init__()
        self.cfg = copy.deepcopy(cfg)
        self.yolonet = Darknet(cfg.YOLONET.CONFIG_PATH)
        # self.backbone = build_backbone(cfg)
        # self.rpn = build_retinanet(cfg)
        # # box_coder = BoxCoder(weights=(10., 10., 5., 5.))
        #
        # if self.cfg.MODEL.SPARSE_MASK_ON:
        #     box_selector_test = make_retinanet_detail_postprocessor(
        #         self.cfg, 100, box_coder)
        # else:
        #     box_selector_test = make_retinanet_postprocessor(
        #         self.cfg, 100, box_coder)
        # box_selector_train = None
        # if self.cfg.MODEL.MASK_ON or self.cfg.MODEL.SPARSE_MASK_ON:
        #     box_selector_train = make_retinanet_postprocessor(
        #         self.cfg, 100, box_coder)
        #
        # self.box_selector_test = box_selector_test
        # self.box_selector_train = box_selector_train

        self.mask = None
        if cfg.MODEL.MASK_ON:
            self.mask = build_roi_mask_head(cfg)
        #if cfg.MODEL.SPARSE_MASK_ON:
        #    self.mask = build_sparse_mask_head(cfg)

    def yolo2BoxList(self, box_list, image_size, topk = 100):
        ## DEBUG: Check batch output
        detections = []
        for detection in box_list:
            if detection is None:
                dect = BoxList(torch.zeros((1,4), dtype = torch.float32, device = "cuda:0" ), (image_size[1], image_size[0]), "xyxy")
                dect.add_field("labels", torch.tensor([1], dtype = torch.int64, device = "cuda:0" ).repeat(len(dect.bbox)))
                dect.add_field("objectness", torch.tensor([0], dtype = torch.float32, device = "cuda:0" ).repeat(len(dect.bbox)))
                dect.add_field("scores", torch.tensor([0.01], dtype = torch.float32, device = "cuda:0" ).repeat(len(dect.bbox)))
            else:
                print("num detec: ", len(detection))
                # print("targets ", targets.convert())
                dect = BoxList(detection[:topk,:4], (image_size[1], image_size[0]), "xyxy")#.convert("xyxy")
                dect.add_field("labels", detection[:topk, 6].to(torch.int64))
                dect.add_field("objectness", detection[:topk, 4])
                print("objectness: ", detection[:topk, 4])
                dect.add_field("scores", detection[:topk, 5])
            detections.append(dect)
        return detection

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        # Image is converted to image list class
        images = to_image_list(images)
        image_size = images.image_sizes[0]
        ''' Issue 2: outputing feature, and predicted box'''
        # Yolonet is Darknet in model.py
        # need to return layer feature and anchor box
        # yolonet returns all predictions from three yolo layer
        ''' make sure data set input is x,y,h,w '''
        '''
        detection: (1,anchor,5+class)
        output: (1, Num_anchor, anchorH, anchorW, 5+class)
        '''
        detection, output, features = self.yolonet(images) #, targets)
        # TODO: Fix target list
        '''Target is a boxcoder class and needed to be extracted for yolo processing'''
        if self.training:
            # Targets = [x1, y1, x2, y2]
            target_list = build_targets(self.yolonet, targets, image_size)

        # Compute loss
            loss, loss_items = compute_loss(output, target_list)
            loss = {"sum_loss": loss}
        # Convert yolo anchor (xywh) to retinanet format(xywh)
        # Height, Width
        # anchors = []
        # for i in self.yolonet.yolo_layers:
        #     layer = self.yolonet.module_list[i][0]
        #     # Append anchor box w and h to grid x and y
        #     anchor_list = []
        #     for anchor in layer.anchor_vec:
        #         anchor = anchor.view(1,2).repeat((layer.grid_xy.shape[2]*layer.grid_xy.shape[3], 1))
        #         anchor = torch.cat((layer.grid_xy, anchor.view(layer.grid_xy.shape)), 4)
        #         anchor = anchor.view(-1,4)
        #         anchor *= layer.stride[0]
        #         anchor_list.append(anchor)
        #     anchor = torch.cat(anchor_list, 0)
        #     anchor = BoxList(anchor, (image_size[1], image_size[0]), "xywh")
        #     anchor = anchor.convert("xyxy")
        #     anchors.append(anchor)
        # print(anchors)
        with torch.no_grad():
            detection = non_max_suppression(detection, conf_thres=0.6, nms_thres=0.6)
            # Convert yolo detection to retina boxlist format
            detections = self.yolo2BoxList(detection, image_size, topk = 50)


        #output = list(output)
        # features = []
        # print("[debug yolonet.py] output: ", type(output))
        # for i in range(len(output)):
        #     o = output[i]
        #     feat_map_size = o.shape[2:4]
        #     new_feat = o.permute(0,1,4,2,3).contiguous().view(1,-1,*feat_map_size)
        #     features.append(new_feat)
        # for i, o in enumerate(output):
        #     feat_map_size = o.shape[2:4]
        #     new_feat = o.permute(0,1,4,2,3).contiguous().view(1,-1,*feat_map_size)
        #     features.append(new_feat)
        #     print(output[i].shape)
        #features = output

        """Mask network"""
        if self.training:
            losses = {}
            losses.update(loss)
            if self.mask:
                if self.cfg.MODEL.MASK_ON:
                    # Padding the GT
                    proposals = []
                    for (image_detections, image_targets) in zip(
                        detections, targets):
                        merge_list = []
                        if not isinstance(image_detections, list):
                            merge_list.append(image_detections.copy_with_fields('labels'))

                        if not isinstance(image_targets, list):
                            merge_list.append(image_targets.copy_with_fields('labels'))

                        if len(merge_list) == 1:
                            proposals.append(merge_list[0])
                        else:
                            proposals.append(cat_boxlist(merge_list))
                    '''# TODO:  Double check mask, low loss'''
                    # features has backward
                    x, result, mask_losses = self.mask(features, proposals, targets)
                elif self.cfg.MODEL.SPARSE_MASK_ON:
                    x, result, mask_losses = self.mask(features, anchors, targets)
                losses.update(mask_losses)
            return losses
        else:
            if self.mask:
                proposals = []
                for image_detections in detections:
                    num_of_detections = image_detections.bbox.shape[0]
                    if num_of_detections > self.cfg.RETINANET.NUM_MASKS_TEST > 0:
                        cls_scores = image_detections.get_field("scores")
                        image_thresh, _ = torch.kthvalue(
                            cls_scores.cpu(), num_of_detections - \
                            self.cfg.RETINANET.NUM_MASKS_TEST + 1
                        )
                        keep = cls_scores >= image_thresh.item()
                        keep = torch.nonzero(keep).squeeze(1)
                        image_detections = image_detections[keep]

                    proposals.append(image_detections)

                if self.cfg.MODEL.SPARSE_MASK_ON:
                    x, detections, mask_losses = self.mask(
                        features, proposals, targets
                    )
                else:
                    x, detections, mask_losses = self.mask(features, proposals, targets)
                    print("[debug yolonet.py] detections: ", detections)
                    # print("[debug yolonet.py] detections[0].extra_fields: ", detections[0].extra_fields)
            return detections
