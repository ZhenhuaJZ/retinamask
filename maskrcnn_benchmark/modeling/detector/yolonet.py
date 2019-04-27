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
        # print("[debug] images: ", images)
        # print("[debug] images.tensors: ", images.image_sizes[0].shape)
        # print("[debug] images.tensors.size: ", images)
        # exit()
        # features = self.backbones(images.tensors)
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
        # print("[debug yolonet.py] output: ", output[0].shape)
        # Select most

        # print("[debug yolonet.py] detection: ", detection)
        # print("[debug yolonet.py] len(detection): ", len(detection))
        # print("[debug yolonet.py] detection[0]: ", detection[0])
        # print("[debug yolonet.py] detection[0].shape: ", detection[0].shape)

        # TODO: Fix target list
        '''Target is a boxcoder class and needed to be extracted for yolo processing'''
        target_list = build_targets(self.yolonet, targets)
        # print("[debug yolonet.py] target_list.shape: ", target_list)
        # exit()

        # Compute loss
        loss, loss_items = compute_loss(output, target_list)
        loss = {"sum_loss": loss}
        # print("[debug yolonet.py] loss.shape: ", loss)
        # print("[debug yolonet.py] loss.shape: ", loss.shape)
        print("[debug yolonet.py] loss_items: ", loss_items)
        # print("[debug yolonet.py] loss_items.shape: ", loss_items.shape)
        """ The following section requires (anchors [boxlist], features, detector_losses)"""
        """
        Problems:
        1. the masking prediction require 5 layers of feature map,
           each layer has corresponding number of anchors.
        2. The anchor box [boxlist] generated by yolo is different from retinanet.
           Yolo predict 6 anchor sizes, retina predicts many anchors
        3. Retina samples negative anchors while yolo does not.
        """

        # Convert yolo anchor (xywh) to retinanet format(xywh)
        image_size = images.image_sizes[0] # Height, Width
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

        '''
        non_max_suppression error -->
        variable needed for gradient computation has been modified by inplace operation

        The error may be cause by without torch.no_grad()

        Should nms then loss?
        '''
        with torch.no_grad():
            detection = non_max_suppression(detection, conf_thres=0.5, nms_thres=0.8)
            # print(len(detection[0]))
            # print(detection[0].shape)
        # ''' Use retinanet nms'''
        # if self.training:
        #     detections = self.box_selector_train(anchors, box_cls, box_regression)
        # else:
        #     detections = self.box_selector_test(anchors, box_cls, box_regression)

        #Convert yolo detections (xyhw) to retinanet format(xyxy + extra_fields(labels, scores))
        detections = []
        # for dect in detection:
        dect = BoxList(detection[0][:100,:4], (image_size[1], image_size[0]), "xyxy")#.convert("xyxy")
        # print(detection[0][:])
        # exit()
        # exit()
        # dect = dect.convert("xyxy")
        # TODO: Obtrain correct labels
        # labels =
        dect.add_field("labels", torch.tensor([1], dtype = torch.int64, device = "cuda:0" ).repeat(len(dect.bbox)))
        detections.append(dect)

        #output = list(output)
        features = []
        # print("[debug yolonet.py] output: ", type(output))
        for i, o in enumerate(output):
            feat_map_size = o.shape[2:4]
            new_feat = o.permute(0,1,4,2,3).contiguous().view(1,-1,*feat_map_size)
            features.append(new_feat)
            # print(output[i].shape)
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
                        # print("[debug yolonet.py] image_detections: ", image_detections)
                        # print("[debug yolonet.py] image_targets: ", image_targets.get_field('labels').dtype)
                        # print("[debug yolonet.py] image_targets: ", image_targets.get_field('labels').device)
                        # exit()
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
                    # print("[debug yolonet.py] features: ", features)
                    # print("[debug yolonet.py] proposals: ", proposals)
                    # exit()
                    x, result, mask_losses = self.mask(features, proposals, targets)
                elif self.cfg.MODEL.SPARSE_MASK_ON:
                    x, result, mask_losses = self.mask(features, anchors, targets)
                '''@# TODO: One of the variables needed for gradient computation'''
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
            return detections
