# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.retinanet import build_retinanet
from maskrcnn_benchmark.modeling.roi_heads.mask_head.mask_head import build_roi_mask_head
#from maskrcnn_benchmark.modeling.roi_heads.sparsemask_head.mask_head import build_sparse_mask_head
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
import copy
import cv2


class RetinaNet(nn.Module):
    """
    Main class for RetinaNet
    It consists of three main parts:
    - backbone
    - bbox_heads: BBox prediction.
    - Mask_heads:
    """

    def __init__(self, cfg):
        super(RetinaNet, self).__init__()
        self.cfg = copy.deepcopy(cfg)
        self.backbone = build_backbone(cfg)
        self.rpn = build_retinanet(cfg)
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
        images = to_image_list(images)
        # print("[debug retinanet.py] images.tensors.shape: ", images.tensors.shape)
        features = self.backbone(images.tensors)

        # Retina RPN Output
        rpn_features = features
        if self.cfg.RETINANET.BACKBONE == "p2p7":
            rpn_features = features[1:]
        '''Retinanet module in rpn.retinanet'''
        (anchors, detections), detector_losses = self.rpn(images, rpn_features, targets)
        print("[debug retinanet.py] detections: ", detections)
        # print("[debug retinanet.py] detector_losses: ", detector_losses)
        #print("[debug retinanet.py] anchors[0]: ", anchors[0])
        # print("[debug retinanet.py] len(anchors[0][0]): ", len(anchors[0][0]))
        # print("[debug retinanet.py] anchors[0][0].bbox.shape: ", anchors[0][0].bbox.shape)
        # print("[debug retinanet.py] anchors[0][0].extra_fields: ", anchors[0][0].extra_fields)
        # # print("[debug retinanet.py] anchors[0]: ", anchors[0])
        #
        #print("[debug retinanet.py] detections: ", detections)
        # print("[debug retinanet.py] type(detections): ", type(detections))
        # print("[debug retinanet.py] detections[0].bbox: ", detections[0].bbox)
        # print("[debug retinanet.py] detections[0].extra_fields: ", detections[0].extra_fields)
        #
        # print("[debug retinanet.py] detector_losses: ", detector_losses)
        #print("[debug retinanet.py] len(features): ", len(features))
        #print("[debug retinanet.py] type(features): ", type(features))
        features = features[3:]

        if self.training:
            losses = {}
            losses.update(detector_losses)
            if self.mask:
                if self.cfg.MODEL.MASK_ON:
                    # Padding the GT
                    proposals = []
                    for (image_detections, image_targets) in zip(
                        detections, targets):
                        merge_list = []
                        # print("[debug retinanet.py] image_detections.extra_fields: ", image_detections.extra_fields)
                        # print("[debug retinanet.py] image_detections.bbox: ", image_detections.bbox)
                        # print("[debug retinanet.py] image_targets: ", image_targets)
                        # exit()
                        if not isinstance(image_detections, list):
                            merge_list.append(image_detections.copy_with_fields('labels'))

                        if not isinstance(image_targets, list):
                            merge_list.append(image_targets.copy_with_fields('labels'))
                        if len(merge_list) == 1:
                            proposals.append(merge_list[0])
                        else:
                            proposals.append(cat_boxlist(merge_list))
                    x, result, mask_losses = self.mask(features, proposals, targets)
                elif self.cfg.MODEL.SPARSE_MASK_ON:
                    x, result, mask_losses = self.mask(features, anchors, targets)
                # print("[debug retinanet.py] mask_losses: ", mask_losses)
                # print("[debug retinanet.py] image_targets: ", image_targets)

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
