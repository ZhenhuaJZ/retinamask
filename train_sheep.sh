python tools/train_net.py --config-file "configs/retina/retinanet_mask_R-101-FPN_2x_adjust_std011_ms.yaml" SOLVER.IMS_PER_BATCH 2 MODEL.WEIGHT ./pre_train_model/retinanet_mask_R-101-FPN_2x_adjust_std011_ms_model.pth OUTPUT_DIR ./sheep_model_w_pre-train
