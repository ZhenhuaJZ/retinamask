#!/bin/bash
MODEL=$1

for ITER in 0089999 0080001 0070001 0060001 0050001 0040001 0030001 0020001 0010001
do
	python tools/test_net.py --config-file ./configs/retina/retinanet_mask_R-101-FPN_1.5x_adjust_std011_800.yaml MODEL.WEIGHT ./${MODEL}/model_${ITER}.pth OUTPUT_DIR ./${MODEL}/${ITER} TEST.IMS_PER_BATCH 4
done

#for ITER in 89999
#do
#	python tools/test_net.py --config-file ./configs/retina/retinanet_R-50-FPN_1x.yaml MODEL.WEIGHT ./models/retinanet_R-50-FPN_1x_1101/model_00${ITER}.pth OUTPUT_DIR ./models/retinanet_R-50-FPN_1x_1101/${ITER} TEST.IMS_PER_BATCH 1
#done
