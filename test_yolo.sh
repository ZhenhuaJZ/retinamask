
for iter in {0200001..0210001..0002500}
#while [ -e model_${pth}.pth ];
do
	python tools/test_net.py --config-file ./configs/yolonet/yolonet_mask_R-101-FPN_2x_adjust_std011_ms.yaml MODEL.WEIGHT ./yolo_test/model_${iter}.pth OUTPUT_DIR ./yolo_test/eval/model_${iter} TEST.IMS_PER_BATCH 1
#	echo ./model_${iter}.pth
done

