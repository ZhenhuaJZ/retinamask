

for iter in {0002501..0027501..0002500}
#while [ -e model_${pth}.pth ];
do
	python tools/test_net.py --config-file ./configs/retina/retinanet_mask_R-101-FPN_1.5x_adjust_std011_800.yaml MODEL.WEIGHT ./retina_test_416/model_${iter}.pth OUTPUT_DIR ./eval/model_${iter} TEST.IMS_PER_BATCH 4
#	echo ./model_${iter}.pth
done
