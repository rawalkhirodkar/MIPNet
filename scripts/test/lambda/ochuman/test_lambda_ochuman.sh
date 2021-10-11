cd ../../../..


# # # # -----------------------------------------------------------
# # # # # # # -----------------------------------------------------------
# CUDA_VISIBLE_DEVICES=0,1,2 python tools/lambda/test_lambda.py \
#     --cfg experiments/coco/hrnet/w48_384x288_adam_lr1e-3.yaml \
#     GPUS '(0,1,2)' \
#     OUTPUT_DIR 'Outputs/outputs/ochuman/lambda/lambda_real/' \
#     LOG_DIR 'Outputs/logs/ochuman/lambda/lambda_real/' \
#     TEST.MODEL_FILE '/home/XYZ/Desktop/intelligentmix/hrnet/models/lambda/hrnetw48-384x288/checkpoint_103.pth' \
#     DATASET.TEST_DATASET 'ochuman' \
#     DATASET.TEST_SET 'val2017' \
#     DATASET.TEST_IMAGE_DIR '/home/XYZ/Desktop/datasets/OCHuman/images'\
#     DATASET.TEST_ANNOTATION_FILE '/home/XYZ/Desktop/datasets/OCHuman/annotations/ochuman_coco_format_val_range_0.00_1.00.json' \
#     TEST.USE_GT_BBOX True \
#     TEST.BATCH_SIZE_PER_GPU 256 \
#     TEST.POST_PROCESS True \
#     MODEL.NAME 'pose_hrnet_se_lambda' \
#     MODEL.SE_MODULES '[False, False, True, True]' \
#     PRINT_FREQ 100 \
#     TEST.COCO_BBOX_FILE '/home/ANT.AMAZON.COM/khiXYZ/Desktop/datasets/coco/person_detection_results/COCO_val2017_detections_AP_H_56_person.json'


# CUDA_VISIBLE_DEVICES=0,1,2 python tools/lambda/test_lambda.py \
#     --cfg experiments/coco/hrnet/w48_384x288_adam_lr1e-3.yaml \
#     GPUS '(0,1,2)' \
#     OUTPUT_DIR 'Outputs/outputs/paper/ochuman/lambda/' \
#     LOG_DIR 'Outputs/logs/ochuman/paper/lambda/' \
#     TEST.MODEL_FILE '/home/XYZ/Desktop/intelligentmix/hrnet/models/lambda/hrnetw48-384x288/checkpoint_103.pth' \
#     DATASET.TEST_DATASET 'ochuman' \
#     DATASET.TEST_SET 'val2017' \
#     DATASET.TEST_IMAGE_DIR '/home/XYZ/Desktop/datasets/OCHuman/images'\
#     DATASET.TEST_ANNOTATION_FILE '/home/XYZ/Desktop/datasets/OCHuman/annotations/ochuman_coco_format_val_range_0.00_1.00.json' \
#     TEST.USE_GT_BBOX True \
#     TEST.BATCH_SIZE_PER_GPU 256 \
#     TEST.POST_PROCESS True \
#     MODEL.NAME 'pose_hrnet_se_lambda' \
#     MODEL.SE_MODULES '[False, False, True, True]' \
#     PRINT_FREQ 100 \
#     TEST.OKS_THRE 0.7 \


# CUDA_VISIBLE_DEVICES=0,1,2 python tools/lambda/test_lambda.py \
#     --cfg experiments/coco/hrnet/w48_384x288_adam_lr1e-3.yaml \
#     GPUS '(0,1,2)' \
#     OUTPUT_DIR 'Outputs/outputs/paper/ochuman/lambda/' \
#     LOG_DIR 'Outputs/logs/ochuman/paper/lambda/' \
#     TEST.MODEL_FILE '/home/XYZ/Desktop/intelligentmix/hrnet/models/lambda/hrnetw48-384x288/checkpoint_103.pth' \
#     DATASET.TEST_DATASET 'ochuman' \
#     DATASET.TEST_SET 'val2017' \
#     DATASET.TEST_IMAGE_DIR '/home/XYZ/Desktop/datasets/OCHuman/images'\
#     DATASET.TEST_ANNOTATION_FILE '/home/XYZ/Desktop/datasets/OCHuman/annotations/ochuman_coco_format_val_range_0.00_1.00.json' \
#     TEST.USE_GT_BBOX True \
#     TEST.BATCH_SIZE_PER_GPU 256 \
#     TEST.POST_PROCESS True \
#     MODEL.NAME 'pose_hrnet_se_lambda' \
#     MODEL.SE_MODULES '[False, False, True, True]' \
#     PRINT_FREQ 100 \
#     TEST.OKS_THRE 0.7 \


CUDA_VISIBLE_DEVICES=0,1,2 python tools/lambda/test_lambda.py \
    --cfg experiments/coco/hrnet/w48_384x288_adam_lr1e-3.yaml \
    GPUS '(0,1,2)' \
    OUTPUT_DIR 'Outputs/supp/lambda_0.9/' \
    LOG_DIR 'Outputs/supp/lambda_0.9/' \
    TEST.MODEL_FILE '/home/XYZ/Desktop/intelligentmix/hrnet/models/lambda/hrnetw48-384x288/checkpoint_103.pth' \
    DATASET.TEST_DATASET 'ochuman' \
    DATASET.TEST_SET 'val2017' \
    DATASET.TEST_IMAGE_DIR '/home/XYZ/Desktop/datasets/OCHuman/images'\
    DATASET.TEST_ANNOTATION_FILE '/home/XYZ/Desktop/datasets/OCHuman/annotations/ochuman_coco_format_val_range_0.00_1.00.json' \
    TEST.USE_GT_BBOX True \
    TEST.BATCH_SIZE_PER_GPU 256 \
    TEST.POST_PROCESS True \
    MODEL.NAME 'pose_hrnet_se_lambda' \
    MODEL.SE_MODULES '[False, False, True, True]' \
    PRINT_FREQ 100 \
    TEST.OKS_THRE 0.9 \



# CUDA_VISIBLE_DEVICES=0,1,2 python tools/lambda/test_lambda.py \
#     --cfg experiments/coco/resnet/res50_384x288_d256x3_adam_lr1e-3.yaml \
#     GPUS '(0,1,2)' \
#     OUTPUT_DIR 'Outputs/outputs/ochuman/lambda/lambda_real/' \
#     LOG_DIR 'Outputs/logs/ochuman/lambda/lambda_real/' \
#     TEST.MODEL_FILE '/home/XYZ/Desktop/intelligentmix/hrnet/models/lambda/resnet50-384x288/checkpoint_109.pth' \
#     DATASET.TEST_DATASET 'ochuman' \
#     DATASET.TEST_SET 'val2017' \
#     DATASET.TEST_IMAGE_DIR '/home/XYZ/Desktop/datasets/OCHuman/images'\
#     DATASET.TEST_ANNOTATION_FILE '/home/XYZ/Desktop/datasets/OCHuman/annotations/ochuman_coco_format_val_range_0.00_1.00.json' \
#     TEST.USE_GT_BBOX True \
#     TEST.BATCH_SIZE_PER_GPU 256 \
#     TEST.POST_PROCESS True \
#     MODEL.NAME 'pose_resnet_se_lambda' \
#     MODEL.SE_MODULES '[False, False, True, True]' \
#     PRINT_FREQ 100 \
#     TEST.COCO_BBOX_FILE '/home/ANT.AMAZON.COM/khiXYZ/Desktop/datasets/coco/person_detection_results/COCO_val2017_detections_AP_H_56_person.json'

# CUDA_VISIBLE_DEVICES=0,1,2 python tools/lambda/test_lambda.py \
#     --cfg experiments/coco/hrnet/w32_384x288_adam_lr1e-3.yaml \
#     GPUS '(0,1,2)' \
#     OUTPUT_DIR 'Outputs/outputs/ochuman/lambda/lambda_real/' \
#     LOG_DIR 'Outputs/logs/ochuman/lambda/lambda_real/' \
#     TEST.MODEL_FILE '/home/XYZ/Desktop/intelligentmix/hrnet/models/lambda/hrnetw32-384x288/checkpoint_9.pth' \
#     DATASET.TEST_DATASET 'ochuman' \
#     DATASET.TEST_SET 'val2017' \
#     DATASET.TEST_IMAGE_DIR '/home/XYZ/Desktop/datasets/OCHuman/images'\
#     DATASET.TEST_ANNOTATION_FILE '/home/XYZ/Desktop/datasets/OCHuman/annotations/ochuman_coco_format_val_range_0.00_1.00.json' \
#     TEST.USE_GT_BBOX True \
#     TEST.BATCH_SIZE_PER_GPU 256 \
#     TEST.POST_PROCESS True \
#     MODEL.NAME 'pose_hrnet_se_lambda' \
#     MODEL.SE_MODULES '[False, False, True, True]' \
#     PRINT_FREQ 100 \
#     TEST.COCO_BBOX_FILE '/home/ANT.AMAZON.COM/khiXYZ/Desktop/datasets/coco/person_detection_results/COCO_val2017_detections_AP_H_56_person.json'

