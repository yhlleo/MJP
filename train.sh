
IMG_SIZE=224 
MODE=vit_s_16 
CONFIG=vit_s_16 
LAMBDA_DLOCR=0.01

DATASET=imagenet # imagenet-100, imagenet, cifar-10, cifar-100, svhn, places365, flowers102, pets, clipart, infograph, painting, quickdraw, real, sketch
NUM_CLASSES=1000
DATA_DIR=/path/to/root_dir 
DISK_DATA=${DATA_DIR}/datasets/${DATASET}

TARGET_FOLDER=${DATASET}-${MODE}-sz${IMG_SIZE}-bs128-g8-jigsaw-dlocr${LAMBDA_DLOCR}
SAVE_DIR=${DATA_DIR}/jigsaw-expr/${TARGET_FOLDER}

python3 -m torch.distributed.launch \
    --nproc_per_node=8 \
    --nnodes 1 \
    --node_rank 0 \
    --master_port 12345  main.py \
    --cfg ./configs/${CONFIG}_${IMG_SIZE}.yaml \
    --dataset ${DATASET} \
    --num_classes ${NUM_CLASSES} \
    --data-path ${DISK_DATA} \
    --output ${SAVE_DIR} \
    --repeated-aug \
    --decay-posemb \
    --patch_size 16 \
    --use-jigsaw \
    --use-unk-pos \
    --use-dlocr \
    --lambda_dlocr ${LAMBDA_DLOCR} \
    --dlocr-type nonlinear \
    --mask-type mjp \
    --mask-ratio 0.1
