# !/bin/bash

# custom config
# DATA=$1
# DATASET=$2
# SHOTS=$3  # number of shots (1, 2, 4, 8, 16)
# MODEL=$4
# NCTX=4
# CSC=False
# CTP=end

# METHOD=BiomedCoOp
# TRAINER=BiomedCoOp_${MODEL}

# for SEED in 42 1024 3407 #42 1024 3407 1 2 3
# do
#         DIR=output/${DATASET}/shots_${SHOTS}/${TRAINER}/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
#         if [ -d "$DIR" ]; then
#             echo "Oops! The results exist at ${DIR} (so skip this job)"
#         else
#            python train.py \
#             --root ${DATA} \
#             --seed ${SEED} \
#             --trainer ${TRAINER} \
#             --dataset-config-file configs/datasets/${DATASET}.yaml \
#             --config-file configs/trainers/${METHOD}/few_shot/${DATASET}.yaml  \
#             --output-dir ${DIR} \
#             TRAINER.BIOMEDCOOP.N_CTX ${NCTX} \
#             TRAINER.BIOMEDCOOP.CSC ${CSC} \
#             TRAINER.BIOMEDCOOP.CLASS_TOKEN_POSITION ${CTP} \
#             DATASET.NUM_SHOTS ${SHOTS}
#         fi
# done
#!/bin/bash

# custom config
DATA=$1
DATASET=$2
SHOTS=$3        # number of shots
MODEL=$4
fixshort=128
NCTX=4
CSC=False
CTP=end
METHOD=BiomedCoOp
TRAINER=BiomedCoOp_${MODEL}

for SEED in 42 1024 3407
do
    # -----------------------------
    # Stage 1: train attr_weights
    # -----------------------------
    DIR1=output/${DATASET}/shots_${fixshort}/${TRAINER}/stage1_nctx${NCTX}_seed${SEED}

    if [ -d "$DIR1" ]; then
        echo "Stage 1 already exists at ${DIR1}, skip."
    else
        echo "======== Running Stage 1 (attr_weights) ========"

        python train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/${METHOD}/few_shot/${DATASET}.yaml \
            --output-dir ${DIR1} \
            TRAINER.BIOMEDCOOP.N_CTX ${NCTX} \
            TRAINER.BIOMEDCOOP.CSC ${CSC} \
            TRAINER.BIOMEDCOOP.CLASS_TOKEN_POSITION ${CTP} \
            TRAINER.BIOMEDCOOP.STAGE 1 \
            DATASET.NUM_SHOTS ${fixshort}
    fi

    CKPT1=${DIR1}/prompt_learner/model.pth.tar-100

    # -----------------------------
    # Stage 2: train prompt_learner.att
    # -----------------------------
    DIR2=output/${DATASET}/shots_${SHOTS}/${TRAINER}/nctx${NCTX}_csc${CSC}_ctp${CTP}/stage2_nctx${NCTX}_seed${SEED}

    if [ -d "$DIR2" ]; then
        echo "Stage 2 already exists at ${DIR2}, skip."
    else
        echo "======== Running Stage 2 (prompt_learner.att) ========"

        python train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/${METHOD}/few_shot/${DATASET}.yaml \
            --output-dir ${DIR2} \
            TRAINER.BIOMEDCOOP.N_CTX ${NCTX} \
            TRAINER.BIOMEDCOOP.CSC ${CSC} \
            TRAINER.BIOMEDCOOP.CLASS_TOKEN_POSITION ${CTP} \
            TRAINER.BIOMEDCOOP.STAGE 2 \
            MODEL.INIT_WEIGHTS ${CKPT1} \
            DATASET.NUM_SHOTS ${SHOTS}
    fi
done