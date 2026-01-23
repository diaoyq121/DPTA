# #!/bin/bash

# # 配置参数
# DATA_PATH="data"
# DATASET="kvasir"
# # MODEL="BiomedCopy"
# MODEL="PMCCLIP"
# SHOTS=(1 2 4 8 16)
# GPUS=(0 0 0 0 0) # 指定要as使用的 GPU 编号

# # 循环并行启动任务
# for i in "${!SHOTS[@]}"; do
#     shot=${SHOTS[$i]}
#     gpu=${GPUS[$i]}
    
#     echo "$DATASET 正在 GPU $gpu 上启动任务: Shot $shot"
    
#     # 使用 & 将任务放入后台执行，并重定向日志防止控制台混乱
#     CUDA_VISIBLE_DEVICES=$gpu bash scripts/biomedcoop/few_shot.sh \
#         $DATA_PATH $DATASET $shot $MODEL > log_shot_${shot}.txt 2>&1 &
    
#     # 稍微休眠 2 秒，防止多个任务同时读取数据造成 IO 阻塞
#     sleep 2
# done

# echo "----------------------------------------------------"
# echo "所有并行任务已在后台启动！"
# echo "你可以使用 'nvidia-smi' 查看显卡状态。"
# echo "可以使用 'tail -f log_shot_1.txt' 等命令实时查看进度。"
# echo "----------------------------------------------------"

# # 等待所有后台任务完成
# wait
# echo "所有任务执行完毕！"
#!/bin/bash

# 配置参数
DATA_PATH="data"
# DATASETS=("btmri" "covid" "ctkidney")
# DATASETS=("lungcolon" "octmnist" "kvasir" "retina")
# DATASETS=("chmnist" "kneexray")
DATASETS=("busi")


MODEL="BiomedCopy"
SHOTS=(1 2 4 8 16)

# 指定可用的 GPU 列表 (例如你有 4 张卡)
GPUS=(0 1 2 3)
NUM_GPUS=${#GPUS[@]}

TASK_COUNT=0

for DATASET in "${DATASETS[@]}"; do
    for shot in "${SHOTS[@]}"; do
        # 自动循环分配 GPU (例如第 5 个任务会回到 GPU 0)
        gpu_idx=$((TASK_COUNT % NUM_GPUS))
        gpu=${GPUS[$gpu_idx]}
        
        echo "启动任务: [Dataset: $DATASET] [Shot: $shot] -> GPU $gpu"
        
        # 运行并将日志保存到对应数据集的文件夹
        mkdir -p logs/$DATASET
        CUDA_VISIBLE_DEVICES=$gpu bash scripts/biomedcoop/few_shot.sh \
            $DATA_PATH $DATASET $shot $MODEL > logs/$DATASET/shot_${shot}.log 2>&1 &
        
        TASK_COUNT=$((TASK_COUNT + 1))
        
        # 每启动一个任务稍微休眠，减轻磁盘 I/O 压力
        sleep 3
    done
done

echo "----------------------------------------------------"
echo "已启动 $TASK_COUNT 个任务，分布在 ${NUM_GPUS} 块 GPU 上。"
echo "正在后台运行中... 请使用 'tail -f logs/${DATASETS[0]}/shot_1.log' 查看进度。"
echo "----------------------------------------------------"

# wait

# DATASETS=("lungcolon" "octmnist" "kvasir" "retina")
# for DATASET in "${DATASETS[@]}"; do
#     for shot in "${SHOTS[@]}"; do
#         # 自动循环分配 GPU (例如第 5 个任务会回到 GPU 0)
#         gpu_idx=$((TASK_COUNT % NUM_GPUS))
#         gpu=${GPUS[$gpu_idx]}
        
#         echo "启动任务: [Dataset: $DATASET] [Shot: $shot] -> GPU $gpu"
        
#         # 运行并将日志保存到对应数据集的文件夹
#         mkdir -p logs/$DATASET
#         CUDA_VISIBLE_DEVICES=$gpu bash scripts/biomedcoop/few_shot.sh \
#             $DATA_PATH $DATASET $shot $MODEL > logs/$DATASET/shot_${shot}.log 2>&1 &
        
#         TASK_COUNT=$((TASK_COUNT + 1))
        
#         # 每启动一个任务稍微休眠，减轻磁盘 I/O 压力
#         sleep 3
#     done
# done

# echo "----------------------------------------------------"
# echo "已启动 $TASK_COUNT 个任务，分布在 ${NUM_GPUS} 块 GPU 上。"
# echo "正在后台运行中... 请使用 'tail -f logs/${DATASETS[0]}/shot_1.log' 查看进度。"
# echo "----------------------------------------------------"

wait
echo "所有数据集的 Few-shot 任务处理完毕！"
