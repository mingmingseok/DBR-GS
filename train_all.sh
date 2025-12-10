#!/bin/bash

# ===================================================================
# SLURM 작업 설정
# ===================================================================
#SBATCH -J speedy-splat-all-scenes  
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_ce_ugrad
#SBATCH -w moana-r5                  
#SBATCH -t 1-0                       
#SBATCH -o logs/slurm-%A.out         # 로그 파일 경로

# ===================================================================
# 훈련 환경 설정
# ===================================================================
echo "========== JOB START =========="
echo "Host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# (필수) Conda 환경 활성화
source /home/jms2236/miniconda3/etc/profile.d/conda.sh # 개인 Miniconda 경로
conda activate gaussian_splatting # yml 파일에 정의된 환경 이름

# (필수) GPU 아키텍처 설정 (RTX 3090)
export TORCH_CUDA_ARCH_LIST="8.6"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "Python Path: $(which python)"
echo "PyTorch & CUDA Version:"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"
echo "==============================="

# ===================================================================
# 훈련 설정 및 반복문
# ===================================================================
# 1. Mip-NeRF 360 데이터셋이 있는 상위 경로를 설정합니다.
DATASET_ROOT="/data/jms2236/repos/speedy-splat/mipnerf360" 

# 2. 훈련된 모델들을 저장할 상위 경로를 설정합니다.
MODELS_ROOT="/data/jms2236/repos/speedy-splat/models_ss" 

# 3. 훈련시킬 씬들의 목록을 배열로 만듭니다.
SCENES=("bonsai")

#"bicycle" "bonsai" "counter" "garden" "kitchen" "room" "stump" "flowers" "treehill"



# 4. 각 씬에 대해 반복문을 실행합니다.
for scene in "${SCENES[@]}"
do
    echo " "
    echo "----------- Training scene: $scene -----------"
    
    # 훈련 시작 시간 기록
    start_time=$(date +%s)

    # 5. 각 씬에 맞는 환경 변수를 설정합니다.
    export SCENE_DATA_PATH="$DATASET_ROOT/$scene"
    export SCENE_MODEL_PATH="$MODELS_ROOT/$scene"
    
    # 6. 훈련 스크립트(train.sh)를 실행합니다.
    bash train.sh
    
    # 훈련 종료 시간 기록 및 경과 시간 계산
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    # 분과 초로 변환하여 출력
    minutes=$((duration / 60))
    seconds=$((duration % 60))
    
    echo "----------- Finished training $scene in $minutes minutes and $seconds seconds -----------"

    # bash compute_scene_metrics.sh
done

echo " "
echo "================================================="
echo "            ALL SCENES TRAINED!                  "
echo "================================================="