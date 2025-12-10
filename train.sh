data_path=$SCENE_DATA_PATH
model_path=$SCENE_MODEL_PATH
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
python train_density.py \
  -s ${data_path} \
  -m ${model_path} \ 
  --eval
