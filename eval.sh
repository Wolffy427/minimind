export CUDA_VISIBLE_DEVICES=2
# #    --model_args pretrained=Qwen/Qwen2.5-0.5B-Instruct \
export HF_ENDPOINT=https://hf-mirror.com
HF_ENDPOINT=https://hf-mirror.com lm_eval --model minimind \
   --model_args pretrained=./model/hf/minimind_full_sft \
   --tasks cmmlu \
   --output_path ./evaluate/minimind/cmmlu \
   --wandb_args project=lm-eval,name=minimind-cmmlu \
   --apply_chat_template False
   # --tasks ceval-valid,cmmlu,aclue,tmmluplus \
# export CUDA_VISIBLE_DEVICES=1,2 
# export HF_ENDPOINT=https://hf-mirror.com 
# lm_eval --model hf \
#    --model_args pretrained=Qwen/Qwen2.5-0.5B-Instruct \
#    --tasks aclue \
#    --output_path ./evaluate/aclue \
#    --wandb_args project=lm-eval,name=qwen-aclue \
#    --apply_chat_template True
# 设置环境变量
# export CUDA_VISIBLE_DEVICES=1,2 
# export HF_ENDPOINT=https://hf-mirror.com 

# # 验证变量
# echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
# echo "HF_ENDPOINT: $HF_ENDPOINT"

# # 运行命令
# HF_ENDPOINT=https://hf-mirror.com  lm_eval --model hf \
#    --model_args pretrained=Qwen/Qwen2.5-0.5B-Instruct \
#    --tasks cmmlu  \
#    --output_path ./evaluate/cmmlu \
#    --wandb_args project=lm-eval,name=qwen-cmmlu \
#    --apply_chat_template True