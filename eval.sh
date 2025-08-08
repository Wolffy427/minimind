nohup lm_eval --model minimind \
   --model_args pretrained=./model/hf/minimind_full_sft \
   --tasks ceval-valid,cmmlu,aclue,tmmluplus \
   --batch_size 16 \
   --output_path ./evaluate/minimind/all.json \
   --apply_chat_template \
   --trust_remote_code \
   > logs/eval_sh.log 2>&1 &
echo $! > ./lm-eval.pid

echo "进程已启动，PID: $(cat ./lm-eval.pid)"



# nohup lm_eval --model hf \
#    --model_args pretrained=./model/ms/Qwen/Qwen2.5-7B \
#    --tasks ceval-valid,cmmlu,aclue,tmmluplus \
#    --output_path ./evaluate/qwen/all.json \
#    --batch_size 4 \
#    --apply_chat_template \
#    --trust_remote_code \
#    > logs/eval_sh.log 2>&1 &
# echo $! > ./lm-eval.pid

# echo "进程已启动，PID: $(cat ./lm-eval.pid)"

#      --wandb_args project=lm-eval,name=qwen-all \
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
