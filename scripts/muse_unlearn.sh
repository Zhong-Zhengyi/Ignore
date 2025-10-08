#!/bin/bash

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"


# export NCCL_DEBUG=INFO
# export NCCL_TIMEOUT=3600
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
# export CUDA_LAUNCH_BLOCKING=1
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


per_device_train_batch_size=1
gradient_accumulation_steps=1


model="Llama-2-7b-hf" #phi-1_5Llama-3.2-3B-Instruct"Qwen3-1.7B"

data_splits=(
    "News 204"
    # "Books 277"
)

trainers=(
    "Ignore"
    # "GradAscent"
    # "GradDiff"
    # "NPO"
    # "SimNPO"
)

# #########################################################
# #################### MUSE Unlearning ####################
# #########################################################


# for split in "${data_splits[@]}"; do
#     data_split=$(echo $split | cut -d' ' -f1)
#     gap=$(echo $split | cut -d' ' -f2)
#     for trainer in "${trainers[@]}"; do
#         task_name=muse_${model}_${data_split}_${trainer}
        
# #         CUDA_VISIBLE_DEVICES=0 python src/train.py --config-name=unlearn.yaml \
#         CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port $MASTER_PORT \
#         src/train.py --config-name=unlearn.yaml \
#         experiment=unlearn/muse/default.yaml \
#         model=${model} \
#         data_split=${data_split} \
#         trainer=${trainer} \
#         task_name=${task_name} \
#         model.tokenizer_args.pretrained_model_name_or_path=NousResearch/Llama-2-7b-hf \
#         retain_logs_path=saves/eval/muse_${model}_${data_split}_retrain/MUSE_EVAL.json \
#         trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
#         trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
#         trainer.args.ddp_find_unused_parameters=true \
#         # trainer.args.gradient_checkpointing=true

#         for i in 1 2 3 4 5; do
#             ckpt=$((i * gap))
#             CUDA_VISIBLE_DEVICES=0 python src/eval.py \
#             experiment=eval/muse/default.yaml \
#             data_split=${data_split} \
#             task_name=${task_name} \
#             model=${model} \
#             model.model_args.pretrained_model_name_or_path=saves/unlearn/${task_name}/checkpoint-${ckpt} \
#             model.tokenizer_args.pretrained_model_name_or_path=NousResearch/Llama-2-7b-hf \
#             paths.output_dir=saves/unlearn/${task_name}/evals \
#             retain_logs_path=saves/eval/muse_${model}_${data_split}_retrain/MUSE_EVAL.json
#         done
#     done
# done



# #########################################################
# ########### MUSE News Unlearning Scalability ############
# #########################################################


for split in "${data_splits[@]}"; do
    data_split=$(echo $split | cut -d' ' -f1)
    gap=$(echo $split | cut -d' ' -f2)
    for trainer in "${trainers[@]}"; do
        for scal in "forget_1" "forget_2" "forget_3" "forget_4"; do
            task_name=muse_${model}_${data_split}_${trainer}_scal_${scal}

            CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port $MASTER_PORT \
            src/train.py --config-name=unlearn.yaml \
            experiment=unlearn/muse/scalability.yaml \
            model=${model} \
            data_split=${data_split} \
            forget_split=${scal} \
            trainer=${trainer} \
            task_name=${task_name} \
            model.tokenizer_args.pretrained_model_name_or_path=NousResearch/Llama-2-7b-hf \
            retain_logs_path=saves/eval/muse_${model}_${data_split}_retrain/MUSE_EVAL.json \
            trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
            trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
            trainer.args.ddp_find_unused_parameters=true \
            # trainer.args.gradient_checkpointing=true

            for i in 1 2 3 4 5; do
                ckpt=$((i * gap))
                CUDA_VISIBLE_DEVICES=0 python src/eval.py \
                experiment=eval/muse/default.yaml \
                data_split=${data_split} \ 
                task_name=${task_name} \
                model=${model} \
                model.tokenizer_args.pretrained_model_name_or_path=NousResearch/Llama-2-7b-hf \
                model.model_args.pretrained_model_name_or_path=saves/unlearn/${task_name}/checkpoint-${ckpt} \
                paths.output_dir=saves/unlearn/${task_name}/evals \
                retain_logs_path=saves/eval/muse_${model}_${data_split}_retrain/MUSE_EVAL.json
            done
        done
    done
done



#########################################################
########### MUSE News Unlearning sustainability #########
#########################################################


# for data_split in "${data_splits[@]}"; do
#     for trainer in "${trainers[@]}"; do
#         model_path=muse-bench/MUSE-${data_split}_target
#         for sust in "forget_1" "forget_2" "forget_3" "forget_4"; do
            
#             task_name=muse_${model}_${data_split}_${trainer}_sust_${sust}

#             CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port $MASTER_PORT \
#             src/train.py --config-name=unlearn.yaml \
#             experiment=unlearn/muse/sustainabilty.yaml \
#             model=${model} \
#             model.model_args.pretrained_model_name_or_path=${model_path} \
#             data_split=${data_split} \
#             trainer=${trainer} \
#             task_name=${task_name} \
#             model.tokenizer_args.pretrained_model_name_or_path=NousResearch/Llama-2-7b-hf \
#             retain_logs_path=saves/eval/muse_${model}_${data_split}_retrain/MUSE_EVAL.json \
#             trainer.args.per_device_train_batch_size=1 \
#             trainer.args.gradient_accumulation_steps=8 \
#             trainer.args.ddp_find_unused_parameters=true \
#             trainer.args.gradient_checkpointing=true

#             CUDA_VISIBLE_DEVICES=0 python src/eval.py \
#             experiment=eval/muse/default.yaml \
#             data_split=${data_split} \ 
#             task_name=${task_name} \
#             model=${model} \
#             model.tokenizer_args.pretrained_model_name_or_path=NousResearch/Llama-2-7b-hf \
#             model.model_args.pretrained_model_name_or_path=saves/unlearn/${task_name} \
#             paths.output_dir=saves/unlearn/${task_name}/evals \
#             retain_logs_path=saves/eval/muse_${model}_${data_split}_retrain/MUSE_EVAL.json

#             model_path=saves/unlearn/${task_name}
#         done
#     done
# done