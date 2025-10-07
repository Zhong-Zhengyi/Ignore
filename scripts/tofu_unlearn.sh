#!/bin/bash


export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

# 设置分布式训练环境变量
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=1800
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

models=(
    # "phi-1_5"
    "Qwen3-1.7B"
    # "Phi-3.5-mini-instruct"
    # "Llama-3.2-3B-Instruct"
)
trainers_experiments=(
    # "MyUnlearning unlearn/tofu/idk.yaml"
    # "MyUnlearning unlearn/tofu/default.yaml"
    # "MyUnlearning unlearn/tofu/generate.yaml"
    "embodied_GA unlearn/tofu/default.yaml"
    # "GradAscent unlearn/tofu/default.yaml"
    # "GradDiff unlearn/tofu/default.yaml"
    # "NPO unlearn/tofu/default.yaml"
    # "DPO unlearn/tofu/idk.yaml"
    # "RMU  unlearn/tofu/default.yaml"
)
splits=(
    "forget05 holdout05 retain95 7"
    # "forget01 holdout01 retain99 2"
    # "forget10 holdout10 retain90 13"
)

per_device_train_batch_size=2 # on two gpus would make effective batch size 32
gradient_accumulation_steps=8

#####################################################################################################################
########################################### Unlearn TOFU models #####################################################
#####################################################################################################################

for split in "${splits[@]}"; do
    forget_split=$(echo $split | cut -d' ' -f1)
    holdout_split=$(echo $split | cut -d' ' -f2)
    retain_split=$(echo $split | cut -d' ' -f3)
    gap=$(echo $split | cut -d' ' -f4)

    for model in "${models[@]}"; do
        for trainer_experiment in "${trainers_experiments[@]}"; do
            trainer=$(echo $trainer_experiment | cut -d' ' -f1)
            experiment=$(echo $trainer_experiment | cut -d' ' -f2)
            
            task_name=tofu_${model}_${forget_split}_${trainer} 
            model_path=saves/finetune/tofu_${model}_full
            echo ${task_name}: Unlearning ${model_path} using ${trainer}

            # Unlearn
            CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port $MASTER_PORT \
            src/train.py --config-name=unlearn.yaml \
            experiment=${experiment} \
            trainer=${trainer} \
            task_name=${task_name} \
            model=${model} \
            forget_split=${forget_split} \
            retain_split=${retain_split} \
            model.model_args.pretrained_model_name_or_path=${model_path} \
            retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json \
            trainer.args.per_device_train_batch_size=$per_device_train_batch_size \
            trainer.args.gradient_accumulation_steps=$gradient_accumulation_steps \
            trainer.args.ddp_find_unused_parameters=true \
            trainer.args.num_train_epochs=5 \
            # trainer.args.gradient_checkpointing=true \

            # Eval
            for i in 1 2 3 4 5; do
                ckpt=$((i * gap))
                PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 python src/eval.py \
                experiment=eval/tofu/default.yaml \
                trainer=${trainer} \
                forget_split=${forget_split} \
                holdout_split=${holdout_split} \
                retain_split=${retain_split} \
                model=${model} \
                task_name=${task_name} \
                model.model_args.pretrained_model_name_or_path=saves/unlearn/${task_name}/checkpoint-${ckpt}\
                paths.output_dir=saves/unlearn/${task_name}/evals \
                retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json
            done
        done
    done
done

# model.model_args.pretrained_model_name_or_path=saves/unlearn/${task_name}/checkpoint-${ckpt}\