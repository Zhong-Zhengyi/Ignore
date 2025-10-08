#!/bin/bash

export CUDA_VISIBLE_DEVICES=0


models=(
    # "phi-1_5"
    "Qwen3-1.7B"
    # "Phi-3.5-mini-instruct"
    # "Llama-3.2-3B-Instruct"
)
trainers_experiments=(
    # "Ignore unlearn/tofu/default.yaml"
    # "GradAscent unlearn/tofu/default.yaml"
    # "GradDiff unlearn/tofu/default.yaml"
    "NPO unlearn/tofu/default.yaml"
    # "DPO unlearn/tofu/idk.yaml"
    # "RMU  unlearn/tofu/default.yaml"
)
splits=(
    "forget01 holdout01 retain99 2"
    "forget05 holdout05 retain95 7"
    "forget10 holdout10 retain90 13"
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
            augmented_data_path=saves/tofu_${model}_${forget_split}_embodied_GA_augmented_data.json
            
            task_name=tofu_${model}_${forget_split}_${trainer}_superficial_forgetting
            model_path=saves/unlearn/tofu_${model}_${forget_split}_${trainer}
            echo ${task_name}: Unlearning ${model_path} using ${trainer}

            
            python src/eval.py \
                experiment=eval/tofu/default.yaml \
                augmented_data_path=${augmented_data_path} \
                trainer=${trainer} \
                paths.output_dir=saves/unlearn/${task_name}\
                forget_split=${forget_split} \
                holdout_split=${holdout_split} \
                retain_split=${retain_split} \
                model=${model} \
                model.model_args.pretrained_model_name_or_path=${model_path}\
                task_name=${task_name} \

        done
    done
done
