#!/bin/bash

set -x

export RAY_TMPDIR=/scratch/gautschi/$USER/ray_tmp2
# export RAY_ADDRESS=""
export RAY_REDIS_PORT=7378
export RAY_DASHBOARD_PORT=9228

virl39k_train_path=PAPOGalaxy/PAPO_ViRL39K_train@train

# virl39k_train_path=/scratch/gautschi/ding432/data/PAPO_ViRL39K_train/shuffled
mmk12_val_path=PAPOGalaxy/PAPO_MMK12_test@train
geo3k_val_path=hiyouga/geometry3k@test

# train_files="['$virl39k_train_path']"
val_files="['$mmk12_val_path','$geo3k_val_path']"
# val_files="['$geo3k_val_path']"
MODEL_PATH=/scratch/gautschi/ding432/model/VLM/Qwen3-VL-8B-Instruct  # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=$virl39k_train_path \
    data.max_prompt_length=2048 \
    data.max_response_length=6144 \
    data.rollout_batch_size=512 \
    data.val_files="$val_files" \
    data.shuffle=true \
    data.format_prompt=./examples/format_prompt/reasoning.jinja \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.global_batch_size=128 \
    worker.actor.micro_batch_size_per_device_for_update=2 \
    worker.actor.micro_batch_size_per_device_for_experience=4 \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.actor.optim.lr=1e-6 \
    worker.actor.loss_type=gspo_token \
    worker.actor.loss_avg_mode=seq \
    worker.actor.clip_ratio_low=3e-4 \
    worker.actor.clip_ratio_high=4e-4 \
    algorithm.concat_responses=false \
    worker.reward.reward_type=batch \
    worker.reward.reward_function=./examples/reward_function/grpo.py:compute_score \
    worker.reward.val_reward_function=./examples/reward_function/grpo.py:compute_score \
    trainer.experiment_name=qwen3_vl_8b_virl39k_gspo_n16 \
    trainer.n_gpus_per_node=8 \
    trainer.total_epochs=5 \
    trainer.save_freq=5 \
    trainer.val_freq=5 \
    trainer.val_before_train=true \
    worker.rollout.n=16 \
    worker.rollout.val_override_config.max_tokens=6144 \
    worker.actor.optim.weight_decay=0.1 \
    worker.actor.optim.lr_warmup_steps=5 \
    algorithm.disable_kl=True \
    