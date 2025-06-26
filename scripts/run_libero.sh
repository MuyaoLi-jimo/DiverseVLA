#! /bin/bash

python  diversevla/evaluate/evaluate_generalization.py \
    --pretrained_checkpoint "/share/lmy/models/openvla-7b-finetuned-libero-object/" \
    --task_suite_name "libero_object" \
    --num_trials_per_task 3 \
    --change_scope "one"

python  diversevla/evaluate/evaluate_generalization.py \
    --pretrained_checkpoint "/share/lmy/models/openvla-7b-finetuned-libero-object/" \
    --task_suite_name "libero_object" \
    --num_trials_per_task 3 \
    --change_scope "two"

python  diversevla/evaluate/evaluate_generalization.py \
    --pretrained_checkpoint "/share/lmy/models/openvla-7b-finetuned-libero-object/" \
    --task_suite_name "libero_object" \
    --num_trials_per_task 3 \
    --change_scope "all"