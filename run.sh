# /bin/bash

# transductive pretrain command

# unbuffer torchrun --standalone --nnodes=1 --nproc_per_node=1 pretrain.py \
# --data wn18rr --batch_size 48 --plm bert  --test_batch_size 16 --use_description \
# --optim adamw --scheduler no_scheduler \
# --nbf_lr 1e-3 \
# --finetune \


# transductive distillation command

# unbuffer torchrun --standalone --nnodes=1 --nproc_per_node=1 main.py \
# --data wn18rr --batch_size 48 --plm bert  --test_batch_size 16 --use_description \
# --optim adamw --scheduler constant --scheduler_step 1800 \
# --bert_lr 4e-5 --no_bert_lr 2e-3  --temperature 1e+10 \
# --finetune \
# --teacher_load_path your_nbfmodel_load_path


# transductive evaluation

# unbuffer torchrun --standalone --nnodes=1 --nproc_per_node=1 main.py \
# --data wn18rr --batch_size 48 --plm bert  --test_batch_size 16 --use_description \
# --optim adamw --scheduler constant --scheduler_step 1800 \
# --bert_lr 4e-5 --no_bert_lr 2e-3  --temperature 1e+10 \
# --link_prediction \
# --load_path your_model_path




  
# inductive pretrain command

# unbuffer torchrun --standalone --nnodes=1 --nproc_per_node=1 pretrain_ind.py \
# --data fb15k-237 --version v4 --batch_size 64 --plm bert  --test_batch_size 48 --use_description \
# --optim adamw --scheduler no_scheduler --scheduler_step 2900 \
# --nbf_lr 1e-3 --temperature 1e+10 \
# --finetune \

# inductive distillation command

# unbuffer torchrun --standalone --nnodes=1 --nproc_per_node=1 main_ind.py \
# --data fb15k-237 --version v4 --batch_size 64 --plm bert  --test_batch_size 48 --use_description \
# --optim adamw --scheduler no_scheduler --scheduler_step 2900 \
# --bert_lr 4e-5 --no_bert_lr 2e-3 --temperature 1e+10 \
# --finetune \
# --teacher_load_path your_nbfmodel_load_path

# inductive evaluation command

# unbuffer torchrun --standalone --nnodes=1 --nproc_per_node=1 main_ind.py \
# --data fb15k-237 --version v4 --batch_size 64 --plm bert  --test_batch_size 48 --use_description \
# --optim adamw --scheduler no_scheduler --scheduler_step 2900 \
# --bert_lr 4e-5 --no_bert_lr 2e-3 --temperature 1e+10 \
# --link_prediction \
# --load_path your_model_path
