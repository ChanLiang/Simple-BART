time=`date '+%F-%H:%M:%S'`

# alpha=0.01 # valid lm_loss slightly below baseline
alpha=0.003 # valid lm_loss slightly below baseline

exp_name=bart_base_baseline_3turn_w_speaktokens_kl_response_detach_finegrain_alpha${alpha}


mkdir -p log/${exp_name}
mkdir -p checkpoints/ConvAI2/bart/${exp_name}

CUDA_VISIBLE_DEVICES=2 python bart_kl.py \
--do_train \
--kl_loss \
--split_loss \
--fine_grain_kl \
--alpha $alpha \
--dataset_type convai2 \
--dumped_token ./data/ConvAI2/convai2_tokenized_multi_turn_segtoken/ \
--dumped_token_shuffle ./data/ConvAI2/shuffle/convai2_tokenized_multi_turn_segtoken_shuffle_ \
--save_model_path checkpoints/ConvAI2/bart/${exp_name} \
--learning_rate 3e-5 \
--batch_size 48 \
--total_epochs 10 \
--log_step 50 \
--valid_frequency 100 \
--print_frequency 5000 \
--warm_up_steps 1000 1>log/${exp_name}/res_${time} 2>log/${exp_name}/err_${time}


