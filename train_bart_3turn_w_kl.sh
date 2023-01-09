time=`date '+%F-%H:%M:%S'`

# alpha=0.3
alpha=0.0

#  exp_name=bart_base_baseline_3turn
#  exp_name=bart_base_baseline_3turn_w_speaktokens

# exp_name=bart_base_baseline_3turn_w_speaktokens_kl_response_detach_alpha${alpha}
exp_name=bart_base_baseline_3turn_w_speaktokens_kl_response_alpha${alpha}



mkdir -p log/${exp_name}
mkdir -p checkpoints/ConvAI2/bart/${exp_name}

CUDA_VISIBLE_DEVICES=2 python bart_kl.py \
--do_train \
--dataset_type convai2 \
--dumped_token ./data/ConvAI2/convai2_tokenized_multi_turn_segtoken/ \
--dumped_token_shuffle ./data/ConvAI2/shuffle/convai2_tokenized_multi_turn_segtoken_shuffle_ \
--save_model_path checkpoints/ConvAI2/bart/${exp_name} \
--learning_rate 3e-5 \
--batch_size 48 \
--total_epochs 10 \
--valid_frequency 500 \
--print_frequency 1000 \
--kl_loss \
--warm_up_steps 1000 1>log/${exp_name}/res_${time} 2>log/${exp_name}/err_${time}

#--split_loss \

