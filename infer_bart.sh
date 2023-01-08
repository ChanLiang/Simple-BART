
# exp_name=bart_base_baseline
# ckp=8

exp_name=bart_base_baseline_3turn_w_speaktokens/
ckp=7 # bart_base_baseline_3turn_w_speaktokens/


# decode_strategy=sampling
# decode_strategy=beam_search
decode_strategy=greedy

mkdir -p decode_results/$exp_name

infer_batch_size=256

echo $ckp
CUDA_VISIBLE_DEVICES=3 python bart.py \
--dumped_token ./data/ConvAI2/convai2_tokenized_multi_turn_segtoken/ \
--dataset_type convai2 \
--encoder_model facebook/bart-base \
--do_predict \
--save_result_path decode_results/$exp_name/${ckp}_${decode_strategy} \
--infer_batch_size $infer_batch_size \
--decode_strategy $decode_strategy \
--exp_name $exp_name \
--eval_epoch ${ckp} 