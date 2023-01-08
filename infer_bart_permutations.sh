
exp_name=bart_base_baseline_3turn_w_speaktokens/
ckp=7 # bart_base_baseline_3turn_w_speaktokens/


decode_strategy=sampling
# decode_strategy=beam_search
# decode_strategy=greedy

mkdir -p decode_results/$exp_name/permutations/

infer_batch_size=256

echo $ckp
# for((id=1;id<=1;id++))
# for((id=2;id<=60;id++))
for((id=61;id<=119;id++))
do
    CUDA_VISIBLE_DEVICES=2 python bart.py \
    --dumped_token ./data/ConvAI2/permutations/convai2_tokenized_multi_turn_segtoken_permutation_${id}/ \
    --dataset_type convai2 \
    --encoder_model facebook/bart-base \
    --do_predict \
    --save_result_path decode_results/$exp_name/permutations/ckp${ckp}_${decode_strategy}_permutations_${id} \
    --infer_batch_size $infer_batch_size \
    --decode_strategy $decode_strategy \
    --exp_name $exp_name \
    --eval_epoch ${ckp} 
done