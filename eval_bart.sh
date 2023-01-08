
exp_name=bart_base_baseline

mkdir -p decode_results/$exp_name

infer_batch_size=512

# for ckp in 1 2 3 
for ckp in 3 4 5 6 7 8 9
# for ckp in 8 
do
 echo $ckp
 CUDA_VISIBLE_DEVICES=1 python bart.py \
 --dumped_token ./data/ConvAI2/convai2_tokenized/ \
 --dataset_type convai2 \
 --encoder_model facebook/bart-base \
 --do_evaluation \
 --do_predict \
 --save_result_path decode_results/$exp_name/${ckp} \
 --infer_batch_size $infer_batch_size \
 --exp_name $exp_name \
 --eval_epoch ${ckp} 
#  --eval_epoch ${ckp} 1>log/eval/res_${ckp} 2>log/eval/err_${ckp}
done