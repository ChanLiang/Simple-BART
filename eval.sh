
for ckp in 5 6 7 8 9 10
do
 CUDA_VISIBLE_DEVICES=1 python bertoverbert.py --dumped_token ./data/ConvAI2/convai2_tokenized/ \
 --dataset_type convai2 \
 --encoder_model ../bert-base-models  \
 --do_evaluation --do_predict \
 --save_result_path decode_results/test_result_${ckp}.tsv \
 --eval_epoch ${ckp} 1>log/eval/res_${ckp} 2>log/eval/err_${ckp}
done