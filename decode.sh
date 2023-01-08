

CUDA_VISIBLE_DEVICES=0 python bertoverbert.py --dumped_token ./data/ConvAI2/convai2_tokenized/ \
 --dataset_type convai2 \
 --encoder_model ../bert-base-models \
 --do_predict \
 --save_result_path decode_results/test_result_7.tsv \
 --eval_epoch 7
