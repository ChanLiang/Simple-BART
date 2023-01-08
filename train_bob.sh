 CUDA_VISIBLE_DEVICES=1 python bob.py --do_train \
 --encoder_model ../bert-base-models \
 --decoder_model ../bert-base-models \
 --decoder2_model ../bert-base-models \
 --save_model_path checkpoints/ConvAI2/bertoverbert --dataset_type convai2 \
 --dumped_token ../BoB/data/ConvAI2/convai2_tokenized/ \
 --learning_rate 7e-6 \
 --batch_size 32
