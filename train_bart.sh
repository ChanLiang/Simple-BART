 time=`date '+%F-%H:%M:%S'`
 exp_name=bart_base_baseline

 CUDA_VISIBLE_DEVICES=1 python encoder-decoder.py \
 --do_train \
 --dataset_type convai2 \
 --dumped_token ./data/ConvAI2/convai2_tokenized/ \
 --save_model_path checkpoints/ConvAI2/bart \
 --learning_rate 2e-5 \
 --batch_size 64 \
 --total_epochs 10 \
 --warm_up_steps 2000 1>log/res_${exp_name}_${time} 2>log/err_${exp_name}_${time}
