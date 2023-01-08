 time=`date '+%F-%H:%M:%S'`
 exp_name=bart_base_baseline

 mkdir -p log/${exp_name}
 mkdir -p checkpoints/ConvAI2/bart/${exp_name}

 CUDA_VISIBLE_DEVICES=1 python bart.py \
 --do_train \
 --dataset_type convai2 \
 --dumped_token ./data/ConvAI2/convai2_tokenized/ \
 --save_model_path checkpoints/ConvAI2/bart/${exp_name} \
 --learning_rate 3e-5 \
 --batch_size 128 \
 --total_epochs 10 \
 --warm_up_steps 500 \
 --valid_frequency 200 \
#  --warm_up_steps 2000 1>log/${exp_name}/res_${time} 2>log/${exp_name}/err_${time}
