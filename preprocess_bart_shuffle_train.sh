
# for((id=1;id<=10;id++))
# for((id=11;id<=20;id++))
for id in 2
do
    echo $id
    python preprocess_shuffle_train.py \
    --dataset_type convai2 \
    --trainset /misc/kfdata01/kf_grp/lchen/data/train_self_original.txt \
    --encoder_model_name_or_path 'facebook/bart-base' \
    --max_source_length 128 \
    --max_target_length 32 \
    --multi_turn 3 \
    --shuffle $id
done