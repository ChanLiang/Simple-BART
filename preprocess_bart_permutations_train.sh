
# for((id=0;id<=119;id++))
for id in 1 3 5 7 9 40 41 55 78 76 99 101 119 114 88 66 22 33 49 82 87
do
    echo $id
    python preprocess_permutation_train.py \
    --dataset_type convai2 \
    --trainset /misc/kfdata01/kf_grp/lchen/data/train_self_original.txt \
    --encoder_model_name_or_path 'facebook/bart-base' \
    --max_source_length 160 \
    --max_target_length 40 \
    --multi_turn 3 \
    --permutation $id
done