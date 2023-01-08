
# for((id=1;id<=60;id++))
for((id=61;id<=119;id++))
# for((id=1;id<=1;id++))
do
    echo $id
    python preprocess_permutation.py \
    --dataset_type convai2 \
    --testset /misc/kfdata01/kf_grp/lchen/data/test_self_original.txt \
    --encoder_model_name_or_path 'facebook/bart-base' \
    --max_source_length 160 \
    --max_target_length 40 \
    --multi_turn 3 \
    --permutation $id
done