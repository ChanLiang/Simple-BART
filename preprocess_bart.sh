 python preprocess.py --dataset_type convai2 \
 --trainset /misc/kfdata01/kf_grp/lchen/data/train_self_original.txt \
 --validset /misc/kfdata01/kf_grp/lchen/data/valid_self_original.txt \
 --testset /misc/kfdata01/kf_grp/lchen/data/test_self_original.txt \
 --encoder_model_name_or_path 'facebook/bart-base' \
 --max_source_length 128 \
 --max_target_length 32