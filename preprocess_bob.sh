 python preprocess.py --dataset_type convai2 \
 --trainset /misc/kfdata01/kf_grp/lchen/data/persona/parlai/parl.ai/downloads/personachat/personachat/train_self_original_wo_candidates.txt \
 --testset /misc/kfdata01/kf_grp/lchen/data/persona/parlai/parl.ai/downloads/personachat/personachat/valid_self_original_wo_candidates.txt \
 --nliset ../data/mnli/multinli_1.0/ \
 --encoder_model_name_or_path ../bert-base-models \
 --max_source_length 64 \
 --max_target_length 32