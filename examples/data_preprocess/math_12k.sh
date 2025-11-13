set -x

stage_splits="30,40,30"
stage_noises="0.0,1.0,1.0"

python math_12k_srt.py \
    --local_dir ./data/math_12k \
    --add_self_consistency_labels \
    --stage_noises=$stage_noises \
    --stage_splits=$stage_splits \
    --dataset_path hiyouga/math12k