version='dvToF_norm_T_weight'

list_path="Path/to/pbrt/dataset/list/test.txt"
metric_save="results/result_metrics/$version"

python mat2depth.py \
    --version $version \
    --list_path $list_path \
    --visualize False

python eval.py \
    -out $metric_save \
    -in "results/$version/depth" \
    -gt "Path/to/pbrt/dataset/gt_depth" \
    -flow "Path/to/pbrt/dataset/optic_flow" \
    --list_path $list_path \
    -v "2.0"