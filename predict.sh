version='dvToF_norm_T_weight'
version_out='dvToF_norm_T_weight'

model_path="models/$version/checkpoint_best.pth"

iq_out="results/$version_out/iq"
mu_out="results/$version_out/mu"
d_out="results/$version_out/depth"

list_path="dataset/list/test.txt"

python predict.py \
    -in 'dataset' \
    --dev 2 \
    -ls $list_path \
    -out $iq_out \
    -out_mu $mu_out \
    -m $model_path 

python IQ2corr.py \
   --version $version_out \
   --list_path $list_path
