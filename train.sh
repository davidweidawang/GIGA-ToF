version='dvToF_norm_T_weight'

python -u train.py \
    -b 32 \
    --dev 0 \
    -lr 2.5e-3 \
    --weight_decay 1e-5 \
    -out "./models/$version" \
    -d "./results/$version/debug" \
    -in './dataset' \
    #-m "/models/checkpoint_best.pth" \
    -e 400 | tee giga_norm_T_weight.txt