gpu=$1
prefix=$2
ptname=$3
savedir=$4
python main.py --data-dir dataset --data-file ${ptname}_tcr_${prefix}.npy --target-file ${ptname}_targets.npy --dataset tcr --model tcr  --emb_size 10 --save-dir $savedir --epoch 500 --gpu-selection $gpu  --mlp-layers-size 50 25 10 --load-folder $savedir --reload True 


