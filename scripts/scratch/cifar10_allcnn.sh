declare -a trials=(0 1 2)

for i in "${trials[@]}"
do
    python UN_train_scratch.py \
    --model allcnn \
    --dataset cifar10 \
    --batch-size 256 \
    --lr 0.1 \
    --epoch 50 \
    --lr_decay_milestones 30 \
    --gpu 0 \
    --trial $i
done

