declare -a trials=(0 1 2)
declare -a classes=(0 1 2 3 4 5 6 7 8 9)

for i in "${trials[@]}"
do
    for j in "${classes[@]}"
    do
        python UN_retrain_scratch.py \
        --model allcnn \
        --dataset cifar10 \
        --batch-size 256 \
        --lr 0.1 \
        --epoch 50 \
        --lr_decay_milestones 30 \
        --gpu 0 \
        --trial $i \
        --uncls $j
    done
done

