declare -a trials=(0 1 2)
declare -a classes=(0 1 2 3 4 5 6 7 8 9)
declare -a strategies=("BlockF" "GKT" "IS_GKT" "PF" "IS_PF")

echo "Running CIFAR-10 allcnn allcnn"
for i in "${trials[@]}"
do
    for j in "${classes[@]}"
    do
        for k in "${strategies[@]}"
        do
            python UN_datafree_kd.py \
            --method dfq \
            --dataset cifar10 \
            --batch_size 256 \
            --teacher allcnn \
            --student allcnn \
            --lr 0.05 \
            --epochs 200 \
            --warmup 2 \
            --kd_steps 10 \
            --ep_steps 50 \
            --g_steps 1 \
            --lr_g 1e-3 \
            --adv 1 \
            --T 1 \
            --bn 1 \
            --oh 1 \
            --act 0 \
            --balance 1 \
            --gpu 0 \
            --seed 0 \
            --trial $i \
            --uncls $j \
            --strategy $k
        done
    done
done

