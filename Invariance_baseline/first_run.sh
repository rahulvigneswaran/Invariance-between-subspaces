declare -a array=("10 15 20" "10 15 20")
arraylength=${#array[@]}


# use for loop to read all values and indexes
inp=20
for (( i=1; i<${arraylength}+1; i++ ));do
for topk in 5 10 20 30 40 500; do
#for pr in 5; do
        python run.py --layer-sizes ${array[i-1]} --top $topk --input-dim $inp #--per $pr
#done
done
done


