declare -a array=("40 50 10" "50 70 40 10")
arraylength=${#array[@]}


# use for loop to read all values and indexes
inp=20
for (( i=1; i<${arraylength}+1; i++ ));do
for topk in 5 10 20 30 40 500; do
for pr in 15; do
        python server_lottery_run.py --layer-sizes ${array[i-1]} --top $topk --input-dim $inp --per $pr
done
done
done


