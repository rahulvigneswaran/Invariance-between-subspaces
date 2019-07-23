
declare -a array=("20 10" "30 40")
arraylength=${#array[@]}

# use for loop to read all values and indexes
inp=10
for (( i=1; i<${arraylength}+1; i++ ));do
for topk in 50 100; do
python run.py --layer-sizes ${array[i0]} --top $topk --input-dim $inp 
done
done

