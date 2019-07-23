
declare -a array=("40 10" "50 70 10")
arraylength=${#array[@]}

# use for loop to read all values and indexes
inp=20
for (( i=1; i<${arraylength}+1; i++ ));do
for topk in 500 1000; do
python run.py --layer-sizes ${array[i-1]} --top $topk --input-dim $inp --max-iterations 50
done
done
