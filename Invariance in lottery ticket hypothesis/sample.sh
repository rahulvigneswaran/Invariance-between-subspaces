
declare -a array=("40 50 10" "50 70 40 10")
arraylength=${#array[@]}


# use for loop to read all values and indexes
inp=20
for (( i=1; i<${arraylength}+1; i++ ));do
for topk in 5 10 20 30 40 500; do
for pr in 5 7 10 15; do
	python Lottery_run.py --layer-sizes ${array[i-1]} --top $topk --input-dim $inp --per $pr 
done
done
done

inp=50
declare -a array=("70 50 20" "80 50 10")
arraylength=${#array[@]}

for ((i=1; i<${arraylength}+1; i++))  ; do
for topk in 500 1000; do
python run.py --layer-sizes ${array[i-1]} --top $topk --input-dim $inp 
done
done

inp=100
declare -a array=("70 30 10" "80 20")

for ((i=1; i<${arraylength}+1; i++))  ; do
for topk in 500 1000; do
python run.py --layer-sizes ${array[i-1]} --top $topk --input-dim $inp
done
done
