DATA_PATH="D:\datasets\subject1-data"

size=8
python main.py --hidden_size $size --w_mode 'w' --dataset_dir $DATA_PATH

while [ $size -lt 16 ]
	do
		((size=size+1))
		python main.py --hidden_size $size --w_mode 'a' --dataset_dir $DATA_PATH
	done
	
	