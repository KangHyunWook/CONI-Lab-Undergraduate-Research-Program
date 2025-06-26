#below approach brings consistent accuracy for the corresponding learning rate	
	
python main.py --learning_rate 0.001 --w_mode 'w' --save_path 'consistent_lr_acc.csv' 

lr_list=(0.003 0.005 0.007 0.009)

for lr in "${lr_list[@]}"
do
    python main.py --learning_rate "$lr" --w_mode 'a' --save_path 'consistent_lr_acc.csv'
done



lr=0.001

python main.py --learning_rate $lr --w_mode 'w' --save_path 'inconsistent_lr_acc.csv' --optimizer RMSprop
#below approach brings inconsistent accuracy for the corresponding learning rate, since floating points numbers are represented approximately with 0 and 1s

for i in {1..5}
do
	((lr=lr+0.002)) | bc
	python main.py --learning_rate $lr --w_mode 'a' --save_path 'inconsistent_lr_acc.csv' --optimizer RMSprop
done
	


