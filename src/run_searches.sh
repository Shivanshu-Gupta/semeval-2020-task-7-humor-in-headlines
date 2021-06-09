log_dir=../outputs/logs/
mkdir -p ${log_dir}/task1 ${log_dir}/task2

task_id=1
hp_space=base2
num_epochs=20

python search.py -o --comet --task $task_id --hp_space ${hp_space} --num_epochs ${num_epochs} --gpus_per_trial 1 --evaluate_best 2>&1 | tee ${log_dir}/task${task_id}/${hp_space}.txt
