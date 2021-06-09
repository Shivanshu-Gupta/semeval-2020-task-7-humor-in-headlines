python driver.py -o --write_preds --num_epochs 7 --gpu_idx 7 --learning_rate 3e-5 --weight_decay 0.2 task-1 --transformer roberta-base
# eval: 0.5207139253616333
# test: 0.5253996849060059

python driver.py -o --write_preds --num_epochs 4 --gpu_idx 6 --learning_rate 3e-4 --weight_decay 0.1 task-1 --transformer roberta-base --add_word_embs
# eval: 0.5228080153465271
# test: 0.5229454040527344