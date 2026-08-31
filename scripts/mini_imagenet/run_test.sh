gpuid=0
cd ../../
DATA_ROOT=/home/mxt/hjw/datasets/mini-imagenet-bdc

MODEL_1SHOT_PATH=./checkpoints/ResNet12_meta_deepbdc_5way_1shot_metatrain/best_model.tar
MODEL_5SHOT_PATH=./checkpoints/ResNet12_meta_deepbdc_5way_5shot_metatrain/best_model.tar


N_SHOT=1
python test.py --dataset mini_imagenet --data_path $DATA_ROOT --model ResNet12 --method meta_deepbdc --image_size 84 --gpu ${gpuid} --n_shot $N_SHOT --reduce_dim 640 --model_path $MODEL_1SHOT_PATH --test_task_nums 1 --test_n_episode 2000 --distance inner_product


N_SHOT=5
python test.py --dataset mini_imagenet --data_path $DATA_ROOT --model ResNet12 --method meta_deepbdc --image_size 84 --gpu ${gpuid} --n_shot $N_SHOT --reduce_dim 640 --model_path $MODEL_5SHOT_PATH --test_task_nums 1 --test_n_episode 2000 --distance euclidean


echo "_____________________addddwm_____________________"
N_SHOT=1
python test_addddwm.py --dataset mini_imagenet --data_path $DATA_ROOT --model ResNet12 --method meta_deepbdc --image_size 84 --gpu ${gpuid} --n_shot $N_SHOT --reduce_dim 640 --dc_k 4 --dc_alpha 0.01 --n_aug 1000 --model_path $MODEL_1SHOT_PATH --test_task_nums 1 --test_n_episode 2000 --distance inner_product

N_SHOT=5
python test_addddwm.py --dataset mini_imagenet --data_path $DATA_ROOT --model ResNet12 --method meta_deepbdc --image_size 84 --gpu ${gpuid} --n_shot $N_SHOT --reduce_dim 640 --dc_k 2 --dc_alpha 0.01 --n_aug 500 --model_path $MODEL_5SHOT_PATH --test_task_nums 1 --test_n_episode 2000 --distance euclidean

