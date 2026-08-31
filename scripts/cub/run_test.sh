gpuid=0
N_SHOT=1

DATA_ROOT=/media/hk/22A08EEFA08EC92D/fewshot/data/CUB_200_2011/CUB # path to the json file of CUB
MODEL_1SHOT_PATH=/media/hk/22A08EEFA08EC92D/fewshot/DeepBDC-main/checkpoints/cub/ResNet18_meta_deepbdc_5way_1shot_metatrain/best_model.tar
MODEL_5SHOT_PATH=/media/hk/22A08EEFA08EC92D/fewshot/DeepBDC-main/checkpoints/cub/ResNet18_meta_deepbdc_5way_5shot_metatrain/best_model.tar

cd ../../
echo "============= meta-test 1-shot ============="
N_SHOT=1
#python test_addddwm.py --dataset cub --data_path $DATA_ROOT --model ResNet18 --method meta_deepbdc --image_size 224 --gpu ${gpuid} --n_shot $N_SHOT --reduce_dim 48 --dc_k 2 --dc_alpha 0.01 --n_aug 500 --model_path $MODEL_1SHOT_PATH --test_task_nums 5

python test.py --dataset cub --data_path $DATA_ROOT --model ResNet18 --method meta_deepbdc --image_size 224 --gpu ${gpuid} --n_shot $N_SHOT --reduce_dim 256 --model_path $MODEL_1SHOT_PATH --test_task_nums 5

echo "============= meta-test 5-shot ============="
N_SHOT=5
python test.py --dataset cub --data_path $DATA_ROOT --model ResNet18 --method meta_deepbdc --image_size 224 --gpu ${gpuid} --n_shot $N_SHOT --reduce_dim 256 --model_path $MODEL_5SHOT_PATH --test_task_nums 5

