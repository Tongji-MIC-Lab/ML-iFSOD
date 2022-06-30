cd ../src
# base training
python main.py --task ctdet --exp_id coco_resdcn101_base --arch resdcn_101 --batch_size 32 --master_batch 1  \
--lr 1.25e-4 --gpus 4,5,6,7 --num_workers 16
# meta-learning
python ../src/main_meta.py --task ctdet_meta --dataset coco_meta --num_workers 16 --gpus 4,5,6,7 \
--arch resdcn_101 --exp_id coco_resdcn101 \
--batch_size 4 --master_batch 1 --num_epochs 3 --lr 1e-3 \
--update_lr 1e-3 --update_step 2 \
--fte_path ../exp/ctdet/coco_resdcn101_base/model_last.pth \
# few-shot finetuning
python ../src/main_meta.py --task ctdet_meta --dataset coco_meta --num_workers 16 --gpus 4,5,6,7 \
--arch resdcn_101 --exp_id coco_resdcn101 \
--fs_train_type novel --fs_lr 8e-3 --fs_epoch 1 --Kshot 10 --test_type x --fs_batch_size 40 \
--fte_path ../exp/ctdet/coco_resdcn101_base_256/model_last.pth \
--fs_train --resume
