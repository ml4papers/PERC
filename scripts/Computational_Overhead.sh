
python run_read.py --dataset 'ks50' --json-root './json_csv_files/ks50' --label-csv './json_csv_files/class_labels_indices_ks50.csv' --pretrain_path './data_path/cav_mae_ks50.pth' --tta-method 'ABPEM' --severity-start 5 --severity-end 5 --corruption-modality 'video' --gpu 0 --output-csv './outputs/ours_ks50_video_5.csv' --lr 1e-4 --batch-size 64 --use-cache --cache-size 64 --ent-weight 1. --bal-weight 0.8 --ce-weight 0.2 --cons-weight 0.2 --repeat 1 #1.65s/it


#======================================================================================================================================================================================================================================================================================================================================================================================================================================================================================
#python run_read_0.py --dataset 'ks50' --json-root './json_csv_files/ks50' --label-csv './json_csv_files/class_labels_indices_ks50.csv' --pretrain_path './data_path/cav_mae_ks50.pth' --tta-method 'READ' --severity-start 5 --severity-end 5 --corruption-modality 'video' --gpu 0 --output-csv './outputs/ours_ks50_video_5.csv' --lr 1e-4 --batch-size 1 --use-cache --cache-size 64 --ent-weight 1.0 --attn-weight 1.0 --prime-ent-k 8 --fix-loss-bal #0.58s/it


#python run_read_0.py --dataset 'ks50' --json-root './json_csv_files/ks50' --label-csv './json_csv_files/class_labels_indices_ks50.csv' --pretrain_path './data_path/cav_mae_ks50.pth' --tta-method 'ABPEM' --severity-start 5 --severity-end 5 --corruption-modality 'video' --gpu 0 --output-csv './outputs/ours_ks50_video_5.csv' --lr 1e-4 --batch-size 1 --use-cache --cache-size 64 --ent-weight 1.0 --attn-weight 1.0 --prime-ent-k 8 --fix-loss-bal #0.68s/it











