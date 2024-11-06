CUDA_VISIBLE_DEVICES=0 python eval.py experiments/rdb/mpsc/res50.yaml --box_thresh 0.5 --resume workspace/mpsc/mpsc_res50
# CUDA_VISIBLE_DEVICES=1 python eval.py experiments/rdb/total/res50.yaml --box_thresh 0.65 --polygon --resume workspace/total/total_res50
# CUDA_VISIBLE_DEVICES=0 python eval.py experiments/rdb/td500/res50.yaml --box_thresh 0.45 --resume workspace/td500/td500_res50
# CUDA_VISIBLE_DEVICES=2 python eval.py experiments/rdb/ic15/res50.yaml --box_thresh 0.55 --resume workspace/ic15/ic15_res50
# CUDA_VISIBLE_DEVICES=0 python eval.py experiments/rdb/ctw/res50.yaml --box_thresh 0.5 --thresh 0.15 --polygon --resume workspace/ctw/ctw_res50