export model_path="Qwen/Qwen2.5-VL-7B-Instruct"

export enable_visionzip=False
export visionzip_ratio=0.55

export enable_kdvz=False
export kdvz_ratio=0.5

export enable_kd_prefill=False
export prefill_anchor="all"
export prefill_ratio=0.5
export prefill_prune_after_layer=8

export enable_kd_decode=False
export decode_anchor="all"
export decode_ratio=0.0
export decode_prune_window=50
export decode_prune_after_layer=8
# 
python run.py --data MME --model Qwen --mode all


# MathVista
# python run.py --data MathVista --model Qwen2.5-VL-3B-Instruct --verbose --mode infer



# NO_PROXY=0.0.0.0,localhost,127.0.0.1 no_proxy=0.0.0.0,localhost,127.0.0.1 HIP_VISIBLE_DEVICES=0 CK_MOE=0 python3 -m sglang.launch_server --model-path Qwen/Qwen2.5-VL-3B-Instruct --chat-template=qwen2-vl --disable-cuda-graph --tp-size 1
