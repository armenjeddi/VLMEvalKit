
# 
python run.py --data MME --model Qwen2.5-VL-3B-Instruct-KD-66 --verbose --mode infer
python run.py --data MME --model Qwen2.5-VL-3B-Instruct-VZ-71 --verbose --mode all  --reuse


# 
# python run.py --data MME --model Qwen2.5-VL-3B-Instruct-Thinking-71 --verbose --mode infer
# python run.py --data MME --model Qwen2.5-VL-3B-Instruct-Thinking --verbose --mode all  --reuse




# MathVista
python run.py --data MathVista --model Qwen2.5-VL-3B-Instruct --verbose --mode infer




# NO_PROXY=0.0.0.0,localhost,127.0.0.1 no_proxy=0.0.0.0,localhost,127.0.0.1 HIP_VISIBLE_DEVICES=0 CK_MOE=0 python3 -m sglang.launch_server --model-path Qwen/Qwen2.5-VL-3B-Instruct --chat-template=qwen2-vl --disable-cuda-graph --tp-size 1
