#!/bin/bash

export HF_HOME="~/.cache/huggingface"
date_tag=$(date +"%b%d" | tr '[:upper:]' '[:lower:]')

QWEN25_7B="Qwen/Qwen2.5-VL-7B-Instruct"


unset_all() {
    for v in model_path enable_visionzip visionzip_ratio enable_kdvz kdvz_ratio \
             enable_kd_kvcache kvcache_anchor kvcache_ratio kvcache_prune_after_layer \
             enable_kd_tokens tokens_anchor tokens_ratio tokens_prune_layers \
             enable_kd_decode decode_anchor decode_ratio decode_prune_window decode_prune_after_layer \
             data model_name device; do
        unset "$v" || true
    done
}

set_defaults() {
    model_path="${model_path:-$QWEN25_7B}"
    model_name="${model_name:-Qwen}"
    data="${data:-MME}"
    device="${device:-0}"
}

run_one() {
    set_defaults
    
    config_tags=()
    
    [[ "$enable_visionzip" == "True" ]] && config_tags+=("vz${visionzip_ratio//./}")
    [[ "$enable_kdvz" == "True" ]]      && config_tags+=("kdvz${kdvz_ratio//./}")
    
    if [[ "$enable_kd_tokens" == "True" ]]; then
        config_tags+=("tk_${tokens_anchor}_r${tokens_ratio//./}_l${tokens_prune_layers}")
    fi
    
    if [[ "$enable_kd_decode" == "True" ]]; then
        config_tags+=("dec_${decode_anchor}_r${decode_ratio//./}_w${decode_prune_window}")
    fi

    # Fallback for base run
    if [ ${#config_tags[@]} -eq 0 ]; then config_tags=("base"); fi
    
    config_suffix=$(IFS=_; echo "${config_tags[*]}")
    work_dir="./results/${model_name}/${data}-${config_suffix}-${date_tag}"

    echo "------------------------------------------------"
    echo "Running Data: $data | Model: $model_name"
    echo "Work Dir: $work_dir"
    echo "------------------------------------------------"

    export model_path enable_visionzip visionzip_ratio enable_kdvz kdvz_ratio \
           enable_kd_kvcache kvcache_anchor kvcache_ratio kvcache_prune_after_layer \
           enable_kd_tokens tokens_anchor tokens_ratio tokens_prune_layers \
           enable_kd_decode decode_anchor decode_ratio decode_prune_window decode_prune_after_layer

    CUDA_VISIBLE_DEVICES="$device" python run.py \
        --data "$data" \
        --model "$model_name" \
        --mode all \
        --work-dir "$work_dir"
}


experiments=(

    # "device=0 data=MME model_name=Qwen \
    # enable_kdvz=True kdvz_ratio=0.5"
  
    "device=0 data=MME model_name=Qwen \
    enable_kdvz=True visionzip_ratio=0.5 \
    enable_kd_tokens=True tokens_anchor=all tokens_ratio=0.5 tokens_prune_layers=4"

    
  
    # "device=0 data=MME model_name=Qwen

  
)


for exp in "${experiments[@]}"; do
    (
        unset_all
        eval "$exp"
        run_one
    )
done