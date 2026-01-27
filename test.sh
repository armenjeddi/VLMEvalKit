#!/bin/bash

export HF_HOME="/home/minhle/projects/aip-btaati/shared/minhle/.cache/huggingface"
date_tag=$(date +"%b%d" | tr '[:upper:]' '[:lower:]')

QWEN25_7B="Qwen/Qwen2.5-VL-7B-Instruct"
INTERNVL="OpenGVLab/InternVL3_5-4B"

VISIONR1="Osilly/Vision-R1-7B"
OPENVL="ydeng9/OpenVLThinker-7B"
VLRETHINKER="TIGER-Lab/VL-Rethinker-7B"



mode="${1:-seq}"
if [[ "$mode" != "seq" && "$mode" != "con" ]]; then
    echo "Usage: $0 [con]" >&2
    echo "  (no arg)  run experiments sequentially (default)" >&2
    echo "  con       run experiments concurrently" >&2
    exit 1
fi

unset_all() {
    for v in model_path num_return_sequences temperature \
            enable_visionzip visionzip_ratio enable_kdvz kdvz_ratio \
            enable_kd_kvcache kvcache_anchor kvcache_ratio kvcache_prune_after_layer \
            enable_kd_tokens tokens_anchor tokens_ratio tokens_prune_layers \
            enable_kd_decode decode_anchor decode_ratio decode_prune_window decode_prune_after_layer \
            data model_name device run_id SPLIT_THINK enable_policy; do
        unset "$v" || true
    done
}

set_defaults() {
    model_name="${model_name:-Qwen}"
    data="${data:-MME}"
    device="${device:-0}"
    num_return_sequences="${num_return_sequences:-1}"
    temperature="${temperature:-0.000001}"
}

run_one() {
    set_defaults
    
    config_tags=()

    if [[ -n "${run_id:-}" ]]; then
        config_tags+=("run${run_id}")
    fi

    if [[ "${num_return_sequences}" -gt 1 ]]; then
        config_tags+=("major${num_return_sequences}")
    fi

    if awk "BEGIN{exit !(${temperature} > 0.000001)}"; then
        config_tags+=("temp${temperature//./}")
    fi

    [[ "$enable_visionzip" == "True" ]] && config_tags+=("vz${visionzip_ratio//./}")
    [[ "$enable_kdvz" == "True" ]]      && config_tags+=("kdvz${kdvz_ratio//./}")
    
    if [[ "$enable_kd_kvcache" == "True" ]]; then
        config_tags+=("kdkv_${kvcache_anchor}${kvcache_ratio//./}_i${kvcache_prune_after_layer}")
    fi

    if [[ "$enable_kd_tokens" == "True" ]]; then
        config_tags+=("kdt_${tokens_anchor}${tokens_ratio//./}_l${tokens_prune_layers}")
    fi
    
    if [[ "$enable_kd_decode" == "True" ]]; then
        config_tags+=("kdd_${decode_anchor}${decode_ratio//./}_w${decode_prune_window}")
    fi
    if [[ "$enable_policy" == "True" ]]; then
        config_tags+=("policy")
    fi
    # Fallback for base run
    if [ ${#config_tags[@]} -eq 0 ]; then config_tags=("base"); fi
    
    config_suffix=$(IFS=_; echo "${config_tags[*]}")
    work_dir="./results/${date_tag}/${data}_${config_suffix}"

    echo "------------------------------------------------"
    echo "Running Data: $data | Model: $model_name"
    echo "Output Dir: $work_dir"
    echo "------------------------------------------------"

    if [[ -n "${run_id:-}" ]]; then
        export run_id
    else
        unset run_id || true
    fi

    export model_path \
        num_return_sequences temperature \
        enable_visionzip visionzip_ratio enable_kdvz kdvz_ratio \
        enable_kd_kvcache kvcache_anchor kvcache_ratio kvcache_prune_after_layer \
        enable_kd_tokens tokens_anchor tokens_ratio tokens_prune_layers \
        enable_kd_decode decode_anchor decode_ratio decode_prune_window decode_prune_after_layer enable_policy

    CUDA_VISIBLE_DEVICES="$device" python run.py \
        --data "$data" \
        --model "$model_name" \
        --mode all \
        --work-dir "$work_dir"
}


export enable_thinking="True"
export policy_path="policy/best_checkpoint.pt"
experiments=(

    "device=0 data=MathVista_MINI model_name=Qwen" \

    "device=0 data=MathVista_MINI model_name=Qwen \
    enable_kdvz=True kdvz_ratio=0.5" \

    "device=0 data=MathVista_MINI model_name=Qwen \
    enable_kd_tokens=True tokens_anchor='all' tokens_ratio=0.5 tokens_prune_layers='4'" \

    "device=0 data=MathVista_MINI model_name=Qwen \
    num_return_sequences=5 temperature=0.7" \

)



bg_pids=()
for exp in "${experiments[@]}"; do
    if [[ "$mode" == "con" ]]; then
        (
            unset_all
            eval "$exp"
            echo "=== Experiment: ${exp} ==="
            run_one
        ) & bg_pids+=("$!")
    else
        (
            unset_all
            eval "$exp"
            echo "=== Experiment: ${exp} ==="
            run_one
        )
    fi
done

if [[ "$mode" == "con" ]]; then
    echo "Waiting for ${#bg_pids[@]} concurrent runs to finish..."
    wait "${bg_pids[@]}"
fi
