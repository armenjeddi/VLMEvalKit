## Set up

Run `bash setup.sh`

## Running experiments
To run experiments, run `bash test.sh`

In this file, there are parameters to change:

#### Model to evaluate
`model_path="Qwen/Qwen2.5-VL-7B-Instruct"`


#### VisionZip parameters
`enable_visionzip=False`\
`visionzip_ratio=0.55`

#### Pre-LLM KeyDiff Pruning (KeyDiff visionzip-style)
`enable_kdvz=False`\
`kdvz_ratio=0.5`

#### Prefill KeyDiff Pruning
`enable_kd_kvcache=False`\
`kvcache_anchor="all"`\
`kvcache_ratio=0.5`\
`kvcache_prune_after_layer=8`

`enable_kd_tokens=False`\
`tokens_anchor="all"\
`tokens_ratio=0.5`\
`tokens_prune_after_layer=8`\

#### Decode KeyDiff Pruning
`enable_kd_decode=False`\
`decode_anchor="all"`\
`decode_ratio=0.0`\
`decode_prune_window=50`\
`decode_prune_after_layer=8`


## Our contributions

Relevant changes can be found in these files:

Generic Qwen model for VLMEvalKit
- `VLMEvalKit/vlmeval/config.py`, Line 1480-1503

Transformers modifications:
- `VLMEvalKit/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py`
- `VLMEvalKit/transformers/src/transformers/integrations/flash_attention.py`
