## Set up

Run `bash setup.sh`

## Running experiments
To run experiments, run `bash test.sh`. To run experiments consecutively, run `bash test.sh con` (remember to change the device numbers)

An experiment example:
`"device=0 data=MathVista_MINI model_name=Qwen majority_vote=8 temperature=0.7 enable_kdvz=True kdvz_ratio=0.5"`

This means to run Qwen model on MathVista_MINI, prune 50% of visual tokens using KeyDiff, and generate 8 different responses per question with `temperature=0.7`.

In this file, there are parameters to change:

## Model Parameters

#### HF_HOME
Change this to your own HF_HOME path to access downloaded HF checkpoints.

#### Model to evaluate
`model_name = {Qwen, InternVL}`\
`model_path` is defaulted in `VLMEvalKit/vlmeval/config.py`, but can also be set manually in this file

#### Other parameters:
`run_id` - helpful for when running multiple experiments with identical configurations, as the output folder names will be different\
`enable_thinking` - force model to generate thinking\
`enable_cot` - force model to generate chain-of-thought thinking\

## Majority Voting Parameters
`majority_vote` - number of responses per question \
`temperature` - generation temperature, needs to be set to around 0.5-0.7 for majority vote experiments


## Pruning Parameters

#### VisionZip parameters
`enable_visionzip=False`\
`visionzip_ratio=0.55`

#### Pre-LLM KeyDiff Pruning parameters (KeyDiff visionzip-style)
`enable_kdvz=True/False`\
`kdvz_ratio=0.5` (float between 0.0 to 1.0)

#### Prefill KeyDiff Tokens Pruning parameters
`enable_kd_tokens=True/False`\
`tokens_anchor="all"`\
`tokens_ratio=0.5` (float between 0.0 to 1.0)\
`tokens_prune_layers=8` (can be multiple layers, for example `4,8,10`)

#### Prefill KeyDiff KV-Cache Pruning parameters (not used anymore)
`enable_kd_kvcache=True/False`\
`kvcache_anchor="all"/"text"/"vision"`\
`kvcache_ratio=0.5` (float between 0.0 to 1.0)\
`kvcache_prune_after_layer=8` (integer between 0 and num_layers)

#### Decode KeyDiff Pruning parameters (not used anymore)
`enable_kd_decode=True/False`\
`decode_anchor="all"`\
`decode_ratio=0.0` (float between 0.0 to 1.0)\
`decode_prune_window=50`\
`decode_prune_after_layer=8`



## Running Majority Vote Experiments
First run experiments as usual. This will produce a raw log file.
Find the script `majority.py`, change API_KEY, DATASET_TYPE, input and output files, and run it for evaluation.

Due to nature of VLMEvalKit, it's not straightforward to evaluate majority vote experiments so we resort to running generation and evaluation separately.

## Our contributions

Relevant changes can be found in these files:

Generic Qwen and InternVL model for VLMEvalKit
- `VLMEvalKit/vlmeval/config.py`

Transformers modifications:
- `VLMEvalKit/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py`
- `VLMEvalKit/transformers/src/transformers/integrations/flash_attention.py`
