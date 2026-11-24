# VLM-SP-Rew

Official implementation of **VLM-SP-Rew: Scene-Preprocessed VLM-Guided Reward Shaping for Model-Based Reinforcement Learning in Robotic Indoor Navigation**.

VLM-SP-Rew samples navigable scene locations, annotates eight RGB views at each location, builds a semantic graph and value field, and adds a regional sign reward to the environment reward during DreamerV3 training. VLM annotation is performed during preprocessing. The deployed policy uses RGB observations and recurrent history.

## Installation

Use Python3.10 and a dedicated environment:

```sh
python3.10 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

For CUDA, install the JAX0.4.33 CUDA12 plugin appropriate to the host driver in this environment. Set `CUDA_VISIBLE_DEVICES` explicitly for each learner. Renderer device indices are configured separately. AI2-THOR needs a working Vulkan renderer. R2R-adapted uses Habitat-Sim with the licensed Matterport3D/VLN-CE reconstruction assets and native semantic annotations; install Habitat-Sim in the corresponding environment.

## Paper configuration

| Setting | Value |
|---|---|
| Actor observation | `uint8` RGB,300×300×3; no goal channel |
| RoboTHOR / ProcTHOR actions | MoveAhead, RotateLeft, RotateRight, LookUp, LookDown, Stop |
| R2R-adapted actions | MoveAhead, RotateLeft, RotateRight, Stop |
| Action increments | 0.25m forward;90° turn |
| Episode horizon | 200 actions |
| Environment reward | −1 per action + previous−current3D target distance; +100 on successful Stop |
| Successful Stop | The fixed target instance is visible and its distance is at most1m |
| Shared study | 8train /2development /8test scenes;10evaluation starts per scene–goal pair |
| Shared fitted seeds | 288681–288685 |
| Shared training budget | 400,000 environment actions; checkpoints0,25,000,…,400,000 |

Each policy run fixes its goal category and instance allocation. Target coordinates, visibility metadata, semantic graphs and field values are used on the training side. They are not policy observation channels. `protocol.py` defines the benchmark-specific field settings; the shared study uses the ProcTHOR column across all three benchmarks.

## Data preparation

`examples/scene_goal_manifest.csv` is the **scene–goal configuration manifest**. It records scene identifiers, goal categories, split allocations, initial positions and orientations, and pinned dataset sources. Use the manifest to select and track scene–goal configurations; the [dataset guide](docs/datasets.md) explains how to bind them to native assets and the JSON episode bank read by the runner. Store licensed scene assets, model weights, caches and run outputs outside this Git checkout. Provide one frozen episode bank with disjoint train/dev/test scene identities and native target instance IDs.

```sh
python -m vlm_sp_rew.preprocess \
  --episode-manifest /data/task/bank.json --episode-id train-scene01-start01 \
  --data-root /data/task --controller-config /data/task/controller.json \
  --goal-reference /data/task/goal.jpg --out /data/cache/scene01

python -m vlm_sp_rew.annotate \
  --output_dir /data/cache/scene01 --target_image /data/task/goal.jpg \
  --execute-api --model gpt-4o-2024-11-20 --max-api-calls 100

python -m vlm_sp_rew.export_field \
  --source-cache /data/cache/scene01 --scene /data/task/scene01.json \
  --goal-metadata /data/cache/scene01/GOAL_METADATA.json \
  --goal-reference /data/task/goal.jpg --config /data/task/field.yaml \
  --out /data/fields/scene01 --plots
```

Set `OPENAI_API_KEY` in the environment. Annotation requests retain ordered image hashes, requested/returned model identifiers, request time, raw responses and token usage. For LLaVA-13B use `--api-base-url`, `--model` and the full model repository `--model-revision` against an OpenAI-compatible local deployment accepting the ordered multi-image prompt.

The same capture produces `GEOMETRY.json` for the geometric control:

```sh
python -m vlm_sp_rew.geometry --geometry /data/cache/scene01/GEOMETRY.json \
  --config /data/task/field.yaml --out /data/fields/scene01-geo.json
```

Freeze the field files and place their paths/SHA256 values in the episode bank before training.

Run `python -m vlm_sp_rew.qualify_paths --help` to generate native SPL references for the frozen evaluation starts. The dataset guide specifies the stop-region conventions.

## Training and evaluation

See the [experiment guide](docs/experiments.md) for field configurations, ablations and cross-scene evaluation, and the [author-source comparison guide](docs/baselines.md) for EmbCLIP, EmbCLIP-Codebook, R2D2 and Dynalang.

```sh
python -m vlm_sp_rew.experiments list
CUDA_VISIBLE_DEVICES=0 python -m vlm_sp_rew.experiments train \
  --experiment shared-vlm --benchmark ProcTHOR --seed 288681 \
  --episode-manifest /data/task/bank.json --data-root /data/task \
  --controller-config /data/task/controller.json --out /data/runs/shared-vlm-288681

CUDA_VISIBLE_DEVICES=0 python -m vlm_sp_rew.evaluate \
  --config /data/runs/shared-vlm-288681/learner/config.yaml \
  --checkpoint /data/runs/shared-vlm-288681/learner/checkpoints/step-000400000 \
  --checkpoint-steps 400000 --group VLM-SP-Rew \
  --episode-manifest /data/task/bank.json --data-root /data/task --split test \
  --controller-config /data/task/controller.json --run-record-root /data/evaluation/records \
  --output /data/evaluation/vlm-288681.csv
```

Combine the per-fit CSV files with one header and export per-seed/per-pair statistics:

```sh
python -m vlm_sp_rew.export_metrics --input /data/evaluation/episodes.csv \
  --out /data/evaluation/summary --checkpoint-steps 400000
```

SPL uses a navigable shortest-path reference to the same fixed goal and success region. Reset observations are excluded from environment-action budgets. RGB300 is retained at the network interface; the CNN uses bottom/right stride padding and crops its reconstruction to300×300.

## Source layout and checks

- `src/vlm_sp_rew/`: preprocessing, annotation, graph/field construction, rewards, benchmark adapters, training and evaluation.
- `src/dreamerv3/`, `src/embodied/`: bundled learner core.
- `tests/`: reward, field, episode allocation, observation and metric checks.
- `tools/`: bounded CPU integration checks using generated test inputs.
- `examples/`: scene metadata and reference-image examples.

```sh
TMPDIR=/data/tmp python -m unittest discover -s tests -v
TMPDIR=/data/tmp JAX_PLATFORMS=cpu python tools/smoke_dreamer_replay_update_cpu.py
TMPDIR=/data/tmp JAX_PLATFORMS=cpu python tools/check_training_checkpoints_cpu.py
```

## License

Project code is MIT-licensed. Bundled learner code retains its upstream notices. Third-party scene assets, baseline implementations and model weights retain their original licenses.
