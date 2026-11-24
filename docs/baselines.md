# Author-source comparison runners

The comparison runners use the author implementations with this project's fixed-goal ObjectNav task. They share300×300 RGB, the benchmark's6/6/4actions,200action horizon, fixed-instance stop predicate and environment reward. Use separate environments for AllenAct, Acme and Dynalang; preserve the pinned source checkouts.

| Entry | Author implementation |
|---|---|
| `vlm_sp_rew.baselines.embclip` | ProcTHOR-RL `ResnetTensorNavActorCritic`, AllenAct PPO |
| Same entry with `--codebook` | ProcTHOR-RL `ResnetTensorNavActorCriticCodebook`, AllenAct PPO |
| `vlm_sp_rew.baselines.r2d2` | Acme recurrent Q network, prioritized recurrent replay and learner |
| `vlm_sp_rew.baselines.dynalang` | Dynalang world-model learner in the RGB-only, no-route-instruction task |

## Source versions and environments

```sh
git clone https://github.com/allenai/procthor-rl.git /data/vendor/procthor-rl
git -C /data/vendor/procthor-rl checkout ed4feaf4944a01ad1d8b12a3dfc46f2c9cb4c992

git clone https://github.com/google-deepmind/acme.git /data/vendor/acme
git -C /data/vendor/acme checkout 1bc05bea97b94a2da7e5315b315118bdaa98f1f3

git clone https://github.com/jlin816/dynalang.git /data/vendor/dynalang
git -C /data/vendor/dynalang checkout 5e3b93d4cdf3318537a97faa64a01219b82912a8
```

Install each author project's dependencies in its own environment. The AllenAct adapter uses AllenAct0.5.4 and the OpenAI CLIP implementation at `a1d071733d7111c9c014f024669f959182114e33`. Add this repository's `src/` to that environment's `PYTHONPATH`; its common native task adapter does not initialize the bundled JAX learner. Add the pinned author checkout with `--source-root`.

For the single-process Acme and Dynalang entries, create a Python3.10 environment and install `requirements/jax-baselines-py310.txt`. Start each entry in its own fresh process with this repository's `src/` on `PYTHONPATH`. The Acme local entry does not require Launchpad. Keep JAX, NumPy, TensorFlow, Flax and Orbax pins together when installing additional packages.

Use local CLIP RN50 weights and pass their actual SHA256. Both TorchScript and ordinary state dictionaries are supported. CLIP's spatial convolutional features are extracted directly from300×300 RGB, producing2048×9×9features. No224pixel input resize is applied. The native one-entry goal embedding is internal to each fixed-goal policy. The environment supplies RGB without a goal sensor or goal channel.

R2D2 converts300×300uint8 RGB to float32 in[0,1] for the native convolutional network, without resizing. It keeps the native observation/action/reward container, with the reward component of the **policy observation fixed to zero**. The learner receives the true task reward for Q targets. This avoids exposing privileged distance information to the deployed recurrent policy.

The single-process Acme entry creates a separate author-source copy under the run-record directory. It replaces distributed Launchpad termination hooks with standard SIGINT/SIGTERM handlers, imports local experiment runners, and uses the fully qualified native JAX sharding class names. The original checkout, R2D2 network, prioritized replay, losses, learner and optimizer are retained. The copy includes the original license and a file-hash receipt of the three runtime compatibility edits.

Dynalang uses its bundled `embodied` implementation in a fresh process. The task has no route instructions, depth input or language-reading actions. RGB is the encoder input. For300pixel reconstruction, its decoder retains six stride stages, starts from a5pixel grid, reconstructs320pixels and crops the output to300; the actor input is not resized.

## Train

The benchmark-specific comparison budget is supplied explicitly. These examples use400kactions; the shared Env–Geo–VLM study is run through `vlm_sp_rew.experiments` with its own fixed400kprotocol.

```sh
CUDA_VISIBLE_DEVICES=0 python -m vlm_sp_rew.baselines.embclip train \
  --source-root /data/vendor/procthor-rl --seed 288681 --steps 400000 \
  --episode-manifest /data/task/bank.json --data-root /data/task \
  --controller-config /data/task/controller.json --run-record-root /data/runs/embclip/records \
  --clip-weights /data/models/CLIP-RN50.pt --clip-sha256 "$CLIP_RN50_SHA256" \
  --logdir /data/runs/embclip
```

Add `--codebook` and use a separate run directory for EmbCLIP-Codebook. `--actors` sets parallel training samplers per fit; independent fits can run concurrently on separately allocated GPUs. Evaluation uses one pass over the frozen starts. Multiprocessing sockets use a short private local directory; `--ipc-root` can select its parent independently of dataset and run paths. The native PPO learning rate, clipping, entropy/value coefficients and32/64/128rollout progression are retained. The task sampler uses the frozen bank instead of the author's original multi-goal/randomized scene sampler. Checkpoints retain the actual action counts reached by native rollout batches.

```sh
CUDA_VISIBLE_DEVICES=1 python -m vlm_sp_rew.baselines.r2d2 train \
  --source-root /data/vendor/acme --seed 288681 --steps 400000 \
  --episode-manifest /data/task/bank.json --data-root /data/task \
  --controller-config /data/task/controller.json --run-record-root /data/runs/r2d2/records \
  --logdir /data/runs/r2d2

CUDA_VISIBLE_DEVICES=2 python -m vlm_sp_rew.baselines.dynalang train \
  --source-root /data/vendor/dynalang --seed 288681 --steps 400000 \
  --episode-manifest /data/task/bank.json --data-root /data/task \
  --controller-config /data/task/controller.json --run-record-root /data/runs/dynalang/records \
  --logdir /data/runs/dynalang
```

R2D2 writes action-indexed `step-*.pkl` checkpoints. Dynalang writes action-indexed checkpoints and JSON receipts under `dynalang_288681/action_checkpoints/`, alongside the author's native counter/logging files. The actual environment-action ceiling excludes resets, initialization observations and language-reading steps. Both retain the original learning algorithms.

## Evaluate

Evaluate one frozen checkpoint at a time on the bank's dev/test starts. Use fresh output files. Every runner exports the same per-episode CSV columns consumed by `vlm_sp_rew.export_metrics`.

For EmbCLIP/Codebook, pass `evaluate`, the original `--source-root`, task/controller/CLIP arguments, `--checkpoint /data/runs/embclip/checkpoints/...pt`, `--task-contract /data/runs/embclip/TASK_CONTRACT.json`, `--split test`, `--output /data/evaluation/embclip-288681.csv`, and a separate `--logdir`. The action count is read from the checkpoint. Keep `--codebook` for the Codebook model.

For R2D2, pass `evaluate`, the task/controller/source arguments, `--checkpoint /data/runs/r2d2/step-000400000.pkl`, `--split test` and `--output /data/evaluation/r2d2-288681.csv`.

For Dynalang, pass `evaluate`, the task/controller/source arguments, `--config /data/runs/dynalang_288681/config.yaml`, `--checkpoint /data/runs/dynalang_288681/action_checkpoints/step-000400000.ckpt`, `--split test` and `--output /data/evaluation/dynalang-288681.csv`.

The evaluation bank must have the same fixed category and observation/action contract as the fitted model, qualified native SPL references, and the declared held-out scene allocation. Checkpoint identity checks prevent accidental use of the author's different action ordering or a model trained for another target category.
