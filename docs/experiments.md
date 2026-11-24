# Running the manuscript experiments

`python -m vlm_sp_rew.experiments list` lists the Dreamer experiments. Each invocation prepares an immutable run directory and launches the same RGB-only learner with the selected reward/field. Use each of the five seeds for fitted-seed comparisons.

| Experiment entry | Purpose | Field |
|---|---|---|
| `benchmark-dreamer` | Benchmark-specific DreamerV3 control | None |
| `benchmark-vlm` | Benchmark-specific VLM-SP-Rew | Full semantic field |
| `shared-env` | Shared-configuration environment reward | None |
| `shared-geo` | Matched geometric control | Native reachable-edge/visibility field |
| `shared-vlm` | Shared-configuration VLM-SP-Rew | Full semantic field |
| `weight-0.5`, `weight-2` | ProcTHOR reward-weight sensitivity | Full field, selected outer coefficient |
| `visibility-only` | ProcTHOR semantic-signal ablation | High constant for visible points, lower constant elsewhere |
| `distance-only` | ProcTHOR semantic-signal ablation | Non-visible graph-distance bands at every point |
| `llava-13b` | ProcTHOR annotation-model comparison | Independently annotated LLaVA-13B field |
| `cross-scene-dreamer`, `cross-scene-vlm` | Single-scene fit, same/cross-scene evaluation | None / full field |

The zero/one coefficient experiments reuse `shared-env` and `shared-vlm`. They are not additional fitted models.

## Field configurations

Generate the configuration before capture/export. `shared=True` selects the common multi-scene settings; `shared=False` selects the benchmark-specific manuscript column:

```python
from vlm_sp_rew.protocol import write_configs
write_configs('/data/task', 'RoboTHOR', shared=False)
```

Use the resulting `field.yaml` for `export_field`/`geometry`, and pass its `reward.json` to direct training. The experiment launcher checks the field parameter profile and signal variant before starting the learner. The shared launcher fixes400k actions. Other experiment entries require an explicit `--steps` budget;400k is not silently imposed on benchmark-specific learning curves.

Generate the visibility/distance variants without changing scene allocation or graph nodes:

```sh
python -m vlm_sp_rew.ablate_field --field /data/fields/scene01/FIELD.json \
  --source-cache /data/cache/scene01 --goal-reference /data/task/goal.jpg \
  --config /data/task/field.yaml --variant visibility --out /data/fields/scene01-visibility.json
```

Use `--variant distance` for the distance-only field. For LLaVA-13B, run the same ordered prompts against a revision-pinned local model server, export its cache separately, and bind that exported field to the corresponding episode bank.

## Cross-scene and checkpoints

A cross-scene fit has exactly one training scene. Evaluate the same frozen checkpoint on separately declared same-scene starts and disjoint cross-scene scenes, with no further optimization. Reuse the fixed goal category and deployed RGB interface. A separate evaluation bank may contain held-out starts in the training scene; do not place one scene in two splits of the same bank.

`learner/checkpoints.json` indexes retained checkpoint directories. Use `vlm_sp_rew.evaluate` for every declared checkpoint on development episodes to construct learning curves. Apply the final common checkpoint to the test episodes. The source benchmark-specific learning curves and shared400k study have separate budget definitions.

## External comparison implementations

The paper's broader architecture comparisons use these author implementations:

- [EmbCLIP](https://github.com/allenai/embodied-clip), including its AllenAct policy and CLIP visual encoder.
- [EmbCLIP-Codebook](https://github.com/allenai/procthor-rl/tree/ed4feaf4944a01ad1d8b12a3dfc46f2c9cb4c992), native entries `rgb_clipresnet50gru_ddppo` and `rgb_clipresnet50gru_codebook_ddppo`.
- [R2D2 / Acme](https://github.com/google-deepmind/acme/tree/1bc05bea97b94a2da7e5315b315118bdaa98f1f3/acme/agents/jax/r2d2).
- [Dynalang](https://github.com/jlin816/dynalang/tree/5e3b93d4cdf3318537a97faa64a01219b82912a8).

Keep their dependency environments separate from the bundled Dreamer environment. Preserve their model and optimization definitions when adapting the task interface. Scene allocation,300×300 RGB, action space, stop predicate and environment reward are supplied by this project's frozen task contract. Their original multi-goal task samplers, resized image defaults and reward configurations are not interchangeable with the fixed-goal protocol here.
