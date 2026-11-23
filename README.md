# VLM-SP-Rew

Official project repository for **VLM-SP-Rew: Scene-Preprocessed VLM-Guided Reward Shaping for Model-Based Reinforcement Learning in Robotic Indoor Navigation**.

The implementation performs VLM annotation during scene preprocessing, constructs a graph and semantic value field, and supplies field-based rewards during model-based RL training. The deployed actor uses RGB observations and recurrent history.

## Repository layout

- collection.py, vlm_annotation.py: image capture and annotation requests.
- Graph, room and shortest-path utilities: graph construction and state-to-value lookup.
- paper_contract_r5.py, config_paper_r5.yaml, export_annotation_field.py: paper field equations and export interface.
- dreamerv3/, embodied/, revised_env.py: learner and environment interfaces.
- supplement/experiment_config_r6.json: shared seeds, rewards, field settings and checkpoints.
- supplement/scene_goal_manifest.csv: scene, target and start metadata.
- supplement/reconstructed_scene_cases.json: source scene configurations and geometry identifiers.
- supplement/R2R_scene_candidates.json: R2R source scenes, routes and viewpoints.
- supplement/costs.json: query workload and end-to-end cost-accounting definitions.
- release_audit.py: request date/model, image hashes, response and provider token-use recording.
- report_seed_pair_results.py: per-seed and per-pair SR/SPL aggregation from episode records.

## Installation and checks

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements_cpu_validation_RESOLVED.txt
TMPDIR=/path/on/large-disk/tmp python -m unittest discover -s tests_revision -v
```

The checks use generated test inputs to validate reward/field calculations, annotation caching, exports, environment-interface contracts and statistical aggregation.

## Configuration and execution

The paper configuration uses 300-pixel RGB observations. The legacy demonstration frontend has a separate observation profile; select the matching environment configuration. The field exporter accepts --paper-contract --config config_paper_r5.yaml.

```sh
python vlm_annotation.py --help
python export_annotation_field.py --help
python report_seed_pair_results.py --input artifacts/runs/episodes.csv --out artifacts/results --checkpoint-steps 400000
```

Provide annotation credentials through OPENAI_API_KEY. Annotation execution requires explicit enablement, a dated model and a positive call limit. Each request stores its actual input/output model identifiers, UTC, image hashes, response and provider usage. The reward loop does not require online VLM calls.

Run records contain record_kind, episode_id, benchmark, group, seed, pair_id, checkpoint_steps, success and spl. The reporter checks binary success, valid SPL, duplicate episodes, matched seeds and pair counts; it computes sample SD across fitted seeds. Cost accounting includes capture, annotation, graph/field construction, learning and evaluation.

## License

Project code is MIT-licensed. Bundled Dreamer code retains its notice; external scene assets and model weights retain their respective licenses.
