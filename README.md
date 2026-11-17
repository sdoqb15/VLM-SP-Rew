# VLM-SP-Rew

Official project repository for **VLM-SP-Rew: Scene-Preprocessed VLM-Guided Reward Shaping for Model-Based Reinforcement Learning in Robotic Indoor Navigation**.

The implementation moves VLM inference into scene preprocessing. An annotation/graph/value-field pipeline supplies the training reward; the deployed navigation policy uses its visual observations and recurrent history.

## Repository structure

- `collection.py`, `vlm_annotation.py`: view collection and annotation requests.
- Graph, room and path utilities: scene graph construction and state-to-value lookup.
- `paper_contract_r5.py`, `config_paper_r5.yaml`, `export_annotation_field.py`: original-paper field equations and export profile.
- `dreamerv3/`, `embodied/`: navigation learning implementation and environment interfaces.
- `supplement/experiment_config_r6.json`: shared comparison protocol, five seeds,400k environment-step ceiling,25k evaluation intervals,8/2/8 scene quotas.
- `supplement/costs.json`: query and compute budget, token rates, hardware reference and accounting formulas.
- `supplement/scene_goal_manifest.csv`: scene/target metadata; `supplement/metadata_proxy_examples/`: metadata-derived format fixtures.
- `release_audit.py`: per-request model/date/image-hash/response/token-use ledger.
- `report_seed_pair_results.py`: matched per-seed and per-pair SR/SPL exports.

## Installation and checks

Reference Python3.10.12 environment and pinned dependencies:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements_cpu_validation_RESOLVED.txt
TMPDIR=/path/on/large-disk/tmp python -m unittest discover -s tests_revision -v
```

The checks cover annotation caching, reward and field calculations, export schemas, environment-interface contracts and seed/pair aggregation. The harness reserves110GiB for operational storage.

## Configuration profiles

The legacy demonstration frontend uses its existing96-pixel/goal-channel observation profile. `--paper-contract --config config_paper_r5.yaml` selects the paper's graph/value-field exporter. The R6 JSON specifies the shared300-pixel RGB-only experimental protocol; select the corresponding environment profile rather than mixing observation interfaces.

## Annotation and field export

```sh
python vlm_annotation.py --help
python export_annotation_field.py --help
```

The annotation profile pins `gpt-4o-2024-11-20`. Provide credentials through `OPENAI_API_KEY`; execution requires explicit enablement and a positive call budget. Calls retain the requested/returned model, UTC timestamp, ordered image hashes, response and usage. The online reward loop requires no VLM calls.

## Results and accounting

```sh
python report_seed_pair_results.py --input artifacts/runs/episodes.csv --out artifacts/results --checkpoint-steps 400000
```

Input columns are `record_kind,episode_id,benchmark,group,seed,pair_id,checkpoint_steps,success,spl`. The exporter checks binary success, valid SPL, duplicate IDs, matched seeds and balanced pair counts. It writes per-seed, per-pair-seed, per-pair and aggregate CSVs using sample SD across fitted seeds, retaining the input record kind.

Store run-derived artifacts under `artifacts/`: scene manifests, annotation request/response ledgers, graphs, value fields, per-episode navigation records and measured cost logs. The supplied cost ledger is a budget profile; per-run accounting uses observed usage and wall time.

## License

Project code is MIT-licensed. Bundled Dreamer code retains its third-party notice; scene assets and external model weights retain their original licenses.
