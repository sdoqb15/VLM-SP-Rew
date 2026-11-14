# VLM-SP-Rew — original-paper revision source

This repository contains the recovered author implementation, CPU-checked reconstruction helpers, and a concrete configuration for the IEEE Access revision. It does not contain completed supplementary navigation measurements. The original manuscript equations and pipeline are retained.

## What is available

- `collection.py`, `vlm_annotation.py`, graph/room/path/value scripts: recovered preprocessing pipeline.
- `paper_contract_r5.py`, `config_paper_r5.yaml`, `export_annotation_field.py --paper-contract`: opt-in original-paper graph/value profile.
- `tests_revision/`: 72 passing synthetic CPU engineering checks at the pinned source snapshot. These are not navigation efficacy measurements.
- `supplement/experiment_config_r6.json`: prepared shared field/reward/training/split specification; five seeds,400k maximum environment steps,25k evaluation grid,16/4/16 scene quotas.
- `supplement/scene_goal_manifest.csv`: schema only, currently zero scene rows. No invented scene identities or original four-pair records are supplied.
- `release_audit.py`: ordered image hashes, pinned model, raw-response and usage recording for explicitly enabled future calls. API execution is disabled by default.

## Installation and CPU checks

Reference CPU environment: Python3.10.12, AI2-THOR5.0.0, JAX/JAXLIB0.4.33, NumPy1.26.4, Pillow11.3.0, scikit-learn1.2.2. Exact reference dependencies are in `requirements_cpu_validation_RESOLVED.txt`.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements_cpu_validation_RESOLVED.txt
mkdir -p /path/on/large-disk/tmp
TMPDIR=/path/on/large-disk/tmp python -m unittest discover -s tests_revision -v
```

The engineering harness retains a110GiB operational free-space guard; use a disk with sufficient free space. This is a resource guard, not a navigation-method requirement.

## Annotation and field export

The original point-view collection and prompt scripts are retained. Set `OPENAI_API_KEY` only in the process environment; no credential is supplied. Newly executed calls require an explicitly dated model and a positive per-command call budget. The prepared profile requests `gpt-4o-2024-08-06`; provider documentation lists it as deprecated as checked on2026-11-14. Actual availability must be checked before execution; never silently substitute another model or claim it was the historical model. No paid calls were made for this release.

```sh
python vlm_annotation.py --help
python export_annotation_field.py --help
```

Use `--paper-contract --config config_paper_r5.yaml` for the original-paper field exporter, with actual scene/target/cache inputs required by its CLI. The legacy defaults and the opt-in paper profile are distinct. No synthetic cache may be passed off as real VLM annotations.

## Validation boundary

The full300-pixel RGB-only environment/reward profile, geometric-control exporter, native end-to-end training/evaluation, and RoboTHOR/R2R adapters remain unqualified. Do not combine paper-contract fields with the legacy96-pixel/goal-channel wrapper and claim an exact original-paper reproduction. `shared_reward_revision_config.json` is a preserved earlier prototype profile; the current prospective specification is `supplement/experiment_config_r6.json`. A specification is not proof of runtime equivalence.

Original four scene-goal identities, historical exact GPT snapshot/query dates, and historical raw predictions are not recovered. The release includes no fabricated raw annotation data, graphs, value fields, scene allocations, measured costs or navigation results. The CSVs are empty schemas; configured quotas are not actual population counts. Run-derived records belong under `artifacts/`, with the layout described below.

## Required empirical artifact layout

`artifacts/scene_manifest.csv`: actual scene/goal/start/split and asset identifiers.
`artifacts/annotations/`: per-point ordered input hashes, exact request/response/model/date, parsed object/room/visibility/navigability labels and usage/cost records.
`artifacts/graphs/`: actual edges, unreachable masks, graph distances and source hashes.
`artifacts/fields/`: frozen per-point values and field/config provenance.
`artifacts/runs/`: commands, code/environment/seed manifests, checkpoints, per-episode SR/SPL and all checkpoint curves.
`artifacts/costs.json`: actual call/token/billing and preprocessing-plus-training wall-time/hardware records. No estimate is labelled an actual bill.

These empirical directories are not present in this initial source release. Code is MIT-licensed with the bundled Dreamer notice retained; third-party scene/model licenses still apply.
