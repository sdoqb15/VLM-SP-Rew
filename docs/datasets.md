# Native datasets and frozen episode banks

Use native simulator coordinates and fixed object instance identities. Store licensed data and runtime outputs outside the source checkout. Metadata in `examples/scene_goal_manifest.csv` helps locate scenes and source episodes; the runtime consumes the JSON episode bank described below.

## Episode bank

```json
{
  "schema_version": 1,
  "benchmark": "ProcTHOR",
  "goal_vocabulary": ["HousePlant"],
  "episodes": [
    {
      "episode_id": "train-scene01-start01",
      "pair_id": "scene01-HousePlant-instance01",
      "split": "train",
      "seed": 288681,
      "scene": {"path": "scenes/scene01.json", "sha256": "SCENE_FILE_SHA256"},
      "scene_identity_sha256": "NATIVE_SCENE_IDENTITY_SHA256",
      "goal_category": "HousePlant",
      "goal_instance_ids": ["NATIVE_OBJECT_INSTANCE_ID"],
      "start_position": {"x": 1.0, "y": 0.9, "z": 2.0},
      "start_yaw_degrees": 0.0,
      "start_horizon_degrees": 0.0,
      "fields": {
        "vlm": {"path": "fields/scene01/FIELD.json", "sha256": "FIELD_FILE_SHA256"},
        "geo": {"path": "fields/scene01-geo.json", "sha256": "GEOMETRIC_FIELD_SHA256"}
      }
    }
  ]
}
```

Fill the IDs, poses and hashes from the actual native scene allocation. Relative paths resolve under `--data-root`. The benchmark must be `RoboTHOR`, `ProcTHOR`, or `R2R-adapted`. A bank contains one goal category and one fixed target instance per scene–goal allocation. Include all18disjoint scenes for the shared protocol, with10frozen starts per development/test scene. Scene identities are checked across splits independently of filenames.

Compute a scene's identity with `vlm_sp_rew.reward.floor_identity(json.loads(scene_bytes))` and file SHA256 with `hashlib.sha256(scene_bytes).hexdigest()`. ProcTHOR scene JSON contains the actual house and floor polygons. Start allocations are checked after native teleportation; target disappearance or target substitution fails allocation.

## RoboTHOR

Use official preset scenes and the corresponding pinned AI2-THOR engine. A scene descriptor contains:

```json
{
  "backend": "robothor",
  "scene_id": "FloorPlan_Train1_1",
  "engine_sha256": "PINNED_ENGINE_BINARY_SHA256",
  "assets": [{"path": "metadata/FloorPlan_Train1_1.json", "sha256": "NATIVE_SCENE_METADATA_SHA256"}]
}
```

The metadata asset records the actual scene/object allocation. The descriptor's engine hash must equal the controller's binary hash. Official category-goal episode lists provide start poses and a category; bind the fixed target to an actual native object instance before using the bank.

## AI2-THOR controller

A controller file supplies a local engine binary, its SHA256 and the allocated renderer device:

```json
{
  "simulator_version": "ai2thor-5.0.0",
  "executable_sha256": "PINNED_ENGINE_BINARY_SHA256",
  "physical_gpu_uuid": "ALLOCATED_RENDERER_GPU_UUID",
  "vulkan_device_index": 0,
  "controller_kwargs": {
    "local_executable_path": "/data/engines/AI2-THOR",
    "width": 300,
    "height": 300,
    "gridSize": 0.25,
    "rotateStepDegrees": 90,
    "visibilityDistance": 1.5
  }
}
```

Use the preset-compatible engine/agent configuration for RoboTHOR and the house-compatible engine for ProcTHOR. The renderer's Vulkan device index is distinct from the learner's CUDA-visible index. Record the actual physical GPU allocation. Do not trigger an automatic engine download in the experiment loop.

## R2R-adapted ObjectNav

The task is fixed-object navigation in a Matterport3D/VLN-CE reconstruction, not instruction-following evaluation at a route endpoint. Use Habitat-native Y-up start coordinates and a native semantic object instance. Record a supplied quaternion as `start_rotation_xyzw` when retaining the original Habitat orientation, and its matching `start_yaw_degrees`. The four-action camera has zero pitch.

A Habitat scene descriptor contains `backend: habitat`, `coordinate_frame: Habitat_Y_up`, native `scene_id`, `scene_dataset_config`, an `assets` list with paths/SHA256 for the reconstruction, semantic assets, navmesh and dataset configuration, and `semantic_category_mapping` such as `{"plant": "HousePlant"}`. Paths resolve under the Habitat controller's `asset_root`. The same relative source root is used when validating the episode-bank assets.

The Habitat controller file contains `backend: habitat`, `asset_root`, explicit `gpu_device_id`, actual `habitat_sim_version`, and the chosen camera/agent height, radius and horizontal FOV. Native semantic pixels define trainer-side visibility of the fixed instance; only RGB enters the policy.

## SPL reference

Each evaluation episode additionally carries `spl_reference` with:

- `reference_kind: qualified_navigable_shortest_path`, `units: metres`, `success_rule_matched: true`;
- `shortest_path_m` measured on the native navigable mesh to the same goal and valid stop region;
- matching `scene_sha256`, `goal_category`, `start_position`, `start_yaw_degrees`, `success_distance_m: 1.0`.

Retain the native shortest-path/visibility qualification record with the bank. A straight-line target distance, graph-hop count, route endpoint or action count is not a replacement for the ObjectNav SPL denominator.
