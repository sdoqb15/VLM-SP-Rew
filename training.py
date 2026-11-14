#!/usr/bin/env python3
"""
Training Stage:
- Step-Wise Value Assignment 
- VLM-Guided Reward Function 
- Integration with DreamerV3 
"""

import os
import sys
import time
import numpy as np
from pathlib import Path
import elements
import yaml
from train_utils import (
    setup_gpu_env,
    build_arg_parser,
    setup_random_seed,
    print_training_config,
    per_episode_callback,
    run_training,
)

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Device setup occurs only in the explicit training entry point, never at import.

import prior
from ai2thor.controller import Controller

import embodied

# Load VLM-SP-Rew parameters from config.yaml.
_CONFIG_PATH = Path(__file__).parent / 'config.yaml'
_CONFIG = {}
if _CONFIG_PATH.exists():
    with open(_CONFIG_PATH, 'r', encoding='utf-8') as f:
        _CONFIG = yaml.safe_load(f) or {}
_VLM_PARAMS = _CONFIG.get('vlm_sp_rew_params', {})

# ProcTHOR action set
PROCTHOR_ACTIONS = [
    "MoveAhead",
    "RotateLeft",
    "RotateRight",
    "LookUp",
    "LookDown",
    "Stop"
]

class ProcThorEnv(embodied.Env):
    """ProcTHOR environment wrapper for embodied.Env."""
    
    def __init__(self, split='train', max_houses=100, target_objects=None, max_steps=200, 
                 single_house=False, house_index=0, house_type=None, min_rooms=None, max_rooms=None,
                 episode_order='random', sampling_positions_dir=None, enable_scene_reward=False, 
                 scene_values_dir=None):
        """Initialize ProcTHOR env, dataset filter, and reward/scene state."""
        print("Initializing ProcThorEnv...")
        print(f"   Split: {split}")
        
        self.split = split
        self.max_houses = max_houses
        self.max_steps = max_steps
        self.single_house = single_house
        self.house_index = house_index
        self.house_type = house_type.lower() if house_type else None
        self.min_rooms = min_rooms
        self.max_rooms = max_rooms
        self.episode_order = episode_order
        self.sampling_positions_dir = sampling_positions_dir
        self.enable_scene_reward = enable_scene_reward
        self.scene_values_dir = scene_values_dir
        self.target_objects = target_objects if target_objects else ['HousePlant', 'Television', 'Laptop']
        
        self.predefined_positions = []
        self.current_position_idx = 0
        if self.sampling_positions_dir:
            self._load_sampling_positions()
            if self.predefined_positions:
                print(f"   Loaded {len(self.predefined_positions)} predefined start positions")
        
        self.scene_values = {}  # {(x, z): value}
        self.scene_positions = []  # [(x, z), ...]
        self.scene_room_types = {}
        self.scene_pos_to_dir = {}
        self.last_scene_value = None
        self.current_room_type = None
        self.last_closest_pos = None
        
        if self.enable_scene_reward:
            print("Scene-value reward enabled")
            if self.scene_values_dir:
                self._load_scene_values()
            else:
                print("   Warning: scene values dir not set, disabling scene-value reward")
                self.enable_scene_reward = False
        else:
            print("Scene-value reward disabled")
        
        if self.single_house:
            print(f"   Single-house mode: house index {house_index}")
        else:
            print(f"   Multi-house mode: up to {max_houses} houses")
        
        print(f"   Target objects: {', '.join(self.target_objects)}")
        print(f"   Episode order: {self.episode_order}")
        
        if self.house_type:
            print(f"   House type filter: {self.house_type}")
        if self.min_rooms:
            print(f"   Min rooms: {self.min_rooms}")
        if self.max_rooms:
            print(f"   Max rooms: {self.max_rooms}")
        
        print("Loading ProcTHOR-10k dataset...")
        self.dataset = prior.load_dataset("procthor-10k")
        
        print("Filtering houses...")
        self.houses = self._filter_houses(split, max_houses)
        print(f"   Loaded {len(self.houses)} houses after filtering")
        
        print("Initializing AI2-THOR controller...")
        from ai2thor.platform import CloudRendering
        self.controller = Controller(
            width=300,
            height=300,
            platform=CloudRendering,
            agentMode="bot",
            gridSize=0.25,
            rotateStepDegrees=90,
            visibilityDistance=1.5,
            fieldOfView=90
        )
        print("   Controller ready")
        
        self.current_house_idx = 0
        self.current_house = None
        self.current_scene = None
        self.step_count = 0
        self.done = False
        self.last_event = None
        self.last_action_name = None
        
        self.episode_reward = 0.0
        self.last_target_distance = None
        self.last_target_visible = None
        
        if not self.single_house and self.episode_order == 'random':
            import random
            random.shuffle(self.houses)
            print("   Shuffled house order")
        
        self._obs_space = {
            'image': elements.Space(np.uint8, (96, 96, 3)),
            'reward': elements.Space(np.float32, ()),
            'is_first': elements.Space(bool, ()),
            'is_last': elements.Space(bool, ()),
            'is_terminal': elements.Space(bool, ()),
        }
        
        self._act_space = {
            'action': elements.Space(np.int32, (), 0, len(PROCTHOR_ACTIONS)),
        }
        
        print("ProcThorEnv initialized")
    
    @property
    def obs_space(self):
        return self._obs_space
    
    @property
    def act_space(self):
        return self._act_space
    
    def _load_sampling_positions(self):
        """Load predefined start positions from a sampling directory."""
        from pathlib import Path
        
        sampling_dir = Path(self.sampling_positions_dir)
        if not sampling_dir.exists():
            print(f"   Warning: sampling positions dir not found: {sampling_dir}")
            return
        
        position_dirs = [d for d in sampling_dir.iterdir() if d.is_dir() and d.name.startswith('pos_')]
        
        if not position_dirs:
            print("   Warning: no sampling position dirs found")
            return
        
        for pos_dir in position_dirs:
            try:
                parts = pos_dir.name.split('_')
                if len(parts) >= 3:
                    x = float(parts[1])
                    z = float(parts[2])
                    self.predefined_positions.append({
                        'x': x,
                        'y': 0.9009995460510254,
                        'z': z
                    })
            except (ValueError, IndexError) as e:
                print(f"   Warning: cannot parse position dir: {pos_dir.name}")
        
        self.predefined_positions.sort(key=lambda p: (p['x'], p['z']))
        
        print(f"   Loaded {len(self.predefined_positions)} positions from {sampling_dir}")
    
    def _load_scene_values(self):
        """Load scene values and room types from value.txt and analysis files."""
        from pathlib import Path
        
        scene_values_path = Path(self.scene_values_dir)
        
        if not scene_values_path.exists():
            print(f"   Warning: scene values dir not found: {scene_values_path}")
            print("   Scene-value reward disabled")
            self.enable_scene_reward = False
            return
        
        print("   Loading scene values and room types...")
        print(f"      Dir: {scene_values_path}")
        
        loaded_count = 0
        loaded_room_count = 0
        for pos_dir in scene_values_path.iterdir():
            if not pos_dir.is_dir():
                continue
            
            value_file = pos_dir / 'value.txt'
            if not value_file.exists():
                continue
            
            parts = pos_dir.name.split('_')
            if len(parts) < 3:
                continue
            
            try:
                x = float(parts[1])
                z = float(parts[2])
                coord = (x, z)
                
                with open(value_file, 'r') as f:
                    first_line = f.readline().strip()
                    if first_line.startswith('Value:'):
                        value = int(first_line.split(':')[1].strip())
                        self.scene_values[coord] = value
                        self.scene_positions.append(coord)
                        self.scene_pos_to_dir[coord] = pos_dir.name
                        loaded_count += 1
                
                analysis_file = pos_dir / 'gpt4o_analysis.txt'
                if analysis_file.exists():
                    with open(analysis_file, 'r') as f:
                        for line in f:
                            if line.startswith('Room:'):
                                room_type = line.replace('Room:', '').strip()
                                self.scene_room_types[coord] = room_type
                                loaded_room_count += 1
                                break
            except (ValueError, IndexError) as e:
                continue
        
        if loaded_count > 0:
            self.scene_positions = np.array(self.scene_positions)
            print(f"      Loaded values for {loaded_count} positions")
            print(f"      Loaded room types for {loaded_room_count} positions")
            
            value_distribution = {}
            for value in self.scene_values.values():
                value_distribution[value] = value_distribution.get(value, 0) + 1
            
            print("      Value distribution:")
            for value in sorted(value_distribution.keys(), reverse=True):
                count = value_distribution[value]
                print(f"         V{value}: {count} positions")
            
            room_distribution = {}
            for room_type in self.scene_room_types.values():
                room_distribution[room_type] = room_distribution.get(room_type, 0) + 1
            
            print("      Room type distribution:")
            for room_type in sorted(room_distribution.keys()):
                count = room_distribution[room_type]
                print(f"         {room_type}: {count} positions")
        else:
            print("      Warning: no values loaded, disabling scene-value reward")
            self.enable_scene_reward = False
    
    def _get_current_scene_value(self, agent_pos_2d, is_first_step=False):
        """Map agent position to a scene value using room-consistent nearest sampling."""
        if len(self.scene_positions) == 0:
            return None
        
        agent_pos = np.array([agent_pos_2d[0], agent_pos_2d[1]])
        distances = np.linalg.norm(self.scene_positions - agent_pos, axis=1)
        
        sorted_indices = np.argsort(distances)
        
        if is_first_step:
            closest_idx = sorted_indices[0]
            closest_pos = tuple(self.scene_positions[closest_idx])
            closest_distance = distances[closest_idx]
            
            if closest_distance < 0.1 and closest_pos in self.scene_room_types:
                self.current_room_type = self.scene_room_types[closest_pos]
                print(f"      Step1 room type: {self.current_room_type} (sample: {self.scene_pos_to_dir.get(closest_pos, 'unknown')})")
            else:
                if closest_pos in self.scene_room_types:
                    self.current_room_type = self.scene_room_types[closest_pos]
                    print(f"      Step1 room type inferred: {self.current_room_type} (distance {closest_distance:.2f}m)")
        else:
            if self.last_closest_pos and self.last_closest_pos in self.scene_room_types:
                self.current_room_type = self.scene_room_types[self.last_closest_pos]
            else:
                closest_pos = tuple(self.scene_positions[sorted_indices[0]])
                if closest_pos in self.scene_room_types:
                    self.current_room_type = self.scene_room_types[closest_pos]
        
        print(f"      Current room: {self.current_room_type}")
        
        for idx in sorted_indices:
            candidate_pos = tuple(self.scene_positions[idx])
            candidate_distance = distances[idx]
            candidate_value = self.scene_values[candidate_pos]
            candidate_room = self.scene_room_types.get(candidate_pos, None)
            candidate_dir = self.scene_pos_to_dir.get(candidate_pos, 'unknown')
            
            if candidate_distance <= 0.2:
                self.last_closest_pos = candidate_pos
                print(f"      Use nearest sample {candidate_dir} (distance {candidate_distance:.2f}m), value {candidate_value}")
                return candidate_value
            
            if candidate_room == self.current_room_type:
                if candidate_distance >= 1.5:
                    fallback_pos = tuple(self.scene_positions[sorted_indices[0]])
                    fallback_value = self.scene_values[fallback_pos]
                    fallback_dir = self.scene_pos_to_dir.get(fallback_pos, 'unknown')
                    self.last_closest_pos = fallback_pos
                    print(f"      Room match but far ({candidate_distance:.2f}m), use nearest {fallback_dir}, value {fallback_value}")
                    return fallback_value
                else:
                    self.last_closest_pos = candidate_pos
                    print(f"      Room match {candidate_room}, use {candidate_dir}, value {candidate_value}")
                    return candidate_value
            else:
                print(f"      Skip {candidate_dir}: room mismatch ({candidate_room} vs {self.current_room_type}), distance {candidate_distance:.2f}m")
                continue
        
        fallback_pos = tuple(self.scene_positions[sorted_indices[0]])
        fallback_value = self.scene_values[fallback_pos]
        fallback_distance = distances[sorted_indices[0]]
        fallback_dir = self.scene_pos_to_dir.get(fallback_pos, 'unknown')
        self.last_closest_pos = fallback_pos
        print(f"      No same-room sample, use nearest {fallback_dir}, distance {fallback_distance:.2f}m, value {fallback_value}")
        return fallback_value
    
    def _filter_houses(self, split, max_houses):
        """Filter houses by split and room constraints."""
        filtered_houses = []
        dataset_split = self.dataset[split]
        
        for i in range(len(dataset_split)):
            if len(filtered_houses) >= max_houses:
                break
            
            house = dataset_split[i]
            
            num_rooms = len(house['rooms']) if 'rooms' in house else 0
            
            inferred_type = None
            if num_rooms <= 2:
                inferred_type = 'studio'
            elif num_rooms <= 5:
                inferred_type = 'apartment'
            else:
                inferred_type = 'house'
            
            if self.house_type and inferred_type != self.house_type:
                continue
            
            if self.min_rooms and num_rooms < self.min_rooms:
                continue
            if self.max_rooms and num_rooms > self.max_rooms:
                continue
            
            filtered_houses.append(house)
        
        return filtered_houses
    
    def _find_target_object(self, event):
        """Return the first visible target object from event metadata."""
        all_objects = event.metadata["objects"]
        
        for obj in all_objects:
            if obj['objectType'] in self.target_objects:
                return obj
        
        return None
    
    def _reset(self):
        """Reset episode, choose start pose, and compute initial target stats."""
        print("\nResetting environment...")
        
        if self.single_house:
            self.current_house = self.houses[self.house_index]
            print(f"   Single-house mode: house #{self.house_index}")
        else:
            if self.current_house_idx >= len(self.houses):
                self.current_house_idx = 0
                if self.episode_order == 'random':
                    import random
                    random.shuffle(self.houses)
                    print("   Completed a round, reshuffled houses")
                else:
                    print("   Completed a round, keep fixed order")
            
            self.current_house = self.houses[self.current_house_idx]
            print(f"   Multi-house mode: house {self.current_house_idx + 1}/{len(self.houses)}")
            self.current_house_idx += 1
        
        self.controller.reset(scene=self.current_house)
        
        if self.predefined_positions and self.episode_order == 'fixed':
            position = self.predefined_positions[self.current_position_idx]
            print(f"   Use predefined position {self.current_position_idx + 1}/{len(self.predefined_positions)}: ({position['x']:.2f}, {position['z']:.2f})")
            
            self.current_position_idx = (self.current_position_idx + 1) % len(self.predefined_positions)
            
            import random
            rotation = random.choice([0, 90, 180, 270])
        else:
            event = self.controller.step(action="GetReachablePositions")
            reachable_positions = event.metadata["actionReturn"]
            
            if reachable_positions:
                import random
                position = random.choice(reachable_positions)
                rotation = random.choice([0, 90, 180, 270])
                print(f"   Random position: ({position['x']:.2f}, {position['z']:.2f})")
            else:
                print("   Warning: no reachable positions, use default")
                position = {'x': 0, 'y': 0.9, 'z': 0}
                rotation = 0
        
        self.controller.step(
            action="Teleport",
            position=position,
            rotation={"x": 0, "y": rotation, "z": 0},
            horizon=30
        )
        
        self.last_event = self.controller.last_event
        
        target_obj = self._find_target_object(self.last_event)
        
        self.shortest_path_length = None
        
        if target_obj:
            print(f"   Target: {target_obj['objectType']}")
            
            agent_pos = np.array([
                self.last_event.metadata["agent"]["position"]["x"],
                self.last_event.metadata["agent"]["position"]["z"]
            ])
            obj_pos = np.array([
                target_obj["position"]["x"],
                target_obj["position"]["z"]
            ])
            self.last_target_distance = np.linalg.norm(agent_pos - obj_pos)
            self.last_target_visible = target_obj["visible"]
            
            estimated_steps = int(self.last_target_distance / 0.25)
            estimated_rotations = 2
            self.shortest_path_length = estimated_steps + estimated_rotations
            print(f"   Estimated shortest path: {self.shortest_path_length} steps (heuristic)")
            
            print(f"   Initial distance: {self.last_target_distance:.2f}m")
            print(f"   Initially visible: {self.last_target_visible}")
        else:
            print("   Warning: no target object found in this house")
            self.last_target_distance = None
            self.last_target_visible = None
        
        self.step_count = 0
        self.done = False
        self.episode_reward = 0.0
        
        self.last_scene_value = None
        self.current_room_type = None
        self.last_closest_pos = None
        
        obs = self._get_obs(0.0, False, first=True)
        return obs
    
    def reset(self):
        """Public reset wrapper for embodied.Env."""
        return self._reset()
    
    def step(self, action):
        """Apply action, compute reward, and return next observation."""
        step_start_time = time.time()
        
        if isinstance(action, dict) and action.get('reset', False):
            return self._reset()
        
        action_name = PROCTHOR_ACTIONS[action['action']]
        self.last_action_name = action_name
        
        action_start_time = time.time()
        event = self.controller.step(action_name)
        action_time = time.time() - action_start_time
        
        self.last_event = event
        
        target_obj = self._find_target_object(event)
        
        target_visible = False
        target_distance = float('inf')
        target_pos = None
        success = False
        
        if target_obj:
            target_visible = target_obj["visible"]
            target_pos = (target_obj["position"]["x"], target_obj["position"]["z"])
            
            agent_pos = np.array([
                event.metadata["agent"]["position"]["x"],
                event.metadata["agent"]["position"]["z"]
            ])
            obj_pos = np.array([
                target_obj["position"]["x"],
                target_obj["position"]["z"]
            ])
            target_distance = np.linalg.norm(agent_pos - obj_pos)
            
            success = (action_name == "Stop" and target_visible and target_distance <= 1.0)
        
        self.step_count += 1
        
        reward = self._calculate_reward(action_name, success, target_visible, target_distance)
        
        self.episode_reward += reward
        
        done = (action_name == "Stop") or (self.step_count >= self.max_steps)
        self.done = done
        
        is_terminal = (action_name == "Stop")
        
        obs_start_time = time.time()
        obs = self._get_obs(reward, done, is_terminal=is_terminal)
        obs_time = time.time() - obs_start_time
        
        total_time = time.time() - step_start_time
        
        print(f"Step {self.step_count}: {action_name} | action {action_time:.3f}s | obs {obs_time:.3f}s | total {total_time:.3f}s")
        
        agent_position = event.metadata["agent"]["position"]
        agent_pos_2d = (agent_position["x"], agent_position["z"])
        
        visibility_change = ""
        if self.last_target_visible is not None:
            if not self.last_target_visible and target_visible:
                visibility_change = " (️️️ gained!)"
            elif self.last_target_visible and not target_visible:
                visibility_change = " (️️ lost!)"
        
        print(f"Agent pos: ({agent_pos_2d[0]:.2f}, {agent_pos_2d[1]:.2f}) | distance {target_distance:.2f}m | visible {target_visible}{visibility_change}")
        print(f"   Step reward: {reward:.3f}, Cumulative: {self.episode_reward:.3f}")
        
        visible_objects = self._get_visible_objects(event)
        if visible_objects:
            object_counts = {}
            for obj_type in visible_objects:
                object_counts[obj_type] = object_counts.get(obj_type, 0) + 1
            
            objects_str = ", ".join([f"{obj_type}×{count}" if count > 1 else obj_type 
                                   for obj_type, count in object_counts.items()])
            print(f"Visible objects: {objects_str} (total {len(visible_objects)})")
        else:
            print("Visible objects: None")
        
        if done:
            obs['log/episode_reward'] = self.episode_reward
            obs['log/episode_length'] = self.step_count
            obs['log/success'] = 1.0 if success else 0.0
            obs['log/shortest_path_length'] = self.shortest_path_length if hasattr(self, 'shortest_path_length') and self.shortest_path_length else max(1, int(self.last_target_distance / 0.25) + 2)
            
            print(f"\nEpisode ended:")
            print(f"   Steps: {self.step_count}/{self.max_steps}")
            print(f"   Reward: {self.episode_reward:.3f}")
            print(f"   Success: {success}")
            if hasattr(self, 'shortest_path_length') and self.shortest_path_length:
                print(f"   Shortest path: {self.shortest_path_length} steps")
                print(f"   Path efficiency: {self.shortest_path_length / max(self.step_count, 1):.2%}")
            if is_terminal:
                print("   TERMINAL: task completed with Stop action")
            else:
                print("   TRUNCATED: episode ended due to timeout")
        
        return obs
    
    def _calculate_reward(self, action_name, success, target_visible, target_distance):
        """Compute reward from distance shaping, success, and scene value change."""
        step_penalty = -0.01
        goal_success_reward = 100.0
        shaping_weight_positive = 1.0
        shaping_weight_negative = 1.0
        reached_horizon_reward = -10.0
        
        tau = _VLM_PARAMS.get('tau', 3)
        w_hi = _VLM_PARAMS.get('w_hi', 1.0)
        w_lo = _VLM_PARAMS.get('w_lo', 0.5)
        r_up = _VLM_PARAMS.get('r_up', 1.0)
        r_down = _VLM_PARAMS.get('r_down', -1.0)
        
        reward = step_penalty
        print(f"   Step penalty: {step_penalty:.3f}")
        
        if self.last_target_distance is not None and target_distance >= 0:
            distance_improvement = self.last_target_distance - target_distance
            max_reward_mag = 0.25
            shaped_reward = max(min(distance_improvement, max_reward_mag), -max_reward_mag)
            
            if target_distance <= 1.0 and target_visible:
                if shaped_reward != 0:
                    print("   Stop condition met, block distance reward")
                shaped_reward = 0.0
            
            if shaped_reward > 0:
                final_reward = shaped_reward * shaping_weight_positive
                print(f"   Distance decreased: {distance_improvement:.3f}m, reward +{final_reward:.3f}")
                reward += final_reward
            elif shaped_reward < 0:
                final_reward = shaped_reward * shaping_weight_negative
                print(f"   Distance increased: {-distance_improvement:.3f}m, penalty {final_reward:.3f}")
                reward += final_reward
            else:
                print(f"   Distance unchanged: {target_distance:.2f}m")
        
        if action_name == "Stop":
            if success:
                reward += goal_success_reward
                print(f"   Stop action success: +{goal_success_reward}")
        elif self.step_count >= self.max_steps:
            reward += reached_horizon_reward
            print(f"   Reached max steps ({self.step_count}/{self.max_steps}), penalty {reached_horizon_reward}")
        
        if self.enable_scene_reward and len(self.scene_positions) > 0:
            agent_pos = self.last_event.metadata["agent"]["position"]
            agent_pos_2d = (agent_pos["x"], agent_pos["z"])
            
            is_first_step = (self.step_count == 1)
            
            current_scene_value = self._get_current_scene_value(agent_pos_2d, is_first_step=is_first_step)
            
            if current_scene_value is not None:
                if self.last_scene_value is not None:
                    high_transition = (current_scene_value >= tau) or (self.last_scene_value >= tau)
                    scale = w_hi if high_transition else w_lo
                    if current_scene_value > self.last_scene_value:
                        scene_reward = r_up * scale
                        level = "high" if high_transition else "low"
                        print(f"   Scene value up ({level}): {self.last_scene_value} -> {current_scene_value}, +{scene_reward:.3f}")
                        reward += scene_reward
                    elif current_scene_value < self.last_scene_value:
                        scene_penalty = r_down * scale
                        level = "high" if high_transition else "low"
                        print(f"   Scene value down ({level}): {self.last_scene_value} -> {current_scene_value}, {scene_penalty:.3f}")
                        reward += scene_penalty
                    else:
                        value_type = "high" if current_scene_value >= tau else "low"
                        print(f"   Scene value unchanged: {current_scene_value} ({value_type})")
                else:
                    value_type = "high(visible)" if current_scene_value >= tau else "low(not visible)"
                    print(f"   Initial scene value: {current_scene_value} ({value_type})")
                
                self.last_scene_value = current_scene_value
        
        if target_distance >= 0:
            self.last_target_distance = target_distance
        self.last_target_visible = target_visible
        
        print(f"   Total reward: {reward:.3f}")
        
        return reward
    
    def _get_visible_objects(self, event):
        """List visible object types in the current frame."""
        visible_objects = []
        for obj in event.metadata["objects"]:
            if obj["visible"]:
                visible_objects.append(obj["objectType"])
        return visible_objects
    
    def _get_obs(self, reward=0.0, done=False, first=False, is_terminal=None):
        """Build observation dict for DreamerV3."""
        event = self.last_event
        rgb = np.array(event.frame)
        
        if is_terminal is None:
            is_terminal = done and hasattr(self, 'last_action_name') and self.last_action_name == "Stop"
        
        obs = {
            'image': self._preprocess_image(rgb),
            'reward': np.float32(reward),
            'is_first': first,
            'is_last': done,
            'is_terminal': is_terminal,
        }
        
        return obs
    
    def _preprocess_image(self, image):
        """Resize RGB image to 96x96 uint8."""
        from PIL import Image
        image = np.array(Image.fromarray(image).resize((96, 96)))
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        return image
    
    def close(self):
        """Close the AI2-THOR controller."""
        if self.controller:
            self.controller.stop()


def main():
    """Training entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()
    setup_gpu_env(platform=args.jax_platform)
    setup_random_seed(args.seed)
    print_training_config(args)
    env_cls = ProcThorEnv
    if args.episode_manifest:
        from revised_env import RevisedProcThorEnv
        env_cls = RevisedProcThorEnv
    run_training(args, env_cls, per_episode_callback)


if __name__ == "__main__":
    main()

