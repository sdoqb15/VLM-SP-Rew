#!/usr/bin/env python3
"""
Scene Preprocessing Stage: Shortest-Path Distances Computation.
Compute BFS shortest paths to the goal and average path position statistics.
"""

import json
import os
import sys
from pathlib import Path
from collections import deque, defaultdict
import numpy as np
import prior
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
import yaml


def load_navigation_graph(json_file):
    """Load navigation graph JSON."""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def find_target_object(controller, target_object_type):
    """Find target object position in the scene."""
    event = controller.last_event
    
    for obj in event.metadata['objects']:
        obj_type = obj['objectId'].split('|')[0] if '|' in obj['objectId'] else obj['objectType']
        
        if obj_type == target_object_type:
            pos = obj['position']
            return (pos['x'], pos['z'])
    
    return None


def add_goal_node_and_edges(nodes, edges, target_pos, max_connection_distance=1):
    """Create goal node and connect to nearby samples."""
    updated_nodes = nodes.copy()
    
    goal_id = len(updated_nodes)
    goal_node = {
        'id': goal_id,
        'position': {
            'x': target_pos[0],
            'z': target_pos[1]
        },
        'dir_name': 'GOAL'
    }
    updated_nodes.append(goal_node)
    
    updated_edges = edges.copy()
    
    connected_count = 0
    for node in nodes:
        node_pos = (node['position']['x'], node['position']['z'])
        dist = np.sqrt((node_pos[0] - target_pos[0])**2 + (node_pos[1] - target_pos[1])**2)
        
        if dist <= max_connection_distance:
            updated_edges.append({
                'from': node['id'],
                'to': goal_id,
                'distance': dist,
                'direction': 'TO_GOAL'
            })
            updated_edges.append({
                'from': goal_id,
                'to': node['id'],
                'distance': dist,
                'direction': 'FROM_GOAL'
            })
            connected_count += 1
    
    print(f"   Goal node created (ID: {goal_id})")
    print(f"   Position: ({target_pos[0]:.3f}, {target_pos[1]:.3f})")
    print(f"   Connected to {connected_count} sampled points")
    
    return updated_nodes, updated_edges, goal_id


def build_adjacency_list(nodes, edges):
    """Build adjacency list (undirected)."""
    adj = defaultdict(list)
    
    for edge in edges:
        from_id = edge['from']
        to_id = edge['to']
        
        if to_id not in adj[from_id]:
            adj[from_id].append(to_id)
    
    return adj


def bfs_shortest_path(adj, start_id, goal_id):
    """BFS shortest path from start_id to goal_id."""
    if start_id == goal_id:
        return [start_id]
    
    queue = deque([(start_id, [start_id])])
    visited = {start_id}
    
    while queue:
        current, path = queue.popleft()
        
        for neighbor in adj[current]:
            if neighbor in visited:
                continue
            
            new_path = path + [neighbor]
            
            if neighbor == goal_id:
                return new_path
            
            visited.add(neighbor)
            queue.append((neighbor, new_path))
    
    return None


def calculate_average_positions(paths, goal_id):
    """Average path position from the goal."""
    node_positions = defaultdict(list)
    
    for start_id, path in paths.items():
        if path is None:
            continue
        
        path_length = len(path)
        for i, node_id in enumerate(path):
            position_from_end = path_length - i  # 1, 2, 3, ...
            node_positions[node_id].append(position_from_end)
    
    average_positions = {}
    for node_id, positions in node_positions.items():
        average_positions[node_id] = np.mean(positions)
    
    return average_positions


def get_node_room_type(analysis_dir, dir_name):
    """Read node room type from gpt4o_analysis.txt."""
    analysis_file = Path(analysis_dir) / dir_name / 'gpt4o_analysis.txt'
    
    if not analysis_file.exists():
        return None
    
    try:
        with open(analysis_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('Room:'):
                    return line.replace('Room:', '').strip()
    except Exception:
        return None
    
    return None


def assign_isolated_node_positions(nodes, average_positions, isolated_nodes, analysis_dir):
    """Assign values for isolated nodes using room-type nearest neighbor."""
    if not isolated_nodes or not average_positions:
        return average_positions
    
    updated_positions = average_positions.copy()
    
    print(f" Handling {len(isolated_nodes)} isolated nodes (room-type heuristic)...")
    
    for isolated_id in isolated_nodes:
        isolated_pos = (nodes[isolated_id]['position']['x'], nodes[isolated_id]['position']['z'])
        isolated_dir = nodes[isolated_id]['dir_name']
        
        isolated_room = get_node_room_type(analysis_dir, isolated_dir)
        
        if isolated_room:
            print(f"   Node {isolated_id} ({isolated_dir}): {isolated_room}")
        else:
            print(f"   Node {isolated_id} ({isolated_dir}): room type not found")
        
        global_min_dist = float('inf')
        global_nearest_id = None
        
        same_room_min_dist = float('inf')
        same_room_nearest_id = None
        
        for node_id, avg_pos in average_positions.items():
            node_pos = (nodes[node_id]['position']['x'], nodes[node_id]['position']['z'])
            dist = np.sqrt((isolated_pos[0] - node_pos[0])**2 + (isolated_pos[1] - node_pos[1])**2)
            
            if dist < global_min_dist:
                global_min_dist = dist
                global_nearest_id = node_id
            
            if isolated_room:
                node_dir = nodes[node_id]['dir_name']
                node_room = get_node_room_type(analysis_dir, node_dir)
                
                if node_room == isolated_room and dist < same_room_min_dist:
                    same_room_min_dist = dist
                    same_room_nearest_id = node_id
        
        if same_room_nearest_id is not None and same_room_min_dist <= 1.5:
            chosen_id = same_room_nearest_id
            chosen_dist = same_room_min_dist
            reason = f"same room ({isolated_room}), dist {chosen_dist:.2f}m <= 1.5m"
        else:
            chosen_id = global_nearest_id
            chosen_dist = global_min_dist
            if same_room_nearest_id is not None:
                reason = f"same room dist {same_room_min_dist:.2f}m > 1.5m, use global nearest"
            else:
                reason = "no same-room neighbor, use global nearest"
        
        if chosen_id is not None:
            updated_positions[isolated_id] = average_positions[chosen_id]
            print(f"      Inherit node {chosen_id} ({nodes[chosen_id]['dir_name']}), "
                  f"value {average_positions[chosen_id]:.2f} ({reason})")
    
    print("   Isolated nodes processed")
    print()
    
    return updated_positions


def get_top_down_frame(controller):
    """Get top-down frame."""
    from PIL import Image
    
    event = controller.step(action="GetMapViewCameraProperties", raise_for_failure=True)
    pose = event.metadata["actionReturn"]
    
    event = controller.step(
        action="AddThirdPartyCamera",
        **pose,
        skyboxColor="white",
        raise_for_failure=True,
    )
    
    topdown_image = event.third_party_camera_frames[-1]
    return Image.fromarray(topdown_image)


def visualize_average_positions(controller, house, nodes, edges, average_positions, goal_id, isolated_nodes, output_path):
    """Render average path positions on top-down view."""
    from PIL import Image, ImageDraw, ImageFont
    
    print(" Rendering average-position visualization...")
    
    controller.reset(scene=house)
    topdown_img = get_top_down_frame(controller)
    
    event = controller.step(action="GetMapViewCameraProperties")
    cam_position = event.metadata["actionReturn"]["position"]
    cam_orth_size = event.metadata["actionReturn"]["orthographicSize"]
    
    img_width, img_height = topdown_img.size
    
    def world_to_image(x, z):
        """Convert world coordinates to top-down image coordinates."""
        img_x = img_width / 2 + (x - cam_position['x']) / (cam_orth_size * 2) * img_width
        img_y = img_height / 2 - (z - cam_position['z']) / (cam_orth_size * 2) * img_height
        return (int(img_x), int(img_y))
    
    node_img_positions = []
    for node in nodes:
        img_pos = world_to_image(node['position']['x'], node['position']['z'])
        node_img_positions.append(img_pos)
    
    draw = ImageDraw.Draw(topdown_img)
    
    for edge in edges:
        from_pos = node_img_positions[edge['from']]
        to_pos = node_img_positions[edge['to']]
        
        if (0 <= from_pos[0] < img_width and 0 <= from_pos[1] < img_height and
            0 <= to_pos[0] < img_width and 0 <= to_pos[1] < img_height):
            
            draw.line([from_pos, to_pos], fill=(200, 200, 200), width=2)
    
    if average_positions:
        min_pos = min(average_positions.values())
        max_pos = max(average_positions.values())
    else:
        min_pos = max_pos = 1
    
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 10)
    except:
        font = ImageFont.load_default()
        font_small = font
    
    node_radius = 8
    
    for i, img_pos in enumerate(node_img_positions):
        if not (0 <= img_pos[0] < img_width and 0 <= img_pos[1] < img_height):
            continue
        
        is_isolated = i in isolated_nodes
        
        if i in average_positions:
            avg_pos = average_positions[i]
            
            if max_pos > min_pos:
                normalized = (max_pos - avg_pos) / (max_pos - min_pos)
            else:
                normalized = 0
            
            if normalized < 0.5:
                r = 255
                g = int(255 * normalized * 2)
                b = 0
            else:
                r = int(255 * (1 - (normalized - 0.5) * 2))
                g = 255
                b = 0
            
            fill_color = (r, g, b)
            
            if i == goal_id:
                fill_color = (0, 150, 255)
                node_radius_actual = 15
                outline_color = (255, 215, 0)
                outline_width = 3
            elif is_isolated:
                node_radius_actual = node_radius
                outline_color = (0, 0, 0)
                outline_width = 3
            else:
                node_radius_actual = node_radius
                outline_color = (0, 0, 0)
                outline_width = 2
        else:
            fill_color = (128, 128, 128)
            node_radius_actual = node_radius
            outline_color = (0, 0, 0)
            outline_width = 2
        
        if i == goal_id:
            star_points = []
            for j in range(10):
                angle = j * 36 - 90
                radius = node_radius_actual if j % 2 == 0 else node_radius_actual // 2
                x = img_pos[0] + radius * np.cos(np.radians(angle))
                y = img_pos[1] + radius * np.sin(np.radians(angle))
                star_points.append((x, y))
            
            draw.polygon(star_points, fill=fill_color, outline=outline_color, width=outline_width)
        else:
            draw.ellipse(
                [img_pos[0]-node_radius_actual, img_pos[1]-node_radius_actual,
                 img_pos[0]+node_radius_actual, img_pos[1]+node_radius_actual],
                fill=fill_color,
                outline=outline_color,
                width=outline_width
            )
        
        if is_isolated and i in average_positions:
            for angle in range(0, 360, 30):
                start_angle = angle
                end_angle = angle + 15
                draw.arc(
                    [img_pos[0]-node_radius_actual-3, img_pos[1]-node_radius_actual-3,
                     img_pos[0]+node_radius_actual+3, img_pos[1]+node_radius_actual+3],
                    start=start_angle,
                    end=end_angle,
                    fill=(255, 0, 255),
                    width=2
                )
        
        if i in average_positions:
            avg_pos = average_positions[i]
            text = f"{avg_pos:.1f}"
            
            bbox = draw.textbbox((0, 0), text, font=font_small)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            text_x = img_pos[0] - text_width // 2
            text_y = img_pos[1] + node_radius_actual + 4
            
            draw.rectangle(
                [text_x - 2, text_y - 2, text_x + text_width + 2, text_y + text_height + 2],
                fill=(255, 255, 255, 230),
                outline=(0, 0, 0),
                width=1
            )
            
            draw.text((text_x, text_y), text, fill=(0, 0, 0), font=font_small)
    
    legend_width = 240
    legend_height = 156
    legend_x = img_width - legend_width - 15
    legend_y = img_height - legend_height - 15
    
    draw.rectangle(
        [legend_x - 5, legend_y - 5, legend_x + legend_width, legend_y + legend_height],
        fill=(255, 255, 255, 240),
        outline=(100, 100, 100),
        width=1
    )
    
    title_y = legend_y + 2
    draw.text((legend_x, title_y), "Average Position from Goal", fill=(0, 0, 0), font=font)
    
    item_y = title_y + 20
    
    star_center_x = legend_x + 6
    star_center_y = item_y + 6
    star_radius = 6
    star_points = []
    for j in range(10):
        angle = j * 36 - 90
        radius = star_radius if j % 2 == 0 else star_radius // 2
        x = star_center_x + radius * np.cos(np.radians(angle))
        y = star_center_y + radius * np.sin(np.radians(angle))
        star_points.append((x, y))
    draw.polygon(star_points, fill=(0, 150, 255), outline=(255, 215, 0), width=2)
    draw.text((legend_x + 18, item_y - 2), 
              f"Goal (actual target): 1.0", 
              fill=(0, 100, 255), font=font_small)
    
    item_y += 20
    
    draw.text((legend_x, item_y), "Distance from Goal:", fill=(0, 0, 0), font=font_small)
    item_y += 16
    
    draw.ellipse([legend_x, item_y, legend_x+10, item_y+10], 
                 fill=(0, 255, 0), outline=(0, 0, 0), width=1)
    draw.text((legend_x + 16, item_y - 2), 
              "Near", 
              fill=(0, 180, 0), font=font_small)
    
    item_y += 16
    
    draw.ellipse([legend_x, item_y, legend_x+10, item_y+10], 
                 fill=(255, 255, 0), outline=(0, 0, 0), width=1)
    draw.text((legend_x + 16, item_y - 2), 
              "Medium", 
              fill=(180, 180, 0), font=font_small)
    
    item_y += 16
    
    draw.ellipse([legend_x, item_y, legend_x+10, item_y+10], 
                 fill=(255, 0, 0), outline=(0, 0, 0), width=1)
    draw.text((legend_x + 16, item_y - 2), 
              "Far", 
              fill=(180, 0, 0), font=font_small)
    
    item_y += 16
    
    draw.ellipse([legend_x, item_y, legend_x+10, item_y+10], 
                 fill=(255, 255, 0), outline=(0, 0, 0), width=2)
    for angle in range(0, 360, 30):
        draw.arc(
            [legend_x-2, item_y-2, legend_x+12, item_y+12],
            start=angle,
            end=angle+15,
            fill=(255, 0, 255),
            width=2
        )
    draw.text((legend_x + 16, item_y - 2), 
              "Isolated (inherited)", 
              fill=(150, 0, 150), font=font_small)
    
    topdown_img.save(output_path, "PNG")
    print(f"   Visualization saved: {output_path}")
    print()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze shortest paths')
    parser.add_argument('--config', type=str, default=str(Path(__file__).parent / 'config.yaml'),
                        help='Config file path')
    parser.add_argument('--analysis_dir', type=str, default='procthor_analysis_house1',
                        help='Analysis directory')
    parser.add_argument('--target_object', type=str, default='HousePlant',
                        help='Target object type')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'],
                        help='ProcTHOR dataset split')
    parser.add_argument('--house-index', type=int, default=1,
                        help='House index')
    parser.add_argument('--min-rooms', type=int, default=4,
                        help='Min rooms')
    parser.add_argument('--max-rooms', type=int, default=4,
                        help='Max rooms')
    
    config_args, _ = parser.parse_known_args()
    config_path = Path(config_args.config)
    config = {}
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
    stage_defaults = config.get('stage_defaults', {})
    
    if 'analysis_dir' in stage_defaults:
        parser.set_defaults(analysis_dir=stage_defaults['analysis_dir'])
    if 'target_object' in stage_defaults:
        parser.set_defaults(target_object=stage_defaults['target_object'])
    
    args = parser.parse_args()
    
    print("Shortest-path analysis")
    print("=" * 80)
    
    graph_file = Path(args.analysis_dir) / "navigation_graph.json"
    if not graph_file.exists():
        print(f"Graph file not found: {graph_file}")
        return
    
    print(f"Loading graph: {graph_file}")
    graph_data = load_navigation_graph(graph_file)
    nodes = graph_data['nodes']
    edges = graph_data['edges']
    
    print(f"   Nodes: {len(nodes)}")
    print(f"   Edges: {len(edges)}")
    print()
    
    print("Loading ProcTHOR scene...")
    dataset = prior.load_dataset("procthor-10k")
    
    houses = []
    for i in range(len(dataset[args.split])):
        house = dataset[args.split][i]
        num_rooms = len(house.get('rooms', []))
        
        if args.min_rooms and num_rooms < args.min_rooms:
            continue
        if args.max_rooms and num_rooms > args.max_rooms:
            continue
        
        houses.append(house)
        
        if len(houses) > args.house_index:
            break
    
    if args.house_index >= len(houses):
        print(f"Error: house_index={args.house_index} out of range")
        return
    
    target_house = houses[args.house_index]
    print(f"   Using house index: {args.house_index}")
    print()
    
    print("Finding target position...")
    controller = Controller(
        width=800,
        height=800,
        platform=CloudRendering,
        agentMode="bot",
    )
    
    controller.reset(scene=target_house)
    target_pos = find_target_object(controller, args.target_object)
    
    if target_pos is None:
        print(f"Target not found: {args.target_object}")
        controller.stop()
        return
    
    print(f"   Target position: ({target_pos[0]:.3f}, {target_pos[1]:.3f})")
    print()
    
    print("Creating goal node and edges...")
    nodes, edges, goal_id = add_goal_node_and_edges(nodes, edges, target_pos, max_connection_distance=1)
    goal_node = nodes[goal_id]
    print()
    
    print("Building adjacency list...")
    adj = build_adjacency_list(nodes, edges)
    
    num_sample_nodes = len(nodes) - 1
    isolated_nodes = [i for i in range(num_sample_nodes) if i not in adj or len(adj[i]) == 0]
    connected_nodes = [i for i in range(num_sample_nodes) if i in adj and len(adj[i]) > 0]
    
    print(f"   Connected nodes: {len(connected_nodes)}")
    print(f"   Isolated nodes: {len(isolated_nodes)}")
    if isolated_nodes:
        print(f"   Isolated node IDs: {isolated_nodes}")
    print()
    
    print("Computing shortest paths...")
    paths = {}
    unreachable_nodes = []
    
    for node_id in connected_nodes:
        if node_id == goal_id:
            paths[node_id] = [goal_id]
        else:
            path = bfs_shortest_path(adj, node_id, goal_id)
            if path is None:
                unreachable_nodes.append(node_id)
            paths[node_id] = path
    
    reachable_count = len([p for p in paths.values() if p is not None])
    print(f"   Reachable paths: {reachable_count}")
    print(f"   Unreachable nodes: {len(unreachable_nodes)}")
    if unreachable_nodes:
        print(f"   Unreachable node IDs: {unreachable_nodes}")
    print()
    
    print("Computing average positions from goal...")
    average_positions = calculate_average_positions(paths, goal_id)
    
    print(f"   Done ({len(average_positions)} connected nodes)")
    print()
    
    average_positions = assign_isolated_node_positions(nodes, average_positions, isolated_nodes, args.analysis_dir)
    
    sorted_positions = sorted(average_positions.items(), key=lambda x: x[1])
    print()
    
    print("=" * 80)
    print("  Detailed results")
    print("=" * 80)
    print()
    
    print(f"Goal node (ID {goal_id}):")
    print("   Type: actual target position")
    print(f"   Position: ({goal_node['position']['x']:.3f}, {goal_node['position']['z']:.3f})")
    print(f"   Average position: {average_positions[goal_id]:.2f} (goal=1)")
    print()
    
    print("Average positions for all samples (goal -> start):")
    print("-" * 80)
    print(f"{'Rank':<6} {'Node':<8} {'AvgPos':<12} {'Dir':<25} {'Type':<15}")
    print("-" * 80)
    
    node_appearances = defaultdict(int)
    for path in paths.values():
        if path:
            for node_id in path:
                node_appearances[node_id] += 1
    
    rank = 1
    for node_id, avg_pos in sorted_positions:
        if node_id == goal_id:
            continue
        
        node = nodes[node_id]
        appearances = node_appearances[node_id]
        
        if node_id in isolated_nodes:
            node_type = "isolated(inherited)"
        elif appearances == 0:
            node_type = "unreachable"
        else:
            node_type = f"path x{appearances}"
        
        print(f"{rank:<6} {node_id:<8} {avg_pos:<12.2f} {node['dir_name']:<25} {node_type:<15}")
        rank += 1
    
    print("-" * 80)
    print()
    
    output_file = Path(args.analysis_dir) / "shortest_paths_analysis.json"
    
    output_data = {
        'target_object': args.target_object,
        'target_position': {'x': target_pos[0], 'z': target_pos[1]},
        'goal_node_id': goal_id,
        'goal_node_type': 'actual_target_location',
        'goal_node_position': {'x': goal_node['position']['x'], 'z': goal_node['position']['z']},
        'statistics': {
            'total_nodes': len(nodes),
            'total_sample_nodes': len(nodes) - 1,
            'connected_nodes': len(connected_nodes),
            'isolated_nodes': len(isolated_nodes),
            'reachable_paths': reachable_count,
            'unreachable_nodes': len(unreachable_nodes)
        },
        'paths': {
            str(start_id): {
                'path': path,
                'path_dirs': [nodes[nid]['dir_name'] for nid in path] if path else None,
                'length': len(path) if path else None
            }
            for start_id, path in paths.items()
        },
        'average_positions': {
            str(node_id): {
                'position': float(avg_pos),
                'appearances': node_appearances[node_id],
                'dir_name': nodes[node_id]['dir_name'],
                'is_isolated': node_id in isolated_nodes,
                'value_inherited': node_id in isolated_nodes
            }
            for node_id, avg_pos in average_positions.items()
        },
        'isolated_nodes': {
            'count': len(isolated_nodes),
            'ids': isolated_nodes
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved: {output_file}")
    print()
    
    visualization_file = Path(args.analysis_dir) / "shortest_paths_visualization.png"
    visualize_average_positions(controller, target_house, nodes, edges, average_positions, goal_id, isolated_nodes, visualization_file)
    
    controller.stop()
    
    print("Example paths (first 5):")
    print("-" * 80)
    
    example_count = 0
    for start_id, path in paths.items():
        if path and len(path) > 1 and example_count < 5 and start_id != goal_id:
            path_str = ' → '.join([nodes[nid]['dir_name'] for nid in path])
            print(f"Node {start_id} to goal: {path_str}")
            print(f"   Path length: {len(path)} steps")
            example_count += 1
    
    print("-" * 80)
    print()
    
    print("=" * 80)
    print("Analysis complete")
    print("=" * 80)
    print("Outputs:")
    print(f"   - JSON: {output_file}")
    print(f"   - Visualization: {visualization_file}")
    print()


if __name__ == "__main__":
    main()

