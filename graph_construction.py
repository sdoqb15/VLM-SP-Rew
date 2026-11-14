#!/usr/bin/env python3
"""
Scene Preprocessing Stage: Graph Construction.
Build the navigation graph from VLM navigability and spatial proximity.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import prior
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
import yaml


def parse_position_from_dirname(dirname):
    """Parse (x, z) from directory name "pos_X.XXX_Z.ZZZ"."""
    parts = dirname.split('_')
    if len(parts) < 3 or parts[0] != 'pos':
        return None
    
    try:
        x = float(parts[1])
        z = float(parts[2])
        return (x, z)
    except (ValueError, IndexError):
        return None


def calculate_distance_2d(pos1, pos2):
    """2D Euclidean distance."""
    dx = pos1[0] - pos2[0]
    dz = pos1[1] - pos2[1]
    return np.sqrt(dx*dx + dz*dz)


def calculate_direction(from_pos, to_pos):
    """Direction from from_pos to to_pos (N, NE, E, SE, S, SW, W, NW)."""
    dx = to_pos[0] - from_pos[0]
    dz = to_pos[1] - from_pos[1]
    
    angle_rad = np.arctan2(dx, dz)
    angle_deg = np.degrees(angle_rad)
    
    if angle_deg < 0:
        angle_deg += 360
    
    # N: 337.5-22.5, NE: 22.5-67.5, E: 67.5-112.5, SE: 112.5-157.5
    # S: 157.5-202.5, SW: 202.5-247.5, W: 247.5-292.5, NW: 292.5-337.5
    directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    direction_index = int((angle_deg + 22.5) / 45) % 8
    
    return directions[direction_index]


def load_navigability(analysis_file):
    """Load navigability from gpt4o_analysis.txt."""
    navigability = {}
    
    if not analysis_file.exists():
        return navigability
    
    with open(analysis_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    for line in content.split('\n'):
        line = line.strip()
        if line.startswith('Navigable_'):
            parts = line.split(':')
            if len(parts) == 2:
                direction = parts[0].replace('Navigable_', '').strip()
                navigable = parts[1].strip().lower() == 'yes'
                navigability[direction] = navigable
    
    return navigability


def build_navigation_graph(analysis_dir, max_distance=1.5):
    """Build navigation graph from analysis directory."""
    print("=" * 80)
    print("  Build navigation graph")
    print("=" * 80)
    print(f" Analysis dir: {analysis_dir}")
    print(f" Max distance: {max_distance}m")
    print()
    
    analysis_path = Path(analysis_dir)
    
    if not analysis_path.exists():
        print(f" Directory not found: {analysis_dir}")
        return [], []
    
    # 1. Collect sampled nodes
    print(" Step 1: collect sampled nodes...")
    nodes = []
    
    for item in sorted(analysis_path.iterdir(), key=lambda item: item.name):
        if item.is_dir() and item.name.startswith('pos_'):
            pos = parse_position_from_dirname(item.name)
            if pos:
                nodes.append({
                    'position': pos,
                    'dir_name': item.name,
                    'analysis_file': item / "gpt4o_analysis.txt"
                })
    
    if len(nodes) < 2:
        print(f" Not enough nodes: {len(nodes)}")
        return nodes, []
    
    print(f"  Found {len(nodes)} nodes")
    print()
    
    # 2. Build edges
    print(" Step 2: build edges...")
    edges = []
    navigable_count = 0
    blocked_count = 0
    
    for i, node_a in enumerate(nodes):
        navigability_a = load_navigability(node_a['analysis_file'])
        
        if not navigability_a:
            print(f"   Skip {node_a['dir_name']} (no navigability)")
            continue
        
        neighbors = []
        for j, node_b in enumerate(nodes):
            if i == j:
                continue
            
            distance = calculate_distance_2d(node_a['position'], node_b['position'])
            if distance <= max_distance:
                neighbors.append((j, node_b, distance))
        
        for j, node_b, distance in neighbors:
            direction = calculate_direction(node_a['position'], node_b['position'])
            
            if navigability_a.get(direction, False):
                edges.append({
                    'from': i,
                    'to': j,
                    'distance': distance,
                    'direction': direction
                })
                navigable_count += 1
            else:
                blocked_count += 1
    
    print("  Edge build done:")
    print(f"      - Directed edges: {navigable_count}")
    print(f"      - Blocked edges: {blocked_count}")
    print()
    
    # 3. Convert to undirected graph
    print(" Step 3: convert to undirected graph...")
    
    # Track existing edges
    edge_set = set()
    for edge in edges:
        pair = (min(edge['from'], edge['to']), max(edge['from'], edge['to']))
        edge_set.add(pair)
    
    # Rebuild edges as bidirectional
    bidirectional_edges = []
    for (node_i, node_j) in edge_set:
        distance = calculate_distance_2d(nodes[node_i]['position'], nodes[node_j]['position'])
        
    # Add i->j edge
        direction_i_to_j = calculate_direction(nodes[node_i]['position'], nodes[node_j]['position'])
        bidirectional_edges.append({
            'from': node_i,
            'to': node_j,
            'distance': distance,
            'direction': direction_i_to_j
        })
        
    # Add j->i edge
        direction_j_to_i = calculate_direction(nodes[node_j]['position'], nodes[node_i]['position'])
        bidirectional_edges.append({
            'from': node_j,
            'to': node_i,
            'distance': distance,
            'direction': direction_j_to_i
        })
    
    print("  Conversion done:")
    print(f"      - Directed edges: {navigable_count}")
    print(f"      - Undirected pairs: {len(edge_set)}")
    print(f"      - Bidirectional edges: {len(bidirectional_edges)}")
    print(f"      - Avg degree: {len(bidirectional_edges) / len(nodes):.1f}")
    print()
    
    return nodes, bidirectional_edges


def get_top_down_frame(controller):
    """Get top-down frame."""
    # Get top-down camera properties
    event = controller.step(action="GetMapViewCameraProperties", raise_for_failure=True)
    pose = event.metadata["actionReturn"]
    
    # Add third-person camera
    event = controller.step(
        action="AddThirdPartyCamera",
        **pose,
        skyboxColor="white",
        raise_for_failure=True,
    )
    
    # Get top-down image
    topdown_image = event.third_party_camera_frames[-1]
    return Image.fromarray(topdown_image)


def visualize_navigation_graph(controller, house, nodes, edges, output_path):
    """Render navigation graph on top-down view."""
    print(" Step 3: render visualization...")
    
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
        img_pos = world_to_image(node['position'][0], node['position'][1])
        node_img_positions.append(img_pos)
    
    draw = ImageDraw.Draw(topdown_img)
    
    print(f"    Draw edges: {len(edges)}")
    for edge in edges:
        from_pos = node_img_positions[edge['from']]
        to_pos = node_img_positions[edge['to']]
        
        if (0 <= from_pos[0] < img_width and 0 <= from_pos[1] < img_height and
            0 <= to_pos[0] < img_width and 0 <= to_pos[1] < img_height):
            
            draw.line([from_pos, to_pos], fill=(0, 255, 0), width=2)
    
    print(f"    Draw nodes: {len(nodes)}")
    node_radius = 6
    for i, img_pos in enumerate(node_img_positions):
        if 0 <= img_pos[0] < img_width and 0 <= img_pos[1] < img_height:
            draw.ellipse(
                [img_pos[0]-node_radius, img_pos[1]-node_radius,
                 img_pos[0]+node_radius, img_pos[1]+node_radius],
                fill=(0, 100, 255),
                outline=(0, 0, 0),
                width=2
            )
    
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 10)
    except:
        font = ImageFont.load_default()
        font_small = font
    
    legend_width = 220
    legend_height = 126
    legend_x = img_width - legend_width - 15
    legend_y = img_height - legend_height - 15
    
    draw.rectangle(
        [legend_x - 5, legend_y - 5, legend_x + legend_width, legend_y + legend_height],
        fill=(255, 255, 255, 240),
        outline=(100, 100, 100),
        width=1
    )
    
    title_y = legend_y + 2
    draw.text((legend_x, title_y), "Navigation Graph", fill=(0, 0, 0), font=font)
    
    item_y = title_y + 20
    
    draw.ellipse([legend_x, item_y, legend_x+8, item_y+8], 
                 fill=(0, 100, 255), outline=(0, 0, 0), width=1)
    draw.text((legend_x + 14, item_y - 2), 
              f"Nodes: {len(nodes)}", 
              fill=(0, 100, 255), font=font_small)
    
    item_y += 18
    
    draw.line([legend_x, item_y+4, legend_x+12, item_y+4], 
              fill=(0, 255, 0), width=2)
    draw.text((legend_x + 14, item_y - 2), 
              f"Edges: {len(edges)}", 
              fill=(0, 180, 0), font=font_small)
    
    item_y += 18
    
    avg_degree = len(edges) / len(nodes) if len(nodes) > 0 else 0
    draw.text((legend_x, item_y), 
              f"Avg Degree: {avg_degree:.1f}", 
              fill=(100, 100, 100), font=font_small)
    
    item_y += 16
    
    draw.text((legend_x, item_y), 
              f"Type: Undirected", 
              fill=(100, 100, 100), font=font_small)
    
    item_y += 16
    draw.text((legend_x, item_y), 
              f"Max Distance: 1.5m", 
              fill=(100, 100, 100), font=font_small)
    
    topdown_img.save(output_path, "PNG")
    print(f"    Visualization saved: {output_path}")
    print()


def save_graph_data(nodes, edges, output_file):
    """Save graph data to JSON."""
    print(" Step 4: save graph data...")
    
    graph_data = {
        'nodes': [
            {
                'id': i,
                'position': {
                    'x': node['position'][0],
                    'z': node['position'][1]
                },
                'dir_name': node['dir_name']
            }
            for i, node in enumerate(nodes)
        ],
        'edges': [
            {
                'from': edge['from'],
                'to': edge['to'],
                'distance': float(edge['distance']),
                'direction': edge['direction']
            }
            for edge in edges
        ],
        'statistics': {
            'graph_type': 'undirected',
            'num_nodes': len(nodes),
            'num_edges': len(edges),
            'num_edge_pairs': len(edges) // 2,
            'avg_degree': len(edges) / len(nodes) if len(nodes) > 0 else 0,
            'max_distance': 1.5
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, indent=2, ensure_ascii=False)
    
    print(f"    Graph saved: {output_file}")
    print()


def print_graph_statistics(nodes, edges):
    """Print graph statistics."""
    print("=" * 80)
    print("  Graph stats (undirected)")
    print("=" * 80)
    
    print(" Basic stats:")
    print(f"   Nodes: {len(nodes)}")
    print(f"   Edges (bidirectional): {len(edges)}")
    print(f"   Edge pairs: {len(edges) // 2}")
    print(f"   Avg degree: {len(edges) / len(nodes):.2f}")
    print()
    
    print(" Degree distribution:")
    degrees = [0] * len(nodes)
    
    for edge in edges:
        degrees[edge['from']] += 1
    
    print(f"   Min: {min(degrees)}")
    print(f"   Max: {max(degrees)}")
    print(f"   Mean: {np.mean(degrees):.2f}")
    print(f"   Median: {int(np.median(degrees))}")
    print()
    
    print(" Edge direction distribution:")
    direction_counts = {}
    for edge in edges:
        direction = edge['direction']
        direction_counts[direction] = direction_counts.get(direction, 0) + 1
    
    for direction in ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']:
        count = direction_counts.get(direction, 0)
        percentage = 100 * count / len(edges) if len(edges) > 0 else 0
        bar = '█' * int(percentage / 2)
        print(f"   {direction:3s}: {count:3d} ({percentage:5.1f}%) {bar}")
    
    print()
    
    if edges:
        distances = [edge['distance'] for edge in edges]
        print(" Edge length distribution:")
        print(f"   Min: {min(distances):.3f}m")
        print(f"   Max: {max(distances):.3f}m")
        print(f"   Mean: {np.mean(distances):.3f}m")
        print(f"   Median: {np.median(distances):.3f}m")
    
    print()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Build navigation graph')
    parser.add_argument('--config', type=str, default=str(Path(__file__).parent / 'config.yaml'),
                        help='Config file path')
    parser.add_argument('--analysis_dir', type=str, default='procthor_analysis_house1',
                        help='Analysis directory (with samples and gpt4o_analysis.txt)')
    parser.add_argument('--max_distance', type=float, default=1.5,
                        help='Max connection distance (meters)')
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
    if 'graph_max_distance' in stage_defaults:
        parser.set_defaults(max_distance=stage_defaults['graph_max_distance'])
    
    args = parser.parse_args()
    
    print("Navigation graph builder")
    print(f"=" * 80)
    print("Config:")
    print(f"   Analysis dir: {args.analysis_dir}")
    print(f"   Max distance: {args.max_distance}m")
    print(f"   House index: {args.house_index}")
    print()
    
    nodes, edges = build_navigation_graph(args.analysis_dir, args.max_distance)
    
    if not nodes:
        print("Build failed: no nodes")
        return
    
    print_graph_statistics(nodes, edges)
    
    output_path = Path(args.analysis_dir)
    graph_file = output_path / "navigation_graph.json"
    save_graph_data(nodes, edges, graph_file)
    
    print("Initializing AI2-THOR controller...")
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
    
    controller = Controller(
        width=800,
        height=800,
        platform=CloudRendering,
        agentMode="bot",
    )
    print("    Controller ready")
    print()
    
    visualization_file = output_path / "navigation_graph.png"
    visualize_navigation_graph(controller, target_house, nodes, edges, visualization_file)
    
    controller.stop()
    
    print("=" * 80)
    print("  Done")
    print("=" * 80)
    print("Graph built")
    print(f"Stats: {len(nodes)} nodes, {len(edges)} edges")
    print("Outputs:")
    print(f"   - Graph: {graph_file}")
    print(f"   - Visualization: {visualization_file}")
    print()


if __name__ == "__main__":
    main()

