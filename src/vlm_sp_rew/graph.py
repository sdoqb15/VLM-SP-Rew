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

