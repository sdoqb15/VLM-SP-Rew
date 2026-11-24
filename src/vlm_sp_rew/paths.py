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

