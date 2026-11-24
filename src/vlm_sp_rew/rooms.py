"""
Scene Preprocessing Stage:
Room label refinement via object-similarity clustering.
"""


import os


import argparse


from pathlib import Path


from collections import defaultdict


import numpy as np


from sklearn.cluster import AgglomerativeClustering


from PIL import Image, ImageDraw, ImageFont


import prior


from ai2thor.controller import Controller


import yaml


def calculate_jaccard_similarity(objects1, objects2):
    """Jaccard similarity of object lists."""
    set1 = set(objects1)
    set2 = set(objects2)
    
    if len(set1) == 0 and len(set2) == 0:
        return 1.0
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    if union == 0:
        return 0.0
    
    return intersection / union


def calculate_distance_2d(pos1, pos2):
    """2D distance between two points."""
    return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)


def cluster_rooms_by_similarity(positions_data, similarity_threshold=0.3, distance_weight=0.3):
    """Cluster room labels using object similarity and spatial distance."""
    n = len(positions_data)
    
    if n == 1:
        return [0]
    
    similarity_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                similarity_matrix[i][j] = 1.0
            else:
                objects_sim = calculate_jaccard_similarity(
                    positions_data[i]['objects'],
                    positions_data[j]['objects']
                )
                
                distance = calculate_distance_2d(
                    positions_data[i]['position'],
                    positions_data[j]['position']
                )
                
                distance_sim = np.exp(-(distance ** 2) / (2 * 3.0 ** 2))
                
                combined_sim = (1 - distance_weight) * objects_sim + distance_weight * distance_sim
                similarity_matrix[i][j] = combined_sim
    
    distance_matrix = 1.0 - similarity_matrix
    
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=1.0 - similarity_threshold,
        metric='precomputed',
        linkage='average'
    )
    
    labels = clustering.fit_predict(distance_matrix)
    
    return labels


def update_gpt4o_analysis_file(file_path, new_room_type):
    """Update Room label in gpt4o_analysis.txt."""
    lines = []
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('Room:'):
                lines.append(f'Room: {new_room_type}\n')
            else:
                lines.append(line)
    
    with open(file_path, 'w') as f:
        f.writelines(lines)

