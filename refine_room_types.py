#!/usr/bin/env python3
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


def parse_gpt4o_analysis(file_path):
    """Parse gpt4o_analysis.txt."""
    room_type = None
    objects = []
    target_visible = None
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('Objects:'):
                objects_str = line.replace('Objects:', '').strip()
                if objects_str:
                    objects = [obj.strip() for obj in objects_str.split(',')]
            elif line.startswith('Room:'):
                room_type = line.replace('Room:', '').strip()
            elif line.startswith('Target_visible:'):
                target_visible = line.replace('Target_visible:', '').strip()
    
    return {
        'room_type': room_type,
        'objects': objects,
        'target_visible': target_visible
    }


def parse_position_from_dirname(dirname):
    """Parse coordinates from dir name (pos_X.XXX_Z.ZZZ)."""
    parts = dirname.split('_')
    if len(parts) >= 3:
        x = float(parts[1])
        z = float(parts[2])
        return (x, z)
    return None


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


def get_top_down_frame(controller):
    """Get top-down frame (aligned with value_assignment.py)."""
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


def world_to_image_coords(world_x, world_z, top_left_x, top_left_z, pixel_scale):
    """World to image coordinates."""
    img_x = int((world_x - top_left_x) * pixel_scale)
    img_y = int((world_z - top_left_z) * pixel_scale)
    return img_x, img_y


def generate_distinct_colors(n):
    """Generate n visually distinct colors."""
    import colorsys
    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.7 + (i % 3) * 0.1  # 0.7-0.9
        value = 0.8 + (i % 2) * 0.1  # 0.8-0.9
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(tuple(int(c * 255) for c in rgb))
    return colors


def visualize_refined_rooms(controller, house, positions_data, refined_room_mapping, output_path):
    """Render refined room labels on top-down view."""
    print("Rendering top-down visualization...")
    
    controller.reset(scene=house)
    
    top_down_image = get_top_down_frame(controller)
    img_width, img_height = top_down_image.size
    
    event = controller.step("GetMapViewCameraProperties", raise_for_failure=True)
    cam_position = event.metadata["actionReturn"]["position"]
    cam_orth_size = event.metadata["actionReturn"]["orthographicSize"]
    
    print(f"   Camera position: ({cam_position['x']:.2f}, {cam_position['z']:.2f})")
    print(f"   Orthographic size: {cam_orth_size:.2f}")
    print(f"   Image size: {img_width}px x {img_height}px")
    
    def world_to_image(x, z):
        """Convert world coordinates to top-down image coordinates."""
        img_x = img_width / 2 + (x - cam_position['x']) / (cam_orth_size * 2) * img_width
        img_y = img_height / 2 - (z - cam_position['z']) / (cam_orth_size * 2) * img_height
        return (int(img_x), int(img_y))
    
    draw = ImageDraw.Draw(top_down_image)
    
    try:
        font_large = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 10)
    except:
        try:
            font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
        except:
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()
    
    unique_rooms = sorted(set(refined_room_mapping.values()))
    room_colors = {}
    colors = generate_distinct_colors(len(unique_rooms))
    for i, room_type in enumerate(unique_rooms):
        room_colors[room_type] = colors[i]
    
    points_in_view = 0
    points_out_of_view = 0
    
    for i, pos_data in enumerate(positions_data):
        dir_name = pos_data['dir_name']
        position = pos_data['position']
        room_type = refined_room_mapping[dir_name]
        
        world_x, world_z = position[0], position[1]
        img_x, img_y = world_to_image(world_x, world_z)
        
        if i < 3:
            print(f"   {dir_name}: world({world_x:.2f}, {world_z:.2f}) -> image({img_x}, {img_y}) [{room_type}]")
        
        if 0 <= img_x < img_width and 0 <= img_y < img_height:
            points_in_view += 1
        else:
            points_out_of_view += 1
            print(f"   Point out of bounds: {dir_name} at ({world_x:.2f}, {world_z:.2f}) -> ({img_x}, {img_y})")
            img_x = max(0, min(img_x, img_width - 1))
            img_y = max(0, min(img_y, img_height - 1))
        
        color = room_colors[room_type]
        
        point_radius = 8
        draw.ellipse(
            [img_x - point_radius, img_y - point_radius,
             img_x + point_radius, img_y + point_radius],
            fill=color,
            outline='white',
            width=3
        )
        
        label_text = room_type
        
        bbox = draw.textbbox((0, 0), label_text, font=font_small)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        text_x = img_x - text_width // 2
        text_y = img_y + point_radius + 4
        
        padding = 2
        draw.rectangle(
            [text_x - padding, text_y - padding,
             text_x + text_width + padding, text_y + text_height + padding],
            fill='white',
            outline='black',
            width=1
        )
        
        draw.text((text_x, text_y), label_text, fill='black', font=font_small)
    
    legend_width = 230
    legend_height = 30 + len(unique_rooms) * 25
    legend_x = img_width - legend_width - 20
    legend_y = img_height - legend_height - 20
    
    draw.rectangle(
        [legend_x, legend_y, legend_x + legend_width, legend_y + legend_height],
        fill='white',
        outline='black',
        width=2
    )
    
    draw.text((legend_x + 10, legend_y + 5), "Refined Room Types", fill='black', font=font_large)
    
    for i, room_type in enumerate(unique_rooms):
        y_offset = legend_y + 30 + i * 25
        color = room_colors[room_type]
        
        square_size = 15
        draw.rectangle(
            [legend_x + 10, y_offset,
             legend_x + 10 + square_size, y_offset + square_size],
            fill=color,
            outline='black',
            width=1
        )
        
        draw.text((legend_x + 30, y_offset), room_type, fill='black', font=font_small)
    
    print(f"   Points in view: {points_in_view}")
    if points_out_of_view > 0:
        print(f"   Points out of view: {points_out_of_view} (clipped to bounds)")
    
    top_down_image.save(output_path)
    print(f"   Top-down view saved: {output_path}")
    print(f"   Labeled {len(positions_data)} points, {len(unique_rooms)} room types")


def main():
    parser = argparse.ArgumentParser(description='Refine room labels by object similarity')
    parser.add_argument('--config', type=str, default=str(Path(__file__).parent / 'config.yaml'),
                        help='Config file path')
    parser.add_argument('--analysis-dir', type=str, required=True,
                        help='Analysis directory (e.g., procthor_analysis_house1/)')
    parser.add_argument('--similarity-threshold', type=float, default=0.3,
                        help='Object similarity threshold (0-1)')
    parser.add_argument('--distance-weight', type=float, default=0.3,
                        help='Distance weight (0-1)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Dry run (no file updates)')
    parser.add_argument('--house-index', type=int, default=1,
                        help='ProcTHOR house index (for visualization)')
    parser.add_argument('--min-rooms', type=int, default=4,
                        help='Min rooms for filtering')
    parser.add_argument('--max-rooms', type=int, default=4,
                        help='Max rooms for filtering')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'],
                        help='Dataset split')
    parser.add_argument('--visualize', action='store_true', default=True,
                        help='Generate visualization (default on)')
    parser.add_argument('--no-visualize', action='store_false', dest='visualize',
                        help='Disable visualization')
    
    config_args, _ = parser.parse_known_args()
    config_path = Path(config_args.config)
    config = {}
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
    params = config.get('vlm_sp_rew_params', {})
    
    if 'tau_room' in params:
        parser.set_defaults(similarity_threshold=params['tau_room'])
    
    args = parser.parse_args()
    
    analysis_dir = Path(args.analysis_dir)
    
    if not analysis_dir.exists():
        print(f"Directory not found: {analysis_dir}")
        return
    
    print(f"Scanning directory: {analysis_dir}")
    print(f"Similarity threshold: {args.similarity_threshold}")
    print(f"Distance weight: {args.distance_weight}")
    print()
    
    all_positions = []
    
    for pos_dir in sorted(analysis_dir.iterdir()):
        if not pos_dir.is_dir() or not pos_dir.name.startswith('pos_'):
            continue
        
        analysis_file = pos_dir / 'gpt4o_analysis.txt'
        if not analysis_file.exists():
            continue
        
        position = parse_position_from_dirname(pos_dir.name)
        if position is None:
            continue
        
        analysis_data = parse_gpt4o_analysis(analysis_file)
        
        all_positions.append({
            'dir_name': pos_dir.name,
            'position': position,
            'room_type': analysis_data['room_type'],
            'objects': analysis_data['objects'],
            'target_visible': analysis_data['target_visible'],
            'file_path': analysis_file
        })
    
    print(f"Found {len(all_positions)} sampled points")
    print()
    
    room_groups = defaultdict(list)
    for pos_data in all_positions:
        room_groups[pos_data['room_type']].append(pos_data)
    
    print("Original room type distribution:")
    for room_type, positions in sorted(room_groups.items()):
        print(f"   {room_type}: {len(positions)} points")
    print()
    
    refined_room_mapping = {}  # {dir_name: new_room_type}
    total_original_rooms = len(room_groups)
    total_refined_rooms = 0
    
    for room_type, positions in sorted(room_groups.items()):
        if len(positions) == 1:
            refined_room_mapping[positions[0]['dir_name']] = room_type
            total_refined_rooms += 1
            print(f" {room_type}: only 1 point, keep label")
            continue
        
        print(f"Analyze {room_type}: {len(positions)} points")
        
        labels = cluster_rooms_by_similarity(
            positions,
            similarity_threshold=args.similarity_threshold,
            distance_weight=args.distance_weight
        )
        
        unique_labels = set(labels)
        n_clusters = len(unique_labels)
        
        if n_clusters == 1:
            for pos_data in positions:
                refined_room_mapping[pos_data['dir_name']] = room_type
            total_refined_rooms += 1
            print("   High similarity, keep as one room")
        else:
            total_refined_rooms += n_clusters
            
            cluster_groups = defaultdict(list)
            for i, label in enumerate(labels):
                cluster_groups[label].append(positions[i])
            
            cluster_id = 1
            for label in sorted(cluster_groups.keys()):
                cluster_positions = cluster_groups[label]
                new_room_type = f"{room_type}_{cluster_id}"
                
                for pos_data in cluster_positions:
                    refined_room_mapping[pos_data['dir_name']] = new_room_type
                
                cluster_objects = [p['objects'] for p in cluster_positions]
                if len(cluster_objects) > 1:
                    similarities = []
                    for i in range(len(cluster_objects)):
                        for j in range(i+1, len(cluster_objects)):
                            sim = calculate_jaccard_similarity(cluster_objects[i], cluster_objects[j])
                            similarities.append(sim)
                    avg_similarity = np.mean(similarities)
                else:
                    avg_similarity = 1.0
                
                print(f"   Cluster {cluster_id} -> {new_room_type}: {len(cluster_positions)} points, avg sim {avg_similarity:.3f}")
                
                for pos_data in cluster_positions:
                    objects_str = ', '.join(pos_data['objects'][:5])
                    if len(pos_data['objects']) > 5:
                        objects_str += f", ... (total {len(pos_data['objects'])})"
                    print(f"      - {pos_data['dir_name']}: {objects_str}")
                
                cluster_id += 1
        
        print()
    
    print("Summary:")
    print(f"   Original room types: {total_original_rooms}")
    print(f"   Refined room types: {total_refined_rooms}")
    print(f"   Split rooms: {total_refined_rooms - total_original_rooms}")
    print()
    
    final_room_distribution = defaultdict(int)
    for new_room_type in refined_room_mapping.values():
        final_room_distribution[new_room_type] += 1
    
    print("Refined room type distribution:")
    for room_type in sorted(final_room_distribution.keys()):
        count = final_room_distribution[room_type]
        print(f"   {room_type}: {count} points")
    print()
    
    if args.dry_run:
        print("Dry run: no files modified")
        print("\nRemove --dry-run to apply changes")
    else:
        print("Updating gpt4o_analysis.txt...")
        updated_count = 0
        
        for pos_data in all_positions:
            dir_name = pos_data['dir_name']
            old_room_type = pos_data['room_type']
            new_room_type = refined_room_mapping[dir_name]
            
            if old_room_type != new_room_type:
                update_gpt4o_analysis_file(pos_data['file_path'], new_room_type)
                updated_count += 1
        
        print(f"   Updated {updated_count} files")
        print(f"   Unchanged {len(all_positions) - updated_count} files")
        print()
        print("Done: room types refined")
    
    if args.visualize:
        print()
        print("=" * 80)
        print("Generate top-down visualization")
        print("=" * 80)
        
        try:
            print("Loading ProcTHOR-10k dataset...")
            dataset = prior.load_dataset("procthor-10k")
            dataset_split = dataset[args.split]
            
            print(f"Filtering houses (rooms: {args.min_rooms}-{args.max_rooms})...")
            filtered_houses = []
            for i in range(len(dataset_split)):
                house = dataset_split[i]
                num_rooms = len(house.get('rooms', []))
                
                if args.min_rooms and num_rooms < args.min_rooms:
                    continue
                if args.max_rooms and num_rooms > args.max_rooms:
                    continue
                
                filtered_houses.append(house)
                
                if len(filtered_houses) > args.house_index:
                    break
            
            print(f"   Found {len(filtered_houses)} matching houses")
            
            if args.house_index >= len(filtered_houses):
                print(f"   House index {args.house_index} out of range, using 0")
                house_index = 0
            else:
                house_index = args.house_index
            
            target_house = filtered_houses[house_index]
            print(f"   Using house index: {house_index}")
            
            print("Initializing AI2-THOR controller...")
            from ai2thor.platform import CloudRendering
            controller = Controller(
                width=800,
                height=800,
                platform=CloudRendering,
                agentMode="bot",
                gridSize=0.25,
                rotateStepDegrees=90,
                visibilityDistance=1.5,
                fieldOfView=90
            )
            
            controller.reset(scene=target_house)
            print("   Scene loaded")
            
            output_path = analysis_dir / 'refined_rooms_visualization.png'
            visualize_refined_rooms(
                controller,
                target_house,
                all_positions,
                refined_room_mapping,
                output_path
            )
            
            controller.stop()
            
            print()
            print(f"Visualization done: {output_path}")
            
        except Exception as e:
            print(f"Visualization failed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()

