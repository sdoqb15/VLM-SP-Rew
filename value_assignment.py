#!/usr/bin/env python3
"""
Scene Preprocessing Stage: Semantic Value Assignment.
Map shortest-path position and target visibility to a discrete value.

"""

import json
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import prior
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
import yaml


def parse_position_from_dirname(dirname):
    """Parse coordinates from dir name (pos_X.XXX_Z.ZZZ)."""
    parts = dirname.split('_')
    if len(parts) >= 3:
        try:
            x = float(parts[1])
            z = float(parts[2])
            return (x, z)
        except ValueError:
            return None
    return None


def get_target_visible(analysis_file):
    """Read Target_visible from gpt4o_analysis.txt."""
    if not analysis_file.exists():
        return None
    
    with open(analysis_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    for line in content.split('\n'):
        if line.startswith('Target_visible:'):
            visible_str = line.replace('Target_visible:', '').strip().lower()
            return (visible_str == 'yes')
    
    return None


def calculate_value(target_visible, position, params):
    """Compute value from visibility and path position."""
    delta_vis = params.get('delta_vis', [2, 5])
    delta_novis = params.get('delta_novis', 5)
    alpha_k = params.get('alpha_k', [5, 4, 3])
    beta_k = params.get('beta_k', [2, 1, 0, -1])
    k_vis = params.get('K_vis', len(alpha_k))
    
    if target_visible:
        for idx, band in enumerate(delta_vis):
            if position <= band:
                return alpha_k[min(idx, len(alpha_k) - 1)]
        last_idx = min(k_vis - 1, len(alpha_k) - 1)
        return alpha_k[last_idx]
    
    if position <= delta_novis:
        return beta_k[0]
    
    step = (beta_k[1] - beta_k[0]) if len(beta_k) > 1 else -1
    steps = int(np.ceil((position - delta_novis) / delta_novis))
    return beta_k[0] + steps * step


def assign_values(analysis_dir, shortest_paths_file, params):
    """Assign values to all sampled positions."""
    print("Assign values from shortest-path position...")
    print("=" * 80)
    
    with open(shortest_paths_file, 'r', encoding='utf-8') as f:
        paths_data = json.load(f)
    
    average_positions = paths_data['average_positions']
    goal_node_id = str(paths_data['goal_node_id'])
    
    print(f"   Goal node ID: {goal_node_id}")
    print(f"   Total sampled points: {len(average_positions)}")
    print()
    
    results = {}
    value_distribution = {}
    
    for node_id_str, node_info in average_positions.items():
        if node_id_str == goal_node_id:
            continue
        
        dir_name = node_info['dir_name']
        position = node_info['position']
        
        coords = parse_position_from_dirname(dir_name)
        if coords is None:
            continue
        
        x, z = coords
        
        pos_dir = analysis_dir / dir_name
        analysis_file = pos_dir / "gpt4o_analysis.txt"
        
        target_visible = get_target_visible(analysis_file)
        
        if target_visible is None:
            print(f"   Skip {dir_name}: cannot read Target_visible")
            continue
        
        value = calculate_value(target_visible, position, params)
        
        delta_vis = params.get('delta_vis', [2, 5])
        delta_novis = params.get('delta_novis', 5)
        alpha_k = params.get('alpha_k', [5, 4, 3])
        beta_k = params.get('beta_k', [2, 1, 0, -1])
        
        if target_visible:
            if position <= delta_vis[0]:
                reason = f"Target visible, position={position:.1f} (<= {delta_vis[0]})"
            elif len(delta_vis) > 1 and position <= delta_vis[1]:
                reason = f"Target visible, position={position:.1f} ({delta_vis[0]}<p<= {delta_vis[1]})"
            else:
                reason = f"Target visible, position={position:.1f} (> {delta_vis[-1]})"
        else:
            if position <= delta_novis:
                reason = f"Target not visible, position={position:.1f} (<= {delta_novis})"
            else:
                reason = f"Target not visible, position={position:.1f} (> {delta_novis})"
        
        value_file = pos_dir / "value.txt"
        with open(value_file, 'w', encoding='utf-8') as f:
            f.write(f"Value: {value}\n")
            f.write(f"Position: {position:.2f}\n")
            f.write(f"Reason: {reason}\n")
        
        results[int(node_id_str)] = {
            'position': position,
            'dir_name': dir_name,
            'x': x,
            'z': z,
            'value': value,
            'reason': reason
        }
        
        value_distribution[value] = value_distribution.get(value, 0) + 1
    
    print("Value assignment done")
    print()
    print("Value distribution:")
    for value in sorted(value_distribution.keys(), reverse=True):
        count = value_distribution[value]
        print(f"   Value {value:2d}: {count:3d} positions")
    print()
    
    return results


def get_top_down_frame(controller):
    """Get top-down frame."""
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


def visualize_with_values(controller, house, results, output_path, analysis_dir):
    """Render top-down view with value colors."""
    print("Rendering value visualization...")
    
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
    
    event = controller.step(action="GetReachablePositions")
    all_positions = event.metadata["actionReturn"]
    
    draw = ImageDraw.Draw(topdown_img)
    
    for pos in all_positions:
        img_pos = world_to_image(pos['x'], pos['z'])
        if 0 <= img_pos[0] < img_width and 0 <= img_pos[1] < img_height:
            draw.ellipse(
                [img_pos[0]-1, img_pos[1]-1, img_pos[0]+1, img_pos[1]+1],
                fill=(200, 200, 200, 128),
                outline=(150, 150, 150)
            )
    
    values = [r['value'] for r in results.values()]
    min_value = min(values)
    max_value = max(values)
    
    print(f"   Value range: {min_value} to {max_value}")
    
    def value_to_color(value):
        """Map value to RGB color for visualization."""
        if max_value == min_value:
            return (0, 255, 0)
        
        normalized = (value - min_value) / (max_value - min_value)
        
        if normalized >= 0.66:
            t = (normalized - 0.66) / 0.34
            r = 0
            g = int(255 * (1 - t))  # 255 -> 0
            b = int(255 * t)  # 0 -> 255
        elif normalized >= 0.33:
            t = (normalized - 0.33) / 0.33
            r = int(255 * (1 - t))  # 255 -> 0
            g = 255
            b = 0
        else:
            t = normalized / 0.33
            r = 255
            g = int(255 * t)  # 0 -> 255
            b = 0
        
        return (r, g, b)
    
    sorted_results = sorted(results.values(), key=lambda r: r['value'])
    
    try:
        font_tiny = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 8)
    except:
        font_tiny = ImageFont.load_default()
    
    for result in sorted_results:
        img_pos = world_to_image(result['x'], result['z'])
        if 0 <= img_pos[0] < img_width and 0 <= img_pos[1] < img_height:
            color = value_to_color(result['value'])
            radius = 5
            
            draw.ellipse(
                [img_pos[0]-radius, img_pos[1]-radius,
                 img_pos[0]+radius, img_pos[1]+radius],
                fill=color,
                outline=(0, 0, 0),
                width=1
            )
            
            pos_dir = Path(analysis_dir) / result['dir_name']
            analysis_file = pos_dir / "gpt4o_analysis.txt"
            target_visible = get_target_visible(analysis_file)
            visible_str = "T" if target_visible else "F"
            
            text_lines = [
                f"{visible_str}",
                f"p:{result['position']:.1f}",
                f"v:{result['value']}"
            ]
            
            text_y = img_pos[1] + radius + 3
            line_spacing = 11
            
            for i, line in enumerate(text_lines):
                bbox = draw.textbbox((0, 0), line, font=font_tiny)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                text_x = img_pos[0] - text_width // 2
                current_y = text_y + i * line_spacing
                
                draw.rectangle(
                    [text_x - 1, current_y - 1, 
                     text_x + text_width + 1, current_y + text_height + 1],
                    fill=(255, 255, 255, 200)
                )
                
                draw.text((text_x, current_y), line, fill=(0, 0, 0), font=font_tiny)
    
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 9)
    except:
        font = ImageFont.load_default()
        font_small = font
    
    legend_width = 200
    legend_height = 135
    legend_x = img_width - legend_width - 15
    legend_y = img_height - legend_height - 15
    
    draw.rectangle(
        [legend_x - 5, legend_y - 5, legend_x + legend_width, legend_y + legend_height],
        fill=(255, 255, 255, 240),
        outline=(100, 100, 100),
        width=1
    )
    
    title_y = legend_y + 2
    draw.text((legend_x, title_y), "Path-Based Value", fill=(0, 0, 0), font=font)
    
    item_y = title_y + 20
    
    gradient_width = 150
    gradient_height = 20
    gradient_x = legend_x + 10
    gradient_y = item_y
    
    for i in range(gradient_width):
        normalized = i / gradient_width
        if normalized <= 0.33:
            t = normalized / 0.33
            r = 255
            g = int(255 * t)
            b = 0
        elif normalized <= 0.66:
            t = (normalized - 0.33) / 0.33
            r = int(255 * (1 - t))
            g = 255
            b = 0
        else:
            t = (normalized - 0.66) / 0.34
            r = 0
            g = int(255 * (1 - t))
            b = int(255 * t)
        
        draw.line(
            [(gradient_x + i, gradient_y), (gradient_x + i, gradient_y + gradient_height)],
            fill=(r, g, b),
            width=1
        )
    
    draw.rectangle(
        [gradient_x, gradient_y, gradient_x + gradient_width, gradient_y + gradient_height],
        outline=(0, 0, 0),
        width=1
    )
    
    draw.text((gradient_x, gradient_y + gradient_height + 5),
              f"{min_value}", fill=(0, 0, 0), font=font_small)
    draw.text((gradient_x + gradient_width - 20, gradient_y + gradient_height + 5),
              f"{max_value}", fill=(0, 0, 0), font=font_small)
    
    item_y += gradient_height + 30
    
    value_dist = {}
    for r in results.values():
        value_dist[r['value']] = value_dist.get(r['value'], 0) + 1
    
    draw.text((legend_x, item_y),
              f"Total: {len(results)} points", fill=(0, 0, 0), font=font_small)
    item_y += 15
    
    draw.text((legend_x, item_y),
              f"Range: [{min_value}, {max_value}]", fill=(0, 0, 0), font=font_small)
    item_y += 15
    
    draw.text((legend_x, item_y),
              f"Labels: T/F, p:pos, v:val", fill=(0, 0, 0), font=font_small)
    
    topdown_img.save(output_path, "PNG")
    print(f"    Visualization saved: {output_path}")
    print()


def visualize_heatmap(controller, house, results, output_path, min_distance=1.0):
    """Render value heatmap."""
    print("Rendering value heatmap...")
    
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
    
    event = controller.step(action="GetReachablePositions")
    all_positions = event.metadata["actionReturn"]
    
    room_mask = Image.new('L', (img_width, img_height), 0)
    mask_draw = ImageDraw.Draw(room_mask)
    
    mask_point_radius = 4
    for pos in all_positions:
        img_pos = world_to_image(pos['x'], pos['z'])
        if 0 <= img_pos[0] < img_width and 0 <= img_pos[1] < img_height:
            mask_draw.ellipse(
                [img_pos[0]-mask_point_radius, img_pos[1]-mask_point_radius,
                 img_pos[0]+mask_point_radius, img_pos[1]+mask_point_radius],
                fill=255
            )
    
    room_mask = room_mask.filter(ImageFilter.MaxFilter(size=25))
    room_mask = room_mask.filter(ImageFilter.MaxFilter(size=25))
    
    pixel_scale = img_width / (cam_orth_size * 2)
    radiation_radius = int(min_distance * pixel_scale * 1.8)
    
    print(f"    Radiation radius: {radiation_radius} px")
    print("    Compositing per-point layers...")
    
    values = [r['value'] for r in results.values()]
    min_value = min(values)
    max_value = max(values)
    
    def value_to_color(value):
        """Map value to RGB color for heatmap visualization."""
        if max_value == min_value:
            return (0, 255, 0)
        
        normalized = (value - min_value) / (max_value - min_value)
        
        if normalized >= 0.66:
            t = (normalized - 0.66) / 0.34
            r = 0
            g = int(255 * (1 - t))  # 255 -> 0
            b = int(255 * t)  # 0 -> 255
        elif normalized >= 0.33:
            t = (normalized - 0.33) / 0.33
            r = int(255 * (1 - t))  # 255 -> 0
            g = 255
            b = 0
        else:
            t = normalized / 0.33
            r = 255
            g = int(255 * t)  # 0 -> 255
            b = 0
        
        return (r, g, b)
    
    heatmap_layer = Image.new('RGBA', (img_width, img_height), (0, 0, 0, 0))
    
    for idx, result in enumerate(results.values()):
        img_pos = world_to_image(result['x'], result['z'])
        
        if not (0 <= img_pos[0] < img_width and 0 <= img_pos[1] < img_height):
            continue
        
        point_layer = Image.new('RGBA', (img_width, img_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(point_layer)
        
        color = value_to_color(result['value'])
        
        steps = 20
        for i in range(steps, 0, -1):
            current_radius = int(radiation_radius * i / steps)
            alpha = int(180 * (steps - i + 1) / steps)
            
            current_color = color + (alpha,)
            draw.ellipse(
                [img_pos[0]-current_radius, img_pos[1]-current_radius,
                 img_pos[0]+current_radius, img_pos[1]+current_radius],
                fill=current_color,
                outline=None
            )
        
        heatmap_layer = Image.alpha_composite(heatmap_layer, point_layer)
        
        if (idx + 1) % 10 == 0:
            print(f"      Processed {idx + 1}/{len(results)} points")
    
    heatmap_layer = heatmap_layer.filter(ImageFilter.GaussianBlur(radius=3))
    
    heatmap_array = np.array(heatmap_layer)
    mask_array = np.array(room_mask)
    heatmap_array[:, :, 3] = (heatmap_array[:, :, 3].astype(float) * (mask_array / 255.0)).astype(np.uint8)
    heatmap_layer = Image.fromarray(heatmap_array)
    
    topdown_img = topdown_img.convert('RGBA')
    result_img = Image.alpha_composite(topdown_img, heatmap_layer)
    
    draw_points = ImageDraw.Draw(result_img)
    sorted_results = sorted(results.values(), key=lambda r: r['value'])
    
    for result in sorted_results:
        img_pos = world_to_image(result['x'], result['z'])
        if 0 <= img_pos[0] < img_width and 0 <= img_pos[1] < img_height:
            color = value_to_color(result['value'])
            radius = 5
            
            draw_points.ellipse(
                [img_pos[0]-radius, img_pos[1]-radius,
                 img_pos[0]+radius, img_pos[1]+radius],
                fill=color,
                outline=(0, 0, 0),
                width=2
            )
    
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 9)
    except:
        font = ImageFont.load_default()
        font_small = font
    
    draw = ImageDraw.Draw(result_img)
    
    legend_width = 200
    legend_height = 100
    legend_x = img_width - legend_width - 15
    legend_y = img_height - legend_height - 15
    
    draw.rectangle(
        [legend_x - 5, legend_y - 5, legend_x + legend_width, legend_y + legend_height],
        fill=(255, 255, 255, 240),
        outline=(100, 100, 100),
        width=1
    )
    
    title_y = legend_y + 2
    draw.text((legend_x, title_y), "Path-Based Heatmap", fill=(0, 0, 0), font=font)
    
    item_y = title_y + 25
    
    gradient_width = 150
    gradient_height = 20
    gradient_x = legend_x + 10
    
    for i in range(gradient_width):
        normalized = i / gradient_width
        if normalized <= 0.33:
            t = normalized / 0.33
            r = 255
            g = int(255 * t)
            b = 0
        elif normalized <= 0.66:
            t = (normalized - 0.33) / 0.33
            r = int(255 * (1 - t))
            g = 255
            b = 0
        else:
            t = (normalized - 0.66) / 0.34
            r = 0
            g = int(255 * (1 - t))
            b = int(255 * t)
        
        draw.line(
            [(gradient_x + i, item_y), (gradient_x + i, item_y + gradient_height)],
            fill=(r, g, b, 255),
            width=1
        )
    
    draw.rectangle(
        [gradient_x, item_y, gradient_x + gradient_width, item_y + gradient_height],
        outline=(0, 0, 0),
        width=1
    )
    
    draw.text((gradient_x, item_y + gradient_height + 5),
              f"{min_value}", fill=(0, 0, 0), font=font_small)
    draw.text((gradient_x + gradient_width - 20, item_y + gradient_height + 5),
              f"{max_value}", fill=(0, 0, 0), font=font_small)
    
    result_img.save(output_path, "PNG")
    print(f"    Heatmap saved: {output_path}")
    print()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Assign values from shortest-path position')
    parser.add_argument('--config', type=str, default=str(Path(__file__).parent / 'config.yaml'),
                        help='Config file path')
    parser.add_argument('--analysis_dir', type=str, default='procthor_analysis_house1',
                        help='Analysis directory')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'],
                        help='ProcTHOR dataset split')
    parser.add_argument('--house-index', type=int, default=1,
                        help='House index')
    parser.add_argument('--min-rooms', type=int, default=4,
                        help='Min rooms')
    parser.add_argument('--max-rooms', type=int, default=4,
                        help='Max rooms')
    parser.add_argument('--min-distance', type=float, default=1.0,
                        help='Min distance for heatmap')
    
    config_args, _ = parser.parse_known_args()
    config_path = Path(config_args.config)
    config = {}
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
    params = config.get('vlm_sp_rew_params', {})
    stage_defaults = config.get('stage_defaults', {})
    
    if 'analysis_dir' in stage_defaults:
        parser.set_defaults(analysis_dir=stage_defaults['analysis_dir'])
    if 'value_min_distance' in stage_defaults:
        parser.set_defaults(min_distance=stage_defaults['value_min_distance'])
    
    args = parser.parse_args()
    
    analysis_dir = Path(args.analysis_dir)
    
    print("Shortest-path value assignment")
    print("=" * 80)
    print(f" Analysis dir: {analysis_dir}")
    print()
    
    shortest_paths_file = analysis_dir / "shortest_paths_analysis.json"
    if not shortest_paths_file.exists():
        print(f" Missing file: {shortest_paths_file}")
        print("   Run shortest_paths_computation.py first")
        return
    
    results = assign_values(analysis_dir, shortest_paths_file, params)
    
    if not results:
        print(" No valid sampled points")
        return
    
    print(" Loading ProcTHOR scene...")
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
        print(f" house_index={args.house_index} out of range")
        return
    
    target_house = houses[args.house_index]
    print(f"    Using house index: {args.house_index}")
    print()
    
    print("  Initializing AI2-THOR controller...")
    controller = Controller(
        width=800,
        height=800,
        platform=CloudRendering,
        agentMode="bot",
    )
    print("    Controller ready")
    print()
    
    viz_path = analysis_dir / "path_value_visualization.png"
    visualize_with_values(controller, target_house, results, viz_path, analysis_dir)
    
    heatmap_path = analysis_dir / "path_value_heatmap.png"
    visualize_heatmap(controller, target_house, results, heatmap_path, args.min_distance)
    
    controller.stop()
    
    print("=" * 80)
    print(" Done")
    print("=" * 80)
    print(" Outputs:")
    print(f"   - Values: {analysis_dir}/pos_*/value.txt")
    print(f"   - Visualization: {viz_path}")
    print(f"   - Heatmap: {heatmap_path}")
    print()


if __name__ == "__main__":
    main()

