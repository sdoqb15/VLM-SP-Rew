#!/usr/bin/env python3
"""
Scene Preprocessing Stage: Points Sampling and Image Collection.
Implements Poisson-disk sampling constraints and 8-view image capture.
"""

import sys
import numpy as np
from pathlib import Path
from PIL import Image
import yaml

# ProcTHOR dependencies
import prior
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering


def capture_eight_directions(controller, position, save_dir):
    """Capture 8 directions at a position and save images."""
    # 8 directions, 45 degrees each
    rotation_angles = [0, 45, 90, 135, 180, 225, 270, 315]
    angle_names = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    
    for i, angle in enumerate(rotation_angles):
        # Teleport to position and rotation
        event = controller.step(
            action="Teleport",
            position=position,
            rotation={"x": 0, "y": angle, "z": 0},
            horizon=10  # look forward
        )
        
        if event.metadata["lastActionSuccess"]:
            # Read frame
            rgb_image = np.array(event.frame)
            
            # Save image
            image_filename = f"{angle_names[i]}_{angle}deg.jpg"
            image_path = save_dir / image_filename
            
            pil_image = Image.fromarray(rgb_image)
            pil_image.save(image_path, "JPEG", quality=90)
            
            # print(f"   {angle_names[i]} ({angle} deg) saved")
        else:
            print(f"   Warning: teleport failed at {angle} deg")
            return False
    
    return True


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


def sample_positions(controller, house, num_samples=50, min_distance=1.0):
    """Sample reachable positions with minimum distance constraint."""
    # Load scene
    controller.reset(scene=house)
    
    # Get reachable positions
    event = controller.step(action="GetReachablePositions")
    reachable_positions = event.metadata["actionReturn"]
    
    if not reachable_positions:
        print("   Warning: no reachable positions")
        return []
    
    print(f"   Reachable positions: {len(reachable_positions)}")
    
    # Return all if few
    if len(reachable_positions) <= num_samples:
        sampled = reachable_positions
        print(f"   Sampled {len(sampled)} positions (all reachable)")
        return sampled
    
    # Poisson-disk sampling for min distance
    import random
    random.seed(42)  # deterministic
    
    sampled = []
    candidates = list(reachable_positions)
    
    first_idx = random.randint(0, len(candidates) - 1)
    sampled.append(candidates[first_idx])
    del candidates[first_idx]
    
    print(f"   Poisson sampling (min distance: {min_distance}m)...")
    
    # Iterative sampling
    attempts = 0
    max_attempts = len(candidates) * 3
    
    while len(sampled) < num_samples and candidates and attempts < max_attempts:
        attempts += 1
        
        # Random candidate
        candidate_idx = random.randint(0, len(candidates) - 1)
        candidate = candidates[candidate_idx]
        
        # Check distance to all sampled
        valid = True
        for sampled_pos in sampled:
            dist = np.sqrt(
                (candidate['x'] - sampled_pos['x'])**2 +
                (candidate['z'] - sampled_pos['z'])**2
            )
            if dist < min_distance:
                valid = False
                break
        
        if valid:
            # Accept candidate
            sampled.append(candidate)
            del candidates[candidate_idx]
            attempts = 0
            
            # Progress
            if len(sampled) % 10 == 0:
                print(f"      Sampled {len(sampled)}/{num_samples}")
        else:
            # Reject candidate
            del candidates[candidate_idx]
    
    if len(sampled) < num_samples:
        print(f"   Warning: only {len(sampled)} positions (target: {num_samples})")
        print(f"      Reason: space limited under min distance {min_distance}m")
    else:
        print(f"   Sampled {len(sampled)} positions")
    
    return sampled


def visualize_sampling(controller, house, sampled_positions, output_path, min_distance=1.0):
    """Render top-down view with sampled points."""
    print("\nGenerating sampling visualization...")
    
    # Top-down frame
    controller.reset(scene=house)
    topdown_img = get_top_down_frame(controller)
    
    # Camera properties for coordinate transform
    event = controller.step(action="GetMapViewCameraProperties")
    cam_position = event.metadata["actionReturn"]["position"]
    cam_orth_size = event.metadata["actionReturn"]["orthographicSize"]
    
    # Image size
    img_width, img_height = topdown_img.size
    
    # All reachable positions
    event = controller.step(action="GetReachablePositions")
    all_positions = event.metadata["actionReturn"]
    
    # Bounds
    all_xs = [p['x'] for p in all_positions]
    all_zs = [p['z'] for p in all_positions]
    x_min, x_max = min(all_xs), max(all_xs)
    z_min, z_max = min(all_zs), max(all_zs)
    
    # Draw
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(topdown_img)
    
    def world_to_image(x, z):
        """World to image coordinates."""
        # Orthographic projection centered at camera
        img_x = img_width / 2 + (x - cam_position['x']) / (cam_orth_size * 2) * img_width
        img_y = img_height / 2 - (z - cam_position['z']) / (cam_orth_size * 2) * img_height
        return (int(img_x), int(img_y))
    
    # Draw all reachable positions (gray)
    for pos in all_positions:
        img_pos = world_to_image(pos['x'], pos['z'])
        if 0 <= img_pos[0] < img_width and 0 <= img_pos[1] < img_height:
            draw.ellipse(
                [img_pos[0]-1, img_pos[1]-1, img_pos[0]+1, img_pos[1]+1],
                fill=(200, 200, 200, 128),
                outline=(150, 150, 150)
            )
    
    # Draw sampled positions (red)
    for i, pos in enumerate(sampled_positions):
        img_pos = world_to_image(pos['x'], pos['z'])
        if 0 <= img_pos[0] < img_width and 0 <= img_pos[1] < img_height:
            # Red point
            radius = 4
            draw.ellipse(
                [img_pos[0]-radius, img_pos[1]-radius, 
                 img_pos[0]+radius, img_pos[1]+radius],
                fill=(255, 0, 0, 200),
                outline=(200, 0, 0),
                width=1
            )
            
            # Index label every 10 points
            if i % 10 == 0:
                try:
                    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 10)
                except:
                    font = ImageFont.load_default()
                
                text = str(i+1)
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                # White background
                draw.rectangle(
                    [img_pos[0] - text_width//2 - 1, img_pos[1] - radius - text_height - 3,
                     img_pos[0] + text_width//2 + 1, img_pos[1] - radius - 1],
                    fill=(255, 255, 255, 200)
                )
                
                # Draw label
                draw.text(
                    (img_pos[0] - text_width//2, img_pos[1] - radius - text_height - 1),
                    text,
                    fill=(255, 0, 0),
                    font=font
                )
    
    # Legend (bottom-right)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 9)
    except:
        font = ImageFont.load_default()
        font_small = font
    
    # Legend placement
    legend_width = 190
    legend_height = 82
    legend_x = img_width - legend_width - 15
    legend_y = img_height - legend_height - 15
    
    # White semi-transparent background
    draw.rectangle(
        [legend_x - 5, legend_y - 5, legend_x + legend_width, legend_y + legend_height],
        fill=(255, 255, 255, 230),
        outline=(100, 100, 100),
        width=1
    )
    
    # Title
    title_y = legend_y + 2
    draw.text((legend_x, title_y), "Sampling Distribution", fill=(0, 0, 0), font=font)
    
    # Legend items
    item_y = title_y + 18
    
    # Reachable
    draw.ellipse([legend_x, item_y, legend_x+6, item_y+6], 
                 fill=(200, 200, 200), outline=(150, 150, 150))
    draw.text((legend_x + 12, item_y - 2), 
              f"Reachable ({len(all_positions)})", 
              fill=(100, 100, 100), font=font_small)
    
    item_y += 16
    
    # Sampled
    draw.ellipse([legend_x, item_y, legend_x+6, item_y+6], 
                 fill=(255, 0, 0), outline=(200, 0, 0), width=1)
    draw.text((legend_x + 12, item_y - 2), 
              f"Sampled ({len(sampled_positions)})", 
              fill=(255, 0, 0), font=font_small)
    
    item_y += 16
    
    # Scene info
    draw.text((legend_x, item_y), 
              f"Area: {x_max-x_min:.1f}m × {z_max-z_min:.1f}m", 
              fill=(100, 100, 100), font=font_small)
    
    item_y += 14
    
    # Min distance
    draw.text((legend_x, item_y), 
              f"Min dist: {min_distance:.1f}m", 
              fill=(100, 100, 100), font=font_small)
    
    # Save
    topdown_img.save(output_path, "PNG")
    print(f"   Visualization saved: {output_path}")
    print("   Stats:")
    print(f"      - Reachable positions: {len(all_positions)}")
    print(f"      - Sampled positions: {len(sampled_positions)}")
    print(f"      - Sampling rate: {len(sampled_positions)/len(all_positions)*100:.1f}%")
    print(f"      - Scene size: {x_max-x_min:.1f}m x {z_max-z_min:.1f}m")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect ProcTHOR scene views')
    parser.add_argument('--config', type=str, default=str(Path(__file__).parent / 'config.yaml'),
                        help='Config file path')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'],
                        help='ProcTHOR dataset split')
    parser.add_argument('--house-index', type=int, default=0,
                        help='House index (0-based)')
    parser.add_argument('--min-rooms', type=int, default=None,
                        help='Minimum room count for filtering')
    parser.add_argument('--max-rooms', type=int, default=None,
                        help='Maximum room count for filtering')
    parser.add_argument('--output_dir', type=str, default='procthor_scene_views',
                        help='Output directory')
    parser.add_argument('--num_samples', type=int, default=50,
                        help='Number of sampled positions per house')
    parser.add_argument('--min_distance', type=float, default=1.0,
                        help='Minimum distance between samples (meters)')
    parser.add_argument('--visualize', action='store_true', default=True,
                        help='Generate sampling visualization (default on)')
    parser.add_argument('--no-visualize', dest='visualize', action='store_false',
                        help='Disable visualization')
    parser.add_argument('--skip_capture', action='store_true',
                        help='Skip image capture, only create dirs and visualization')
    
    config_args, _ = parser.parse_known_args()
    config_path = Path(config_args.config)
    config = {}
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
    params = config.get('vlm_sp_rew_params', {})
    stage_defaults = config.get('stage_defaults', {})
    
    if 'N' in params:
        parser.set_defaults(num_samples=params['N'])
    if 'd_min' in params:
        parser.set_defaults(min_distance=params['d_min'])
    if 'collection_output_dir' in stage_defaults:
        parser.set_defaults(output_dir=stage_defaults['collection_output_dir'])
    
    args = parser.parse_args()
    
    print("ProcTHOR scene view collection")
    print(f"=" * 80)
    print("Config:")
    print(f"   Split: {args.split}")
    print(f"   House index: {args.house_index}")
    print(f"   Samples: {args.num_samples}")
    print(f"   Min distance: {args.min_distance}m")
    print(f"   Output: {args.output_dir}")
    if args.skip_capture:
        print("   Skip image capture")
    
    print()
    
    print("Loading ProcTHOR-10k dataset...")
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
        print(f"Error: house_index={args.house_index} out of range (found {len(houses)} houses)")
        sys.exit(1)
    
    print(f"   Found {len(houses)} matching houses")
    print(f"   Using house index: {args.house_index}")
    print()
    
    target_house = houses[args.house_index]
    num_rooms = len(target_house.get('rooms', []))
    print("Target house info:")
    print(f"   Rooms: {num_rooms}")
    print()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Initializing AI2-THOR controller...")
    controller = Controller(
        width=300,
        height=300,
        platform=CloudRendering,
        agentMode="bot",
        gridSize=0.25,
    )
    print("   Controller ready")
    print()
    
    print("Sampling positions...")
    positions = sample_positions(controller, target_house, args.num_samples, args.min_distance)
    
    if not positions:
        print("Error: no reachable positions")
        controller.stop()
        sys.exit(1)
    
    if args.visualize:
        visualization_path = output_dir / "sampling_visualization.png"
        visualize_sampling(controller, target_house, positions, visualization_path, args.min_distance)
    
    print()
    
    print("=" * 80)
    print("  Start collecting views")
    print("=" * 80)
    
    success_count = 0
    fail_count = 0
    skip_count = 0
    
    for i, position in enumerate(positions, 1):
        pos_name = f"pos_{position['x']:.3f}_{position['z']:.3f}"
        position_dir = output_dir / pos_name
        position_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[{i}/{len(positions)}] Position: ({position['x']:.2f}, {position['z']:.2f})")
        
        if args.skip_capture:
            skip_count += 1
            continue
        
        first_img = position_dir / "N_0deg.jpg"
        if first_img.exists():
            print("   Images exist, skip capture")
            skip_count += 1
            continue
        
        if not capture_eight_directions(controller, position, position_dir):
            print("   Error: capture failed")
            fail_count += 1
            continue
        print("   Capture done")
        success_count += 1
    
    controller.stop()
    
    print()
    print("=" * 80)
    print("  Done")
    print("=" * 80)
    print(f"Success: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Skipped: {skip_count}")
    print(f"Total: {len(positions)}")
    print()
    print(f"Output saved to: {output_dir}")
    

if __name__ == "__main__":
    main()

