#!/usr/bin/env python3
"""
Scene Preprocessing Stage: VLM-Based Semantic Annotation.
Implements prompt-based semantic labels for objects, room type, visibility, and navigability.
"""

import os
import sys
import base64
from pathlib import Path
import yaml
from PIL import Image
from release_audit import configure, request_annotation


def load_api_key():
    """Load OpenAI API key from config.yaml, fallback to env."""
    config_path = Path(__file__).parent / "config.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        key = data.get("openai_api_key") or data.get("OPENAI_API_KEY")
        if key:
            return key
    return os.environ.get("OPENAI_API_KEY")


# No client is created on import. Newly executed calls require a pinned snapshot and budget.


def encode_image(image_path):
    """Encode image to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def analyze_navigability(position_dir):
    """Analyze navigability for 8 directions."""
    directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    images = {}

    for direction in directions:
        angle_map = {'N': '0', 'NE': '45', 'E': '90', 'SE': '135',
                     'S': '180', 'SW': '225', 'W': '270', 'NW': '315'}
        img_path = position_dir / f"{direction}_{angle_map[direction]}deg.jpg"

        if img_path.exists():
            images[direction] = encode_image(img_path)
        else:
            print(f"   Warning: missing image: {img_path.name}")
            return None

    content = [
        {
            "type": "text",
            "text": f"""You are analyzing whether a robot can move forward 2 meters in each direction from its current position.

You will see 8 directional views (N, NE, E, SE, S, SW, W, NW - every 45 degrees). For EACH direction, determine if the robot can move forward approximately 0.5 meters:

Answer in this EXACT format (one line per direction):

Navigable_N: [yes/no]
Navigable_NE: [yes/no]
Navigable_E: [yes/no]
Navigable_SE: [yes/no]
Navigable_S: [yes/no]
Navigable_SW: [yes/no]
Navigable_W: [yes/no]
Navigable_NW: [yes/no]

Note:
-Assuming the robot is small enough to enter small spaces
-if uncertain, answer "yes"

Now analyze these views:"""
        }
    ]

    for direction in directions:
        content.append({
            "type": "text",
            "text": f"\n{direction} view (facing this direction):"
        })
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{images[direction]}",
                "detail": "high"
            }
        })

    try:
        return request_annotation(position_dir, 'navigability', content, max_tokens=200, temperature=0.2)

    except Exception as e:
        print(f"   Error: GPT-4o navigability analysis failed: {e}")
        return None


def analyze_position_with_images(position_dir, target_image_path):
    """Analyze 8 views at a position."""
    directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    images = {}

    for direction in directions:
        angle_map = {'N': '0', 'NE': '45', 'E': '90', 'SE': '135',
                     'S': '180', 'SW': '225', 'W': '270', 'NW': '315'}
        img_path = position_dir / f"{direction}_{angle_map[direction]}deg.jpg"

        if img_path.exists():
            images[direction] = encode_image(img_path)
        else:
            print(f"   Warning: missing image: {img_path.name}")
            return None

    if isinstance(target_image_path, str):
        target_image_path = Path(target_image_path)
    target_base64 = encode_image(target_image_path)

    target_objects = [
        "HousePlant", "Television", "Laptop", "Mug", "Bowl", "GarbageCan",
        "AlarmClock", "Apple", "Vase", "SprayBottle",
        "Chair", "Sofa", "Bed", "DiningTable",
        "Fridge", "Toilet", "Sink",
        "Dresser", "ShelvingUnit", "TVStand", "CounterTop",
        "Painting", "Cart", "Plunger", "LaundryHamper",
        "DogBed", "GarbageBag"
    ]

    content = [
        {
            "type": "text",
            "text": f"""You are analyzing a robot's location in a ProcTHOR indoor scene. You will see 8 directional views (N, NE, E, SE, S, SW, W, NW - every 45 degrees) from this position, plus a target object image.

Please analyze and answer in this EXACT format:

Objects: [list items from: {', '.join(target_objects)}]
Room: [choose ONE from: bathroom, kitchen, bedroom, living_room, hallway, other]
Target_visible: [yes/no - can you see the target object shown in the reference image in ANY of the 8 views, regardless of distance]

Rules:
- Only list objects you can CLEARLY see in the images
- Choose the most appropriate room type
- Target is visible if you can recognize the same type of object in any view
- Keep answers concise, comma-separated for objects

Now analyze these images:"""
        }
    ]

    for direction in directions:
        content.append({
            "type": "text",
            "text": f"\n{direction} view:"
        })
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{images[direction]}",
                "detail": "high"
            }
        })

    content.append({
        "type": "text",
        "text": "\nTARGET object (reference):"
    })
    content.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{target_base64}",
            "detail": "high"
        }
    })

    try:
        return request_annotation(position_dir, 'objects_room_visibility', content, max_tokens=300, temperature=0.3)

    except Exception as e:
        print(f"   Error: GPT-4o call failed: {e}")
        return None


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Analyze ProcTHOR scene views (from images)')
    parser.add_argument('--execute-api', action='store_true', help='Execute newly budgeted annotation calls; default is offline planning only')
    parser.add_argument('--model', default=None, help='Actual dated API model snapshot; never guess a historical version')
    parser.add_argument('--max-api-calls', type=int, default=0, help='Hard limit on newly executed calls')
    parser.add_argument('--config', type=str, default=str(Path(__file__).parent / 'config.yaml'),
                        help='Config file path')
    parser.add_argument('--output_dir', type=str, default='procthor_scene_views',
                        help='Collection output directory')
    parser.add_argument('--target_image', type=str, default='goal.jpg',
                        help='Target reference image (default: goal.jpg in this directory)')
    parser.add_argument('--target_object', type=str, default='HousePlant',
                        help='Target object type (log only)')
    parser.add_argument('--skip_objects', action='store_true',
                        help='Skip object/room/visibility analysis')
    parser.add_argument('--analyze_navigability', action='store_true', default=True,
                        help='Analyze navigability (default on)')
    parser.add_argument('--no-analyze-navigability', dest='analyze_navigability', action='store_false',
                        help='Disable navigability analysis')

    config_args, _ = parser.parse_known_args()
    config_path = Path(config_args.config)
    config = {}
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
    stage_defaults = config.get('stage_defaults', {})
    
    if 'target_image' in stage_defaults:
        parser.set_defaults(target_image=stage_defaults['target_image'])
    if 'target_object' in stage_defaults:
        parser.set_defaults(target_object=stage_defaults['target_object'])
    if 'analysis_dir' in stage_defaults:
        parser.set_defaults(output_dir=stage_defaults['analysis_dir'])
    
    args = parser.parse_args()

    configure(execute=args.execute_api, model=args.model, max_calls=args.max_api_calls)
    if not args.execute_api:
        import json
        root = Path(args.output_dir)
        positions = sorted(p.name for p in root.glob('pos_*') if p.is_dir())
        print(json.dumps(dict(mode='OFFLINE_PLAN_ONLY', points=len(positions), maximum_calls_per_point=int(not args.skip_objects)+int(args.analyze_navigability), requested_snapshot=args.model, actual_API_calls=0, annotation_outputs_created=False)))
        return

    if not args.skip_objects:
        if not args.target_image:
            print("Error: --target_image is required for object analysis")
            sys.exit(1)
        target_image_path = Path(args.target_image)
        if not target_image_path.is_absolute():
            target_image_path = Path(__file__).parent / target_image_path
        if not target_image_path.exists():
            print(f"Error: target image not found: {target_image_path}")
            sys.exit(1)
    else:
        target_image_path = None

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"Error: output directory not found: {output_dir}")
        sys.exit(1)

    position_dirs = sorted([p for p in output_dir.iterdir() if p.is_dir() and p.name.startswith("pos_")])
    if not position_dirs:
        print(f"Error: no position dirs found: {output_dir}")
        sys.exit(1)

    print("ProcTHOR scene analysis")
    print(f"=" * 80)
    print("Config:")
    print(f"   Output: {output_dir}")
    print(f"   Target object: {args.target_object}")
    if target_image_path:
        print(f"   Target image: {target_image_path}")
    if args.skip_objects:
        print("   Skip object/room analysis")
    if args.analyze_navigability:
        print("   Navigability analysis enabled")
    else:
        print("   Navigability analysis disabled")
    print()

    success_count = 0
    fail_count = 0
    skip_count = 0

    for i, position_dir in enumerate(position_dirs, 1):
        print(f"[{i}/{len(position_dirs)}] Position dir: {position_dir.name}")

        analysis_file = position_dir / "gpt4o_analysis.txt"
        needs_object_analysis = not args.skip_objects
        needs_nav_analysis = args.analyze_navigability

        if analysis_file.exists():
            with open(analysis_file, 'r', encoding='utf-8') as f:
                content = f.read()

            has_objects = 'Objects:' in content
            has_room = 'Room:' in content
            has_target = 'Target_visible:' in content
            has_navigability = 'Navigable_N:' in content

            if has_objects and has_room and has_target:
                needs_object_analysis = False
                print("   Objects analysis exists")

            if has_navigability:
                needs_nav_analysis = False
                print("   Navigability analysis exists")

            if not needs_object_analysis and not needs_nav_analysis:
                print("   Analysis complete, skip")
                skip_count += 1
                continue

        if needs_object_analysis:
            print("   Running GPT-4o for objects and room...")
            result = analyze_position_with_images(position_dir, target_image_path)

            if result:
                with open(analysis_file, 'w', encoding='utf-8') as f:
                    f.write(result)
                print("   Object analysis done")
            else:
                print("   Error: object analysis failed")
                fail_count += 1
                continue

        if needs_nav_analysis:
            print("   Analyzing navigability...")
            nav_result = analyze_navigability(position_dir)

            if nav_result:
                mode = 'a' if analysis_file.exists() else 'w'
                with open(analysis_file, mode, encoding='utf-8') as f:
                    if mode == 'a':
                        f.write("\n\n")
                    f.write(nav_result)
                print("   Navigability analysis done")
            else:
                print("   Warning: navigability analysis failed")

        if needs_object_analysis or needs_nav_analysis:
            success_count += 1

    print()
    print("=" * 80)
    print("  Done")
    print("=" * 80)
    print(f"Success: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Skipped: {skip_count}")
    print(f"Total: {len(position_dirs)}")
    print()
    print(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
