# VLM Navigation Pipeline

## Overview
This project uses:
- Grounding DINO (object detection)
- SAM (segmentation)
- MiDaS (monocular depth estimation)
- Multi-pass prompting
- Scene understanding & accessibility classification

## Features
- **Depth Estimation**: Replaces heuristic distance with real depth-map estimates.
- **Accessibility Classification**: Identifies features (curb cuts, ramps) and hazards (potholes, stairs).
- **Conversational Guidance**: Generates WalkGPT-style natural language navigation text.
- **Spatial Map**: Groups objects by left/center/right corridors.

## Usage

### Standard Mode
```bash
python main.py --image images/test.jpg
```

### WalkGPT Mode (Depth + Rich Guidance)
```bash
python main.py --image images/test.jpg --depth --walkgpt
```

Options:
- `--depth`: Enables MiDaS depth estimation.
- `--depth-model`: Choose model (`MiDaS_small`, `DPT_Hybrid`, `DPT_Large`). Default: `MiDaS_small`.
- `--walkgpt`: Uses the rich conversational output format.
- `--no-sam`: Disables SAM segmentation (faster, but less accurate surface masks).

## Example Output (WalkGPT Mode)
```
  ACTION: move_forward_cautious
  RISK  : MEDIUM

  [GUIDANCE]
  Moderate risk ahead — stay alert. The path ahead is partially obstructed. Move forward carefully and watch your step. Detected: pothole(2m, center). Surface: smooth. Accessibility concern: pothole. Path clearance is approximately 4 metres ahead.

  [SPATIAL MAP]
  Left  : walkable   | clear
  Center: uncertain  | pothole(2m)
  Right : clear      | clear

  [ACCESSIBILITY]
  Surface: smooth
  Hazards: pothole (near)
  Width: adequate (>1.2m)
```

## Future Work
- LLM integration for dynamic Q&A
- Walkable path extraction via advanced polygon generation