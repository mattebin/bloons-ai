# bloons-ai

Tools for automating and reading HUD data from Bloons TD 6 via OBS Virtual Camera.

## Requirements
- Python 3.10+
- OBS Virtual Camera running (default index 0)
- `xdotool` installed (window focus + click automation)
- `opencv-python`, `numpy`, `pytesseract`

## Quick start (shadow player)
```bash
python btd6_shadow_player_AUTOMATION_FULL_v3_7.py
```

Common options:
```bash
python btd6_shadow_player_AUTOMATION_FULL_v3_7.py --profile default --camera-index 0
```

## Calibration flow
1. Press `c` to calibrate grid-to-OBS mapping.
2. Press `F6` four times to map OBS corners to your desktop.
3. Press `g` to lock the game window (focus BTD6, then press `g` in the SHADOW window).
4. Use `F9` to cycle OFF → ARMED → LIVE.

If mapping drifts or clicks land wrong, press `F7` to reset screen mapping and re-do `F6`.
