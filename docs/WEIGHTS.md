# Model weights

The project already supports generic YOLO detection/pose weights and custom helmet checkpoints configured in `config.py`.

Download official Ultralytics pretrained checkpoints into the project `weights/` directory:

```bash
python tools/download_weights.py --preset compact
python tools/download_weights.py --preset all
```

`compact` is a practical starter set. `all` can consume several gigabytes and may take a long time. Use only model variants supported by your installed `ultralytics` version. Official model family/catalog: https://github.com/ultralytics/assets/releases and https://docs.ultralytics.com/models/.

**Important:** generic COCO YOLO weights do not inherently detect `no helmet`, `wrong way`, or `stop line` as custom violation classes. These workflows need their configured custom weights and/or their existing geometric/temporal logic. The custom filenames `ppe_yolo26l_96e.pt`, `helmet_yolo11s.pt`, and `helmet_yolov8n.pt` are not replaced by generic weights.
