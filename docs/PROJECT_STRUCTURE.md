# RoadIQ project structure

The existing Python modules remain at the repository root to preserve their imports and current behavior.

## Feature ownership
- `video_service.py`: triple-riding video inference loop and annotated MP4 output.
- `triple_riding.py`, `triple_riding_service.py`: triple-riding logic and service.
- `helmet_detector.py`, `helmet_service.py`: rider/helmet analysis and no-helmet workflow.
- `combined_service.py`: combined triple-riding + no-helmet workflow.
- `wrong_way.py`, `wrong_way_service.py`, `get_fetv_wrong_way.py`: wrong-way detection workflow.
- `stop_line_detector.py`, `stop_line_service.py`, `traffic_signal_detector.py`: stop-line / signal workflow.
- `association.py`, `rider_association.py`: shared association logic.
- `model_loader.py`: model loading.
- `config.py`: paths, model catalogs, thresholds and shared settings.
- `image_service.py`: image-mode routing/processing.
- `app.py`: Streamlit UI and task routing.
- `weights/`: pretrained/custom `.pt` files.
- `datasets/<feature>/`: place feature-specific training/test data here; existing image paths are not moved.
- `outputs/`: processed videos.
- `evidence/`: violation evidence artifacts.
- `Status/`: existing project status notes and test evidence.

## Important upload behavior
Streamlit's standard `st.file_uploader` receives the selected file before Python can open and process it. Once upload finishes and inference starts, the video loop displays annotated frames and progress incrementally. True inference while bytes are still uploading requires a chunked-upload frontend/API and is a larger architectural change; it is not enabled by this patch.
