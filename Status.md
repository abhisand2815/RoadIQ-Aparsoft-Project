# RoadIQ
### Aparsoft Road Intelligence Prototype

> **RoadIQ** is a computer-vision road-safety prototype focused on
> detecting **triple riding on two-wheelers** from images and videos,
> generating annotated results and evidence packets that can later be
> connected to the broader Apar Drishti traffic-violation platform.

🔗 **Repository:** `https://github.com/abhisand2815/RoadIQ-Aparsoft-Project`


---


## 1. Project Overview

RoadIQ is being developed as a roadside-camera intelligence system.

The original project brief defines a broader target system with four
traffic violations:

  -----------------------------------------------------------------------
  Violation               Definition              Current RoadIQ focus
  ----------------------- ----------------------- -----------------------
  `no_helmet`             Rider/pillion without   Planned
                          helmet                  

  `triple_riding`         3+ people on one        **Current focus**
                          two-wheeler             

  `wrong_way`             Vehicle moving against  Planned
                          permitted direction     
                          
  `stop_line`             Vehicle crosses stop    Planned
                          line while signal is    
                          red                     
                          
  -----------------------------------------------------------------------

---


# 2. Current Technical Scope

### Input modes

RoadIQ supports:

-   Image inference
-   Video inference
-   Uploaded images/videos
-   Stored videos from the `videos/` directory
-   Stored images from the `iamges/` directory

### Core pipeline

``` text
Image / Video
     ↓
YOLO Object Detection
     ↓
Person + Motorcycle Detection
     ↓
YOLO Pose
     ↓
Human Keypoints
     ↓
Person ↔ Motorcycle Association
     ↓
Rider Count per Motorcycle
     ↓
3+ Riders?
     ↓
Temporal Confirmation (Video)
     ↓
Triple-Riding Event
     ↓
Annotated Output + Evidence
```

For video, the system uses tracking so a motorcycle can retain an
identity across frames instead of being treated as a new object every
frame.


---


# 3. What Each File Does

## `app.py` --- Frontend / Application Entry Point

This is the Streamlit entry point.

It is responsible for:

-   Launching the RoadIQ UI
-   Selecting Image or Video mode
-   Selecting detector and pose models
-   Showing model benchmark information
-   Configuring confidence and IoU thresholds
-   Configuring rider-association thresholds
-   Configuring temporal confirmation
-   Calling image/video services
-   Displaying runtime metrics and generated results

Run the project with:

``` bash
streamlit run app.py
```

Think of `app.py` as the **control panel**, not the AI engine.


---


## `config.py` --- Central Configuration

This is the project's central configuration layer.

It contains:

-   Project directories
-   Supported model registry
-   Detector models
-   Pose models
-   Default models
-   COCO class IDs
-   Confidence thresholds
-   IoU thresholds
-   Image-size options
-   Rider-association parameters
-   Temporal confirmation parameters
-   Tracking configuration
-   Evidence configuration
-   Published model reference metrics

Keeping these values centralized prevents different services from
silently using different thresholds.


---


## `model_loader.py` --- Model Loading and Caching

This file manages model initialization.

It provides loaders for:

-   Standard YOLO models
-   YOLO detector models
-   YOLO pose models
-   YOLO World / YOLOE compatibility where configured
-   Fresh model instances where isolated tracking state is required

It also handles:

-   Local `weights/` lookup
-   Ultralytics model resolution/download
-   Streamlit model caching
-   CPU/GPU device selection

### Important

The project uses **pretrained checkpoints**, not custom RoadIQ-trained
`.pt` checkpoints.

The rider-association stage is currently a transparent geometry + pose
scoring layer.


---


## `image_service.py` --- Image Inference

This service handles single-image processing.

It:

1.  Receives an image.
2.  Runs the selected detector.
3.  Detects people and motorcycles.
4.  Runs the selected pose model.
5.  Associates people with motorcycles.
6.  Counts riders.
7.  Detects triple-riding candidates.
8.  Annotates the image.
9.  Saves the result under `outputs/`.
10. Creates evidence when a triple-riding candidate is found.

Typical generated result:

``` text
outputs/
└── test_image_triple_riding.jpg
```


---


## `video_service.py` --- Video Inference

This service handles video processing.

It performs:

1.  Video loading.
2.  Frame-by-frame inference.
3.  YOLO object detection.
4.  YOLO pose inference.
5.  Tracking.
6.  Rider-to-motorcycle association.
7.  Temporal confirmation.
8.  Triple-riding confirmation.
9.  Frame annotation.
10. Output-video generation.
11. Evidence generation.

Runtime metrics include:

-   FPS / approximate FPS
-   Inference latency
-   Detection confidence
-   Maximum riders on a motorcycle
-   Confirmed tracks

Typical output:

``` text
outputs/
└── test_video_triple_riding.mp4
```


---


## `triple_riding_service.py` --- Core Triple-Riding Engine

This is the central business/ML service for the current feature.

It combines:

``` text
YOLO Detector
     +
YOLO Pose
     +
Rider Association
     +
Rider Counting
     +
Temporal Confirmation
```

The core decision is based on whether a motorcycle has **three or more
associated people**.

This file should remain focused on the triple-riding pipeline rather
than becoming a general-purpose UI module.


---


## `rider_association.py` --- Person-to-Motorcycle Association


This is one of the most important parts of RoadIQ.

YOLO can tell us:

``` text
Person A
Person B
Person C
Motorcycle 1
Motorcycle 2
```

but detection alone does not tell us which person belongs to which
motorcycle.

This module uses geometric and pose-related cues to estimate:

``` text
Person A → Motorcycle 1
Person B → Motorcycle 1
Person C → Motorcycle 1
```

and therefore:

``` text
Motorcycle 1 → 3 riders
```

This is currently **not a learned custom classifier**.

### Future improvement

A stronger learned rider-association model can be introduced later if
labeled rider-to-vehicle association data becomes available.


---


## `association.py` --- Legacy / Alternate Association Logic

This file exists in the repository separately from
`rider_association.py`.

It should be treated carefully during cleanup.

Before extending it, the team should verify whether it is imported by
the active application path.

Suggested check:

``` bash
grep -R "import association\|from association" .
```

If it is not part of the active pipeline, it should either be documented
as legacy code or removed after team approval.


---


## `triple_riding.py` --- Legacy / Alternate Triple-Riding Implementation

This file is separate from:

``` text
triple_riding_service.py
```

The active architecture should have **one clearly defined triple-riding
pipeline**.

The team should check imports and runtime usage before modifying both
files independently.

Suggested check:

``` bash
grep -R "import triple_riding\|from triple_riding" .
```

If unused, keep it documented as legacy until the team decides whether
to remove it.


---


## `evidence_service.py` --- Evidence Generation

**Owner:** Backend / Evidence

This service creates evidence artifacts after a triple-riding
event/candidate is identified.

Evidence can contain:

-   Key frames
-   Annotated evidence frame
-   Vehicle crop
-   Event metadata in JSON

Example:

``` text
evidence/
└── VIDEO_TRIPLE_20260917_130845/
    └── bike_7/
        ├── frame_001.jpg
        ├── frame_002.jpg
        ├── frame_003.jpg
        ├── frame_004.jpg
        ├── frame_005.jpg
        ├── vehicle_crop.jpg
        └── event.json
```

## `requirements.txt` --- Python Dependencies

Defines the main runtime packages, including:

-   Streamlit
-   Ultralytics
-   OpenCV
-   NumPy
-   Pillow
-   PyTorch
-   TorchVision

Install with:

``` bash
pip install -r requirements.txt
```


---


# 4. Folder Structure

``` text
RoadIQ/
│
├── app.py
├── config.py
├── model_loader.py
│
├── image_service.py
├── video_service.py
├── triple_riding_service.py
├── rider_association.py
├── association.py
├── triple_riding.py
├── evidence_service.py
│
├── requirements.txt
├── README.md
├── .gitignore
│
├── images/
│   └── .gitkeep
│
├── videos/
│   └── .gitkeep
│
├── weights/
│   ├── .gitkeep
│   └── *.pt
│
├── outputs/
│   └── .gitkeep
│
└── evidence/
    └── .gitkeep
```


---


# 5. Generated Files

## Image inference

``` text
outputs/
└── <image>_triple_riding.jpg
```

If a triple-riding candidate is generated:

``` text
evidence/
└── IMAGE_TRIPLE_<timestamp>/
    └── bike_<id>/
        ├── evidence_frame.jpg
        ├── vehicle_crop.jpg
        └── event.json
```

## Video inference

``` text
outputs/
└── <video>_triple_riding.mp4
```

Evidence:

``` text
evidence/
└── VIDEO_TRIPLE_<timestamp>/
    └── bike_<track_id>/
        ├── frame_001.jpg
        ├── frame_002.jpg
        ├── frame_003.jpg
        ├── frame_004.jpg
        ├── frame_005.jpg
        ├── vehicle_crop.jpg
        └── event.json
```


---


# 6. Model Strategy

The current project deliberately uses **pretrained models**.

### Detector families

Configured model families include:

-   YOLO26
-   YOLO11
-   YOLO12
-   YOLOv10
-   YOLOv8

with model sizes such as:

``` text
Nano
Small
Medium
Large
XLarge
```

# 7. Running the Project

## 1. Clone

``` bash
git clone https://github.com/abhisand2815/RoadIQ-Aparsoft-Project.git
cd RoadIQ-Aparsoft-Project
```

## 2. Create environment

``` bash
python -m venv .venv
```

macOS/Linux:

``` bash
source .venv/bin/activate
```

Windows:

``` bash
.venv\Scripts\activate
```

## 3. Install dependencies

``` bash
pip install -r requirements.txt
```

## 4. Download default models

``` bash
python download_models.py
```

## 5. Run

``` bash
streamlit run app.py
```


---

## Project Goal

By the end of the development cycle, RoadIQ should move from:

``` text
YOLO detects objects
```

to:

``` text
Camera
  ↓
Detection
  ↓
Tracking
  ↓
Rider association
  ↓
Violation reasoning
  ↓
Temporal confirmation
  ↓
Evidence packet
  ↓
API/event integration
```

That is the transition from a computer-vision demo to a deployable
road-intelligence prototype.
