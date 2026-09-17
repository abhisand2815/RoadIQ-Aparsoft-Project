
# RoadIQ — Intern Project Brief

A roadside camera watches a junction. When someone rides without a helmet, carries
three people on a two-wheeler, drives the wrong way, or jumps the stop line, we
catch it and produce evidence a traffic officer can act on.

| | |
|---|---|
| **Interns** | Both teams |
| **Duration** | 7 weeks |
| **Setup** | Same brief, same contract, same test footage for both teams |
| **Stack** | Python · YOLO · FastAPI · OpenCV |
| **Ships as** | Apar Drishti prototype — plugs into the existing ProducerEngine alert bus |

---

## 1. What you are building

A FastAPI service. You give it a camera feed and a zone configuration. It watches
the feed and pushes out violations as they happen.

Four violations:

| Violation | What it means |
|---|---|
| `no_helmet` | A rider or pillion on a two-wheeler with no helmet |
| `triple_riding` | Three or more people on one two-wheeler |
| `wrong_way` | Vehicle travelling against the allowed direction for that zone |
| `stop_line` | Vehicle crosses the stop line while the signal is red |

Every violation produces an **evidence packet**: a short clip, key frames, a crop
of the number plate, the time, the zone, and how many frames confirmed it.

> **The one thing that matters**
>
> False positives kill this product. A system that flags 200 innocent riders a day
> gets the pilot cancelled in week one. Missing a violation is a shrug. Flagging a
> man who was wearing a helmet is a complaint. Build for the second problem.

### What we already have — do not rebuild it

- **ANPR exists.** Do not build number plate reading. Crop the plate region, save
  it, put the path in the event. Someone else reads it.
- **The alert bus exists.** Console, LAN webhook, RS-232, GPIO. Your job is to emit
  a clean event. Delivery is not your problem.

---

## 2. Teams and roles

Both teams build the same thing separately. At week 7 we compare them on the same
footage and the same rubric. Because the API contract is frozen, we can also take
the best part from each team.

Inside a team, split three ways.

### Role 1 — Detect & Track
Train the helmet model. Work out which person is sitting on which two-wheeler.
Track each vehicle across frames so it keeps one ID as it crosses the scene.

**Owns:** frames → tracked riders

### Role 2 — Violation Logic
Zones, direction, stop line, signal state. Decide when something is actually a
violation and not a bad frame. Confirmation across frames lives here.

**Owns:** tracked riders → confirmed violations

### Role 3 — Service & Evidence
FastAPI, multiple streams at once, clip extraction, evidence packets, daily report.

**Owns:** violations → delivered

> **Role 1, read this**
>
> Rider-to-vehicle association is the hardest part of the whole project and nobody
> talks about it. YOLO gives you person boxes and motorcycle boxes. It does not
> tell you that person #4 is riding motorcycle #2 and person #7 is just standing
> on the footpath behind it. Get this wrong and every count is wrong. Start it in
> week 1, not week 3.

---

## 3. API contract — frozen

Do not change these. Both teams implement exactly this so we can swap and compare.
If something is genuinely missing, ask — do not invent.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/streams` | Register a feed with its zone config. Returns `stream_id`. |
| `GET` | `/streams` | List streams with live status and FPS. |
| `DELETE` | `/streams/{id}` | Stop a stream. |
| `GET` | `/streams/{id}/events` | SSE. Violations pushed live. |
| `POST` | `/streams/{id}/signal` | Push the current signal phase. |
| `GET` | `/events` | Query past violations. Filter by type, zone, time. |
| `GET` | `/events/{id}` | One violation in full. |
| `GET` | `/events/{id}/clip.mp4` | The evidence clip. |
| `GET` | `/events/{id}/plate.jpg` | Plate crop, for the ANPR handoff. |
| `GET` | `/reports/daily` | Counts by violation type and hour. |
| `GET` | `/healthz` | Liveness. Returns 200. |

### POST /streams

```json
{
  "source": "rtsp://192.168.1.40/stream1",
  "name": "Hosur Road junction, north camera",
  "watch": ["no_helmet", "triple_riding", "wrong_way", "stop_line"],
  "zones": [
    {
      "name": "northbound",
      "polygon": [[120, 400], [900, 400], [980, 720], [60, 720]],
      "allowed_direction": [0, -1]
    }
  ],
  "stop_line": [[120, 470], [900, 470]]
}
```

- `allowed_direction` — unit vector in image space. A track moving against it is
  wrong-way.
- `stop_line` — two points. Omit it and `stop_line` violations are skipped.

Returns:

```json
{ "stream_id": "str_03", "status": "starting" }
```

### POST /streams/{id}/signal

```json
{ "phase": "red", "ts": "2026-10-14T08:41:10+05:30" }
```

- `phase` — one of `red` | `amber` | `green`.
- If nobody pushes signal state, `stop_line` violations are not reported. Do not
  guess the phase.

### A violation event

```json
{
  "event_id": "ev_7c21a9",
  "stream_id": "str_03",
  "type": "no_helmet",
  "ts": "2026-10-14T08:41:22+05:30",
  "zone": "northbound",
  "track_id": 418,
  "confidence": 0.91,
  "frames_confirmed": 14,
  "riders": 2,
  "helmets": 1,
  "vehicle_box": [412, 388, 96, 140],
  "plate_box": [438, 496, 52, 18],
  "clip_url": "/events/ev_7c21a9/clip.mp4",
  "plate_url": "/events/ev_7c21a9/plate.jpg"
}
```

- `frames_confirmed` — how many frames agreed before we fired. Never fire on one.
- `riders` / `helmets` — only for two-wheeler violations. `null` otherwise.
- `plate_box` may be `null` if the plate was not visible. Still report the event.

### SSE stream

```
event: violation
data: { "event_id": "ev_7c21a9", "type": "no_helmet", ... }

event: heartbeat
data: { "stream_id": "str_03", "fps": 22.4, "active_tracks": 12 }
```

Send a heartbeat every 5 seconds so the client knows the stream is alive when
nothing is happening.

---

## 4. Test footage

Both teams use the same footage. Build it in weeks 1–2, together, before splitting.

| Set | What | How much | What it measures |
|---|---|---|---|
| **A** | Real junction footage you shoot yourselves — morning, afternoon, and evening low light | 90 min | **Does it work.** You hand-log every violation you can see. That log is the ground truth. |
| **B** | Staged clips in a private compound — helmet on, helmet off, two up, three up, riding against the arrow | 30 clips | **Does it work on known answers.** Every clip has one correct answer, decided before you run anything. |

Start from public data so you are not blocked in week 1. The India Driving Dataset
(IIIT Hyderabad) and public helmet datasets on Roboflow give you a baseline model
before your own footage is labelled.

> **Safety and law — not optional**
>
> Do not stage violations on a public road. No riding without a helmet, no three
> up, no wrong way on any public street. Use a private compound or a parking lot
> with permission. Set B is staged; Set A is observed only.
>
> Blur faces everywhere except inside an evidence packet. Delete all footage at
> the end of the project. We are recording public roads — treat that seriously.

---

## 5. Week by week

Each gate is a demo on Friday. Not passing a gate is fine — hiding it is not.

| Week | Deliverable | Gate |
|---|---|---|
| **1** | Shoot Set A and Set B. Run stock COCO YOLO on it. Contract handed down. | Show what stock YOLO gets and what it misses. Numbers, not opinions. |
| **2** | Label helmets. Train the model. | Helmet vs no-helmet accuracy on a held-out set. |
| **3** | Rider-to-vehicle association. Tracking with stable IDs. | On a 2-minute clip, correct rider count for 8 out of 10 two-wheelers. |
| **4** | `no_helmet` and `triple_riding`. Confirmation across frames. | Run 20 minutes of Set A. Report both catches **and** false positives. |
| **5** | Zones. `wrong_way` and `stop_line`. | Draw a zone, ride against it, get one event and only one. |
| **6** | FastAPI service. Multiple streams. Evidence packets. | 4 streams at once. Open an evidence clip and see the violation in it. |
| **7** | Full run on Sets A and B. Bake-off. | Both teams scored on the rubric below. |

> **Expect to stall in week 3**
>
> Rider-to-vehicle association is where this project is won or lost. Two riders on
> adjacent bikes, a pillion half hidden, someone walking behind — all of it looks
> the same to a box detector. If your team is still stuck at the end of week 4,
> tell us. We will cut you to `no_helmet` only and let the other team carry the
> full set. Say it early; there is no slack in seven weeks.

---

## 6. How you are scored

Out of 100, on Sets A and B, run by us on the same machine.

| What | How we measure it | Points |
|---|---|---:|
| **Violations caught** | Against your hand-written log for Set A, and the known answers in Set B | 25 |
| **False positives** | Wrong violations per hour of footage. This is scored hardest. | 25 |
| **Rider association** | Correct rider count per two-wheeler | 15 |
| **Evidence quality** | Can a traffic officer look at one packet and act on it — clip, plate, time, type | 15 |
| **Performance** | Streams per box, FPS per stream, how it behaves when a feed drops | 15 |
| **Code and tests** | Can someone else run it and change it | 5 |
| **Total** | | **100** |

Note the shape. Catching violations and not crying wolf are worth the same. A
system that catches everything and flags 40 innocent riders an hour scores worse
than one that catches 70% and flags nobody.

---

## 7. Rules

**Do**

- Start rider-to-vehicle association in week 1. It is the hard part.
- Confirm across frames before firing. Never fire on a single frame.
- Commit every day. Demo every Friday, working or not.
- Report your false positives at every demo, out loud, before we ask.

**Do not**

- Do not build ANPR. Crop the plate and hand it off.
- Do not build alert delivery. Emit the event; the alert bus already exists.
- Do not guess the signal phase. No signal input means no `stop_line` events.
- Do not stage violations on a public road.
- Do not change the API contract. Ask instead.
- Do not pick your own footage. Both teams run the same set.

---

## 8. Questions we should be able to answer at week 7

1. Out of 100 real violations, how many did we catch?
2. How many false violations did we report per hour of footage?
3. How many camera streams run on one box, and at what FPS?
4. Can a traffic officer open one evidence packet and issue a challan from it?

If you can answer these four with real numbers, the project worked.

---

*Apar Drishti · RoadIQ brief · Teams A and B · 7 weeks*
=======
# RoadIQ — Triple Riding Detection (Pretrained Multi-Model)

This version uses **no custom-trained model**.

## Models

- YOLO26 / YOLO11 object detector: person + motorcycle detection and ByteTrack IDs.
- YOLO26 / YOLO11 pose model: human keypoints used to strengthen rider-to-bike association.
- Different pretrained detector and pose variants can be selected from the Streamlit sidebar.

## Pipeline

Input → YOLO detector → motorcycles/persons → YOLO pose → keypoints → person-bike association → rider count → temporal confirmation → triple-riding event.

## Association

The association layer is not a learned custom classifier. It combines pretrained YOLO outputs with transparent geometry and pose cues. This is intentionally the non-custom-model version.

## Run

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Put stored videos in `videos/` if desired. Model `.pt` files can be placed in `weights/`; otherwise Ultralytics can resolve/download the selected pretrained model.
>>>>>>> a9b0f9a (Triple Riding)
