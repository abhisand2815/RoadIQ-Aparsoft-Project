# RoadIQ — No-Helmet Detection: Status & Test Log

Status snapshot of the no-helmet detection pipeline (Detection Task: `No-Helmet Detection`, Inference Mode: `Image Inference`), current as of this test batch.

**Config used for all tests below:** Detector `YOLO11 Nano`, Pose `YOLO11 Pose Nano`, Helmet model `YOLO26 Large PPE/Helmet (High Acc.)`, Helmet confidence threshold `0.35`.

---

## Summary

| # | People visible | Persons evaluated | Status shown | Verdict |
|---|---|---|---|---|
| 1 | 1 rider (small, distant) + 1 on second bike | 1 | NO-HELMET VIOLATION (1/1), 41% | Plausible, needs manual check; second bike marked PARKED / NO RIDER |
| 2 | 3 riders + possible pillion | 3 | NO-HELMET VIOLATION ×2, HELMETS OK ×1 | Violations correct; Helmets Compliant stays 0; possible missed pillion |
| 3 | 2 (same bike, both helmeted) | 1 | HELMETS OK, HEAD OCCLUDED | Pillion missed, counted as pedestrian; Helmets Compliant stays 0 |
| 4 | 3 (same bike, masks, no helmets) | 3 | NO-HELMET VIOLATION (3/3) | Correct |
| 5 | 3 (same bike, no helmets) | 3 | NO-HELMET VIOLATION (3/3), 73% | Correct, label text garbled |
| 6 | 3 (same bike, no helmets) | 2 | NO-HELMET VIOLATION (2/2), 43% | Undercount: one occupant not evaluated |
| 7 | 2 (same bike, both helmeted) | 2 | HELMETS OK, HEAD OCCLUDED ×2 | Correct (no violation); Helmets Compliant stays 0 |
| 8 | 1 (helmeted) | 1 | HELMETS OK, HEAD OCCLUDED | Correct (no violation); Helmets Compliant stays 0 |
| 9 | 2 (same bike, both helmeted) | 1 | HELMETS OK, HEAD OCCLUDED | Same case as #3, repeat capture |
| 10 | Dense crowd, 5 bikes | 4 riders, 3 pedestrians | NO-HELMET VIOLATION ×4 | Needs manual check; pedestrian count suspect, labels unreadable |

No false violations were raised on any of the four frames with helmeted riders (3, 7, 8, 9). Violations were raised on every frame with clearly bareheaded occupants (2, 4, 5, 6, 10); counts were exact on #4 and #5, undercounted on #6, and unverified on #1, #2, and #10. The `Helmets Compliant` counter was `0` on all 10 frames, including every frame where the bike was labeled `HELMETS OK`.

---

## Recurring issues

1. **`Helmets Compliant` counter never increments.** Frames 2, 3, 7, 8, 9 show `BIKE n | RIDERS n | HELMETS OK` on the image, yet the summary counter reads `Helmets Compliant 0`. The per-bike status and the aggregate counter are computed from different paths, or the counter only increments on an explicit helmet class that is never reached.
2. **`HEAD OCCLUDED` used for riders clearly wearing full-face helmets** (frames 3, 7, 8, 9, and the middle rider in 2). The helmet is visible but the rider is tagged as occluded instead of compliant. This is likely why issue 1 occurs: the status falls through to `HEAD OCCLUDED` rather than a counted `HELMET` result, possibly because the helmet model confidence on full-face/visor helmets is below the `0.35` threshold or the head crop is misaligned.
3. **Pillion not associated with the bike and counted as a pedestrian** (frames 3 and 9). Two helmeted people are on one motorcycle, only one is counted as a rider, and the other is shown as `PERSON | NOT ASSOCIATED` with `Pedestrians (Ignored) 1`. Same association weakness as the triple-riding log (test 7): heavy overlap and motion blur drop the rear occupant.
4. **Occupant undercount on multi-person bikes** (frame 6: 3 people, `2/2 without helmet`; frame 2: a second person behind the right-hand rider appears to be unevaluated, `Sitting Behind 0`). Violation counts are lower than the true number of bareheaded occupants.
5. **Possible missed rider labeled `PARKED / NO RIDER`** (frame 1, `BIKE 8`). A person appears to be seated on that motorcycle in the original image.
6. **Hat/cap fallback fires at low confidence.** Frames 1, 6, 10 use `NO HELMET - Hat / Cap detected (ears & face exposed below brim)` at 41–43%, close to the `0.35` threshold. Correct on frame 6 (blue cap, no helmet); unverified on frames 1 and 10.
7. **Overlapping, clipped, and garbled on-image labels** (frames 1, 2, 4, 5, 10). Text such as `BIK E 1`, `NO-H ELMET VIOLAT ON`, `VIOLAT ION (3, 3 withou t helm et)` is corrupted, and labels near the right edge of the image are cut off. Cosmetic, but makes manual QA harder.

---

## Per-image detail

### Test 1
![Test 1](test01.png)

Hazy street scene with a small, distant rider in the center. Detected as `BIKE 5 | NO-HELMET VIOLATION (1/1 without helmet)` with `RIDER (Driver) #0 | NO HELMET - Hat / Cap detected (ears & face exposed below brim) (41%)`. Counters: `Motorcycles 2, Riders 1, Pillions 0, No Helmet Violations 1, Helmets Compliant 0, Pedestrians 1`. The rider is small and low resolution, so the result is plausible but needs a manual check. A second motorcycle at the right (`BIKE 8`) is labeled `PARKED / NO RIDER` although a person appears to be on it, a possible missed rider. The `BIKE 5` label partly overlaps another box and is clipped.

### Test 2
![Test 2](test02.png)

Three motorcycles with one rider each: left rider (yellow shirt) and right rider (dark shirt, white scooter) are bareheaded, middle rider wears a grey helmet. Detected: `RIDER (Driver) #2 | NO HELMET VIOLATION (81%)`, `RIDER (Driver) #1 | NO HELMET VIOLATION`, `RIDER (Driver) #0 | HEAD OCCLUDED`, `BIKE 5 | RIDERS: 1 | HELMETS OK`, `BIKE 1 | NO-HELMET VIOLATION (1/1 without helmet)`. Counters: `Motorcycles 3, Riders 3, Pillions 0, No Helmet Violations 2, Helmets Compliant 0, Pedestrians 0`. Both violations are correct, but the helmeted middle rider is not counted as compliant, and a second person seated behind the right-hand rider appears to be unevaluated. Labels for `BIKE 5` and `RIDER #0` overlap and are partly unreadable (`HE.METS OK`, `O CLUD ED`), and the right-most label is clipped at the frame edge.

### Test 3
![Test 3](test03.png)

Motion-blurred shot of two people on one motorcycle, both wearing dark helmets. Detected as `BIKE 1 | RIDERS: 1 | HELMETS OK` with `RIDER (Driver) #0 | HEAD OCCLUDED`. The second occupant is drawn as a faint `PERSON | NOT ASSOCIATED` label and counted under `Pedestrians (Ignored) 1`. Counters: `Motorcycles 1, Riders 1, Pillions 0, No Helmet Violations 0, Helmets Compliant 0, Pedestrians 1`. No false violation, but the pillion is missed and the helmeted rider is not counted as compliant.

### Test 4
![Test 4](test04.png)

Three men on one motorcycle, all wearing face masks and no helmets. Detected as `BIKE 1 | NO-HELMET VIOLATION (3/3 without helmet)` with `RIDER (Driver) #0` and `PERSON (Sitting Behind) #2 | NO HELMET VIOLATION`. Counters: `Motorcycles 1, Riders 1, Pillions 2, No Helmet Violations 3, Helmets Compliant 0, Pedestrians 0`. Correct. Rider/pillion labels stack on top of each other and are clipped at the right edge; `NO-HELMET VIOLATION` in the bike label renders with broken characters.

### Test 5
![Test 5](test05.png)

Three men on one motorcycle on a rural road, no helmets, slight motion blur. Detected as `BIKE 0 | NO-HELMET VIOLATION (3/3 without helmet)` with `RIDER (Driver) #2 | NO HELMET VIOLATION (73%)` and `PERSON (Sitting Behind) #1 | NO HELM...`. Counters: `Motorcycles 1, Riders 1, Pillions 2, No Helmet Violations 3, Helmets Compliant 0, Pedestrians 0`. Correct. The bike label text is garbled (`VIOLAT ION (3, 3 withou t helm et)`) and the pillion label is cut off at the frame edge.

### Test 6
![Test 6](test06.png)

Three people on one motorcycle (man in a blue cap in front, child in the middle, woman behind), none wearing helmets. Detected as `BIKE 0 | NO-HELMET VIOLATION (2/2 without helmet)` with `RIDER (Driver) #1 | NO HELMET - Hat / Cap detected (ears & face exposed below brim) (43%)`. Counters: `Motorcycles 1, Riders 1, Pillions 1, No Helmet Violations 2, Helmets Compliant 0, Pedestrians 0`. The driver's cap is correctly rejected as a helmet, but only 2 of the 3 occupants are evaluated, so the violation count is one short.

### Test 7
![Test 7](test07.png)

Two people on a sport touring motorcycle, both in full-face helmets and leather jackets. Detected as `BIKE 0 | RIDERS: 2 | HELMETS OK` with `RIDER (Driver) #0 | HEAD OCCLUDED` and `PERSON (Sitting Behind) #1 | HEAD OCCLUDED`. Counters: `Motorcycles 1, Riders 1, Pillions 1, No Helmet Violations 0, Helmets Compliant 0, Pedestrians 0`. Rider and pillion association is correct and no violation is raised, but both helmeted heads are tagged `HEAD OCCLUDED` and `Helmets Compliant` stays at 0.

### Test 8
![Test 8](test08.png)

Single rider on a naked roadster, full-face helmet, red jacket. Detected as `BIKE 0 | RIDERS: 1 | HELMETS OK` with `RIDER (Driver) #0 | HEAD OCCLUDED`. Counters: `Motorcycles 1, Riders 1, Pillions 0, No Helmet Violations 0, Helmets Compliant 0, Pedestrians 0`. Correct outcome (no violation), but the helmeted rider is not counted as compliant.

### Test 9
![Test 9](test09.png)

Same scene as Test 3, second capture. Same result: `BIKE 1 | RIDERS: 1 | HELMETS OK`, `RIDER (Driver) #0 | HEAD OCCLUDED`, second occupant as `PERSON | NOT ASSOCIATED`, `Pedestrians 1`, `Helmets Compliant 0`. Confirms the pillion miss and the compliant-counter bug are reproducible, not one-offs.

### Test 10
![Test 10](test10.png)

Dense traffic scene with five motorcycles and many masked, mostly bareheaded riders. Detected: `BIKE 1`, `BIKE 5`, `BIKE 8`, `BIKE 2` all with `NO-HELMET VIOLATION`, plus `RIDER (Driver) #2 | NO HELMET VIOLATION (84%)` and `RIDER (Driver) #4 | NO HELMET - Hat / Cap...`. Counters: `Motorcycles 5, Riders 4, Pillions 0, No Helmet Violations 4, Helmets Compliant 0, Pedestrians 3`. The violation count is plausible, but `Pedestrians 3` is suspect in a scene where nearly everyone visible is seated on a bike, so some riders or pillions are likely unassociated. Labels overlap heavily and are largely unreadable or clipped at the right edge, and one blue box at the right edge is cut off. Manual verification of the per-bike counts is needed.

---

## Open items

- Fix the `Helmets Compliant` counter so it increments for bikes labeled `HELMETS OK` (tests 2, 3, 7, 8, 9).
- Investigate why full-face helmets are classified as `HEAD OCCLUDED` instead of a helmet detection (tests 3, 7, 8, 9); check helmet model confidence against the `0.35` threshold and head-crop alignment.
- Re-check rider/pillion association under motion blur and heavy overlap (tests 3, 9: pillion dropped and counted as pedestrian; test 10: `Pedestrians 3`).
- Investigate occupant undercount on multi-person bikes (test 6: 2 of 3 evaluated; test 2: possible unevaluated pillion).
- Manually verify the `PARKED / NO RIDER` classification of `BIKE 8` in test 1.
- Review the hat/cap fallback behavior at low confidence (tests 1, 6, 10).
- Fix garbled and clipped label rendering and add de-overlapping for on-image text (tests 1, 2, 4, 5, 10).
