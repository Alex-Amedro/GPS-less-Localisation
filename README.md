# GPS-less Localisation

**Where is a drone when the GPS is lost?** This project estimates the position of a fixed-wing drone flying at about 300–500 m from the images of its downward-looking camera, the last known position, and a satellite map stored on board.

It is a *realistic engineering system*, not research: it assembles standard, proven bricks (a pre-trained DINOv2 model, a particle filter, classical SIFT matching) and reports honestly **what it does and where it fails**. No novelty is claimed.

![final](docs/final_hybrid_idk.gif)

*Real test flight (never used for training or tuning). Green: true trajectory. Red: where the drone thinks it is. Cyan: the hypotheses (particles). Yellow: the answer from a single photo. Orange: dead reckoning without photos. White: last GPS fix. A grey circle means "I don't know".*

## The problem

- A fixed-wing drone flies at 300–500 m; the camera points down; heading and approximate altitude are known.
- At some point the **GPS is lost**. The only things left: the last known position and speed, the heading, the photos, and a pre-loaded satellite map.
- The system must (1) find where it is, (2) use context (previous photos, motion), and (3) say **"I don't know"** when it is not sure.

## How it works (in short)

1. **Fingerprints.** The map is cut into overlapping 305 m disks. A DINOv2 (ViT-B/14) model turns each disk into a short list of numbers (its *descriptor*). Two views of the same place should give similar descriptors.
2. **One photo → candidates.** A photo is rotated north-up with the known heading, cropped to the same 305 m disk, turned into a descriptor, and compared (cosine similarity) with all map disks.
3. **Training.** Only the last two blocks of DINOv2 are fine-tuned, with a contrastive loss, on pairs *(recent photo, older map of the same place)* with random shifts, rotations, scale, blur and colour changes. Training data: IGN orthophotos of Belfort (2017 map vs 2020 photos) mixed 50/50 with real drone flights from UAV-VisLoc. Train / validation / test are separated **by geographic zone**.
4. **Particle filter.** 2 000 hypotheses move along the known heading at an unknown speed, get re-weighted by how well the photo matches the map under each of them, and are resampled. Before any photo is trusted, a **physical constraint** rejects answers the drone could not have reached.
5. **SIFT hybrid.** Classical SIFT matching rarely answers, but when it does it is almost never wrong, so it is added as a precise position measurement.
6. **"I don't know".** The system measures how much of the particle cloud agrees on one place and abstains below a threshold set on the validation flight.

## Results

All numbers below are on a **real flight never used for training or tuning** (UAV-VisLoc flight 03, 768 photos, 466 m altitude, China), with the GPS cut at 7 different moments and 150 photos tracked each time (about 15 minutes of flight). The system gets only: last position and speed, heading, photos, map.

| Method | Median error | Within 100 m | Within 500 m |
|---|---|---|---|
| Dead reckoning (heading + speed, no photos) | 1 518 m | 4 % | 23 % |
| Photo by photo, no context | 60 m | 76 % | 87 % |
| Particle filter (photo + heading) | 55 m | 90 % | 100 % |
| **+ SIFT (hybrid)** | **32 m** | **94 %** | **100 %** |
| + "I don't know" (answers 78 % of photos) | **28 m** | **98 %** | 100 % |

*Filter rows: average over 6–10 random seeds. With the final system no answer was further than 500 m. Before the physical constraint, 9 random runs out of 20 jumped by more than 1 km at the first U-turn (see below). The last row uses a threshold set on the validation flight; it uses a slightly different set of GPS-cut moments (every 40 photos).*

Error of the photo-by-photo model alone, same flight: **75 % within 100 m**, versus 36 % for the same DINOv2 model without any training and 37 % of photos answered by SIFT alone.

### What the training buys (photo by photo, no filter, % within 100 m)

| Test | Pre-trained DINOv2-B | Fine-tuned (last 2 blocks) |
|---|---|---|
| Real flight 03 (China, vs Google map) | 36 % | **75 %** |
| Belfort countryside, 2017 map vs 2020 photos | 20 % | **70 %** |
| Val-d'Oise (other French region) | 21 % | **64 %** |

### Robustness (final system, same real flight, one perturbation at a time)

| Perturbation | Within 100 m | Comment |
|---|---|---|
| None | 90 % (94 % with SIFT) | reference |
| Heading bias 5° / 10° / **20°** | 83 % / 75 % / **45 %** | **main weakness**: the model needs the heading (30°: 23 %) |
| Heading noise up to 20° (no bias) | 86–90 % | insensitive |
| Wrong altitude (view × 1.25 / × 1.5) | 87 % / 78 % | mild |
| Camera tilt 10° / **20°** | 77 % / **51 %** | **second weakness** (never seen in training; 30°: 17 %) |
| Initial speed × 0.5 … × 1.5 | 87–90 % | insensitive |
| Last GPS fix wrong by 100 / 300 m | 90 % / 88 % | insensitive |
| Last GPS fix wrong by 1 000 m | 80 % (87 % with SIFT) | recovers, but 6 % of answers exceed 1 km |
| Photos missing: 6 of every 20 | 68 % (76 % with SIFT) | degrades gracefully, no drift |
| Photos missing: 20 of every 60 (about 2.5 min) | 60 %, 94 % within 500 m | the filter flies blind then recovers |

## Failure cases, shown

A single photo is wrong in some stretches: rice fields that are **pale yellow in the photo** (ripe rice, Oct. 2018) but **dark green on the satellite map** (flooded fields), so the model picks another village up to 5.7 km away. The filter survives most of them because it trusts motion more than one ambiguous photo.

The first version of the filter sometimes believed that wrong answer at the first U-turn and jumped by kilometres (left). A drone cannot fly 5.7 km in 46 s, so answers farther than `max speed × elapsed time` are now rejected (right):

| Before (random jump at the U-turn) | After (physical constraint) |
|---|---|
| ![before](docs/before_random_jump_at_turn.gif) | ![after](docs/after_physical_constraint.gif) |

## Honest limitations

- **One real test flight** (plus one real validation flight). Results on other places, seasons or cameras are not demonstrated; only the model trained on France + 5 flights is tested on China.
- **The test flight was looked at several times** while the filter was being corrected (the model of turns, the ambiguity idea that was rejected, the physical constraint). Every change was validated on a separate flight first, and everything is written down in `experiments.md`, but the test is no longer a pristine one-shot.
- **The heading is assumed known and reliable** (here it is almost perfect). A heading bias of 20° or more breaks the system, and so does a strongly tilted camera.
- **The photo scale is assumed known** (from altitude and optics). Here it was calibrated on the same flight.
- **Fine precision is limited** by the 128 m grid of the map disks (about 90 m worst case); the median error below that comes from SIFT and the filter.
- **The satellite maps of UAV-VisLoc come from Google** and are not redistributed here.
- **Not tested on embedded hardware.** On the development laptop (RTX 5070, 8 GB): about 11 ms per photo for the descriptor and 0.06 s for SIFT.
- **Initial position and speed come from the last GPS fixes**; starting with no position at all is not covered.

## Data (not in this repository)

- **IGN BD ORTHO** (map and training in France): <https://data.geopf.fr/telechargement/resource/BDORTHO> — Licence Ouverte Etalab (credit: "BD ORTHO®, IGN"). One archive per department and edition (used: Belfort D090, editions 2017 and 2020; Val-d'Oise D095, 2018 and 2021).
- **UAV-VisLoc** (real flights and test): <https://github.com/IntelliSensing/UAV-VisLoc> — paper <https://arxiv.org/abs/2405.11936>. Full dataset (16.4 GB, 11 flights): <https://drive.google.com/file/d/1xYODANyilEMM3CfWh85APwkTHQeLTcCT/view?usp=sharing>. Google Drive limits downloads per file; retry later if you get "Quota exceeded".
- **VPAIR** (secondary, not evaluated): <https://zenodo.org/records/6473989>.

## Repository layout

| Path | What |
|---|---|
| `Project.md` | the original brief (in French) |
| `experiments.md` | **the full lab journal** (in French): what was tried, what worked or failed, with measured numbers |
| `scripts/prep_ign_test.py` | builds "recent photo vs older map" tests from IGN editions |
| `scripts/prep_uav_test.py`, `calib_uav_gsd.py` | UAV-VisLoc flights as tests; photo scale calibration |
| `scripts/prep_train_data.py`, `train_dino.py` | training data (by zone) and fine-tuning of the last blocks |
| `scripts/sift_eval.py`, `dino_eval.py` | SIFT baseline and DINOv2 evaluation (Recall@K, error in metres) |
| `scripts/particle_filter.py` | the context filter, the SIFT hybrid, the animations |
| `scripts/confidence.py` | "I don't know": confidence, coverage / error curve |
| `scripts/robustness.py` | heading, altitude, tilt, speed and GPS perturbations |
| `docs/*.gif` | animations used above (the full videos are kept under `data/vis/`, not versioned) |

Python 3.12, PyTorch with CUDA 12.8 (RTX 5070 laptop GPU); dependencies in `requirements.txt`. Most scripts have `--help`.
