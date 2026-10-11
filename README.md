# BayWatch — dealer parking & charging bay operations

**Mode: Partner product (proposed pilot).** Proposed owner: dealership yard operations manager at BYD Royal Motors. No partnership, interviews, deployment in a real dealership, or commercial results are claimed. This is a working final-round prototype built with Codex assistance.

A static web application that turns a fixed-camera image into per-space occupancy, abstentions, and human-verification tasks. Upload any permitted image, outline individual parking bays, run the model, and export an evidence report. The classifier consumes actual pixels; no lookup of filenames, ground-truth labels, or hard-coded answers is used for inference.

## Links

- Prototype: https://baywatch-royal-vision.bayt.chatgpt.site
- Public source: https://github.com/baytoooo/baywatch-vision
- Defence deck: [docs/BayWatch-defence.pptx](docs/BayWatch-defence.pptx)
- Russian defence script: [docs/DEFENCE_RU.txt](docs/DEFENCE_RU.txt)

## Run in 30 seconds

```sh
python3 -m http.server 4173 --directory dist
# open http://localhost:4173
```

Or `npm start` (Python 3 required). No npm install, API key, GPU, backend, or paid inference service is required. All inference is browser-side CPU JavaScript. Serving the static `dist/` directory over HTTPS is the entire production deployment. Camera permission requires HTTPS or localhost.

## Test

```sh
npm test  # Node >=20; seven substantive tests including Python/JS numeric parity
```

Test coverage: feature/inference parity to Python; nearest-neighbour sampling boundaries; dark/white/blank frame abstention; confidence band and minimum crop; three-observation temporal hysteresis and dwell reset; conservative overstay action rules; scenario ROI arithmetic.

## Train and reproduce results

Use Python 3.11+ in a virtual environment and the pinned `requirements.txt` (tested with NumPy 2.3.5 and Pillow 12.3.0). Network access is required for the first data download.

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python ml/train.py --cache /tmp/baywatch-pklot
.venv/bin/python ml/cross_camera.py --cache /tmp/baywatch-pklot
npm test
```

`ml/train.py` fixes the upstream dataset revision, samples frames with seed 42 independently of labels, downloads 117 images, audits camera-day split disjointness, trains a 224→48→1 MLP from scratch, chooses an epoch and a temperature on validation only, and writes model, metrics, demo frames and a parity fixture. `docs/dataset-manifest.json` lists every selected image, its immutable source URL and SHA-256. Re-running replaces generated assets. Tiny numerical differences can occur across NumPy versions/architectures.

The mirror split is by `(camera, capture day)`. All three camera views occur in each split. UFPR04 and UFPR05 show the same physical lot. Crops from one frame and successive captures are correlated; do not treat 1,511 crops as 1,511 independent field trials. The original dataset is not a local dealership benchmark.

### Measured primary experiment

| Split | Frames | Parking-space crops |
|---|---:|---:|
| Train | 72 | 3,931 |
| Validation | 18 | 1,007 |
| Test | 27 | 1,511 |

Test at threshold 0.5 (no quality gate / abstention): accuracy **99.272%**, occupied F1 **99.287%**, precision **100%**, recall **98.584%**. Confusion: TP 766, TN 734, FP 0, FN 11. Majority-class training baseline test accuracy **48.577%**.

At score thresholds 0.2/0.8, **98.941% coverage**, **99.666% accuracy on accepted predictions**. This is score-only rejection; UI quality checks may reduce coverage further. These numbers are Python evaluation results. Browser image decoding can slightly change JPEG pixels; the UI's separate per-demo comparison measures the running browser result.

### Cross-camera stress experiment

Three independently trained audit models exclude one camera from both train and validation, then test on that camera's test frames. Production weights are unchanged.

| Excluded camera | Test crops | Accuracy | Occupied F1 |
|---|---:|---:|---:|
| PUCPR | 900 | 99.778% | 99.672% |
| UFPR04 | 251 | 84.462% | 82.667% |
| UFPR05 | 360 | 97.778% | 98.837% |

**The 84.5% result is a material limitation. New cameras require labelled local validation, calibration and potentially retraining.** This is not a leave-one-dealership-out test. `dist/data/cross-camera.json` contains the protocol and confusion matrices.

## Architecture

1. User supplies an image/video/camera; a fixed per-space rectangle is configured once per view.
2. Nearest-neighbour sample of each zone to 32×32 RGB.
3. HOG: 4×4 cells, 9 unsigned gradient bins (144 values); 16 luminance means + 16 standard deviations + 48 mean RGB values = 224 features.
4. Train-set standardization → 48 ReLU units → sigmoid with validation-set temperature scaling.
5. Basic dark / exposure / low-detail checks; score rejection band. Scores are NOT a calibrated guarantee for out-of-distribution inputs.
6. For live/video: three consecutive observed predictions confirm a state. Uncertainty resets dwell; tab hiding stops observation. One static photo never creates an overstay finding.
7. Rules generate human-verification tasks for restricted occupancy, uncertain zones and long observed dwell at a charging bay. Occupancy does not imply active charging, engine type, vehicle identity or wrongdoing.
8. Downloadable JSON records pixel-derived scores, quality checks, decisions, acknowledgments and observed latency; it does not include image pixels.

This is a specialised trained model with an operational decision layer, not an LLM wrapper. Rectangular zones are manually configured for a new camera; the application does not automatically discover parking geometry or general objects. No tracking of driver identity, faces or number plates.

## Performance and cost

Observed in the Codex browser on an **Apple M5**, macOS 26.4.1: initial demo processing of 100 zones at 1280×720 was **13 ms** (one observation, not p95 or an SLA). The UI measures actual preprocessing, feature extraction, neural inference and rules with `performance.now()` every run. Image decoding, loading, rendering and model download are outside that timer. Video deliberately samples about once per second; tab foreground and a fixed camera are required. No claim is made about phone speed; judges see their actual measurement.

Model JSON is about 233 KiB. No external runtime inference calls or paid API charges: marginal **API cost $0/frame**. This excludes device amortisation, camera, electricity, hosting, integration and maintenance.

Editable illustrative economics: 80 checks/day × 1.5 saved minutes × $4/hour × 26 days = $208/month time-equivalent benefit; $40/month operating allowance leaves $168; $300 setup gives 1.79 months simple payback. These are assumptions, not quoted prices, revenue, or measured partner savings. $40 allowance can be budgeted as hosting $5, existing-device electricity $5 and support $30; verify locally. New cameras/devices and integration work are excluded. Saved time only becomes cash savings if capacity is redeployed.

## Demo sequence (under 2 minutes)

1. Analyse a bundled PKLot image; show free / occupied / review and actual ms.
2. Open “Проверка модели”; compare with ground truth (labels used only here).
3. Upload an allowed jury image, add one rectangle per parking bay, analyse.
4. On a static image, test artificial darkness and show abstention; restore.
5. Export JSON; show adjustable ROI.

Bundled frames are test data selected at fixed positions, not picked by score. Weather names are unavailable in the mirror; no per-weather metrics are claimed. Synthetic darkening and blur are stress manipulations, not real night/rain validation. Real webcam permission and mobile hardware have not been field-tested. Do not display personal/confidential images or identifiable faces/plates without consent.

## Operational and business limits

- No real partner pilot, interviews or validated willingness to pay yet.
- No validated night, rain, local Toyota/BYD/Ducati or plate performance.
- Shadows, unusual viewpoints, occlusion and camera motion can break predictions.
- Open browser only; no continuous unattended 24/7 service or RTSP ingestion.
- No integration with a partner's CRM, charger telemetry or dispatch system.
- Weak moat today: public code and base model are easy to reproduce. Proposed moat is consented local data, feedback and integration. This is a future strategy, not an existing asset.
- Proposed 14-day pilot acceptance targets: ≥95% accepted accuracy at ≥80% coverage and ≥30% reduction in search time, measured against baseline by the yard manager. These are targets, not results.

## Data, attribution and licences

See [DATA_SOURCES.md](DATA_SOURCES.md), [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), [MODEL_CARD.md](MODEL_CARD.md). No elimination-stage code or pre-final team project was supplied. Original code was authored in this session with Codex; team members must review and understand it before presenting it as their submission. Third-party datasets and libraries are explicitly credited.

## Deployment and final freeze

The tracked `.openai/hosting.json` declares static hosting on Sites. Source is pushed before publishing and the published version is bound to a commit. `docs/SUBMISSION.txt` contains the final URLs and a 7-minute defence checklist. A GitHub Actions workflow validates source on push; publishing on Sites records its own commit-bound history.

Before the deadline submit the **HTTPS prototype**, **PPTX deck**, and **public GitHub repository**. At deadline record the Git SHA and saved hosting version. After deadline only redeploy that exact submitted commit; do not run training or edit the production deployment. Prepare and rehearse a real screen recording as a technical fallback.
