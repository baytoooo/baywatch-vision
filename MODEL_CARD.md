# Version 2 scope

Default vehicle detection uses pretrained COCO-SSD Lite MobileNet v2 (Apache-2.0), downloaded from official TensorFlow sources. It has no project-specific measured precision/recall. Small overhead vehicles are a known failure case. Scores are not calibrated probabilities. Vehicle detection does not discover empty parking spaces. The following card and 99.27% result apply ONLY to our separate PKLot occupancy classifier.

# BayWatch HOG-MLP v1

Task: binary occupancy of a single known parking-space crop. Not a general vehicle detector, damage classifier, charging-status detector or plate recogniser.

Architecture: 224 handcrafted features (144 HOG, 16 means, 16 standard deviations, 48 RGB means), standardization, dense 48 ReLU, dense 1 sigmoid; validation temperature 1.1. Approximately 10.8k trainable neural parameters. JSON ~233 KiB, browser CPU execution. Seed 42, Adam lr 0.0015, L2 0.0004, 160 epochs, batch approx 128. Selected epoch 160 by validation cross-entropy. Trained from scratch; no pretrained weights.

Data: 3,931 train, 1,007 validation and 1,511 test space crops from 72/18/27 PKLot frames. Selection independent of labels; camera-day splits audited. Adjacent crops/captures correlated. No test labels used in training, temperature selection or threshold tuning. Fixed rejection thresholds .2/.8 are an operational design choice, not an optimised risk guarantee.

Main evaluation: `dist/data/evaluation.json`. Cross-camera audit: `dist/data/cross-camera.json`. Main accuracy 99.272% describes same-camera future-day generalization only. With a camera omitted during training/validation, worst measured accuracy was 84.462% (UFPR04). Do not transfer the headline score to arbitrary images or partner vehicles.

Quality gate: heuristic darkness, saturation and low Laplacian detail. A passed gate does not prove image suitability; rain/occlusion/camera movement are not reliably identified. Confidence can be high on wrong/out-of-domain crops. Tiny crops (<16px in either dimension) rejected. Scores on transformed or local scenes have not been calibrated.

A human approves operational actions. No automated vehicle movement, penalties, insurance, identity decisions or safety guarantees. Uncertain frames reset dwell; elapsed time is observed foreground wall-clock time, not historical parking duration. Static images cannot substantiate overstay. Local feedback/pilot required before deployment.

Reproducibility: pinned mirror SHA, SHA-256 image manifest, deterministic seed, original training source, Python/JS numeric fixture. JPEG decoder differences and numerical versions may change borderline predictions. UI compares actual browser results to demo truth on request.
