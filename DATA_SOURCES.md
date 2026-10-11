# Data sources and licence inventory

## PKLot images and annotations — used for training, validation, test and demo

Original authors: Paulo R. L. de Almeida, Luiz S. Oliveira, Eunelson J. Silva Jr, Alceu S. Britto Jr, Alessandro L. Koerich.

Almeida et al. (2015), “PKLot – A robust dataset for parking lot classification”, Expert Systems with Applications 42(11):4937–4949. DOI https://doi.org/10.1016/j.eswa.2015.02.009

Official source and licence declaration: https://web.inf.ufpr.br/luizoliveira/research-interests/pklot/
Licence: Creative Commons Attribution 4.0 International (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/

Download mirror: https://huggingface.co/datasets/dronefreak/PKLot
Pinned revision: `577e9c942e596819798d42fdc4d14d0b6646a43f`
Mirror reformats original polygons into axis-aligned boxes and supplies camera-day-disjoint splits. Its detailed data card is retained in `docs/PKLOT_MIRROR.md` with attribution. Mirror is unofficial.

117 frames sampled with seed 42: 72 training, 18 validation, 27 test. All URLs and image SHA-256 are in `docs/dataset-manifest.json`. Three test JPEGs redistributed in `dist/assets/` are byte-for-byte copies (no synthetic imagery). Their identities and geometry are in `dist/data/demos.json`. Ground-truth labels appear only in evaluation, never in inference. Camera locations are public dataset sites, not partner premises. Distant overhead demo shots do not show legible number plates or identifiable faces at supplied resolution.

Transformations: 32×32 nearest-neighbour crops, HOG/color features, train normalization and model fitting. UI overlays drawn at run time. Optional artificial darkness uses 95% black overlay; blur downsamples to 16×9 then enlarges. Those transformed pixels are used only for interactive stress demonstrations and excluded from dataset metrics. No generative images or synthetic labelled data used in training.

## Software and original model

- NumPy 2.3.5: BSD-3-Clause, https://github.com/numpy/numpy/blob/main/LICENSE.txt
- Pillow 12.3.0: HPND, https://github.com/python-pillow/Pillow/blob/main/LICENSE
- Python and Node runtimes: standard language/runtime licences; no bundled copies.
- Original JavaScript UI, feature extraction, decision rules, tests and Python training code: MIT, see LICENSE.
- Original trained weights in `dist/models/occupancy.json`: CC BY 4.0 with BayWatch and original PKLot authors credited. No third-party pretrained weights used.
- Codex: coding, research, documentation and slide assistance, disclosed. The team must own its final decisions and understand the code.

The default website has no external JavaScript, analytics, fonts or inference API dependencies. No uploaded image leaves the browser through application code. Hosting necessarily serves static resources and may maintain ordinary request logs.
