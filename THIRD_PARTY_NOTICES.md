# Third-party notices

PKLot data and derived weights: CC BY 4.0; see DATA_SOURCES.md for creators, paper, licence URL, mirror and changes. No partner trademarks or logos are used; mentions of Toyota Royal Motors, BYD Royal Motors and Ducati identify the competition track and proposed potential users, not endorsements.

NumPy copyright NumPy Developers; BSD-3-Clause. Pillow copyright its contributors; HPND. These are training dependencies, not bundled browser libraries. Their source licence links are in DATA_SOURCES.md. The website uses system fonts and native browser APIs.

## v2 video and pretrained detector

- **Freezy Traffic**, LIFEOFVIDS (2017), https://commons.wikimedia.org/wiki/File:Freezy_Traffic.webm ; CC BY 3.0 https://creativecommons.org/licenses/by/3.0/ . Original: https://upload.wikimedia.org/wikipedia/commons/5/5b/Freezy_Traffic.webm . Converted to H.264 MP4, reduced to 960×540, audio omitted; full ~20-second continuous clip. Used as a functional demo, not a labelled evaluation set or partner footage.
- **TensorFlow.js 4.22.0** and **COCO-SSD 2.2.3**, Google/TensorFlow contributors, Apache-2.0. Official project https://github.com/tensorflow/tfjs-models/tree/master/coco-ssd . Bundles obtained from versioned npm packages via jsDelivr; weights from official Google storage: https://storage.googleapis.com/tfjs-models/savedmodel/ssdlite_mobilenet_v2/model.json . All served locally, no CDN dependency at runtime. Licence preserved in docs/licenses/TensorFlow-Apache-2.0.txt.
- The pretrained vehicle detector is third-party work. BayWatch authors implemented the adapter, tiled inference, duplicate suppression, temporary tracking, upload/player handling and operational UI. The independently trained PKLot occupancy model remains a distinct component with its own licence and evaluation.
- File hashes and source URLs: docs/v2-assets.json.
