# Figure descriptions

Plain-language descriptions of every image in `results/`. Each file here is named after the image it describes, so `sample_photos.md` explains `sample_photos.png`, and so on.

| image | step | what it is |
|-------|------|-----------|
| sample_photos.png | 2 | example photos of the 7 lesion types (the clean data) |
| baseline_confusion.png | 3 | the classifier's mistakes on clean photos |
| retrieval_confusion.png | 4 | the retrieval model's mistakes on clean photos |
| degradation_balanced_accuracy.png | 5 | overall accuracy as photos degrade |
| degradation_melanoma_recall.png | 5 | melanoma detection as photos degrade (the core result) |
| calibration_overconfidence.png | 6 | the overconfidence gap as photos degrade |
| calibration_reliability.png | 6 | reliability: classifier vs the vote-share Lookup |
| calibration_lookup_reliability.png | 7 | reliability: the distance-fix Lookup is the most honest |
| pad_ood_detection.png | 9 | real smartphone photos: the retrieval model knows they are unfamiliar, the classifier does not |
