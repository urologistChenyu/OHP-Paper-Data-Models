# OHP paper data and final models

This module contains the final five-branch OHP model, fixed preprocessing, actual inference source, anonymous case labels and model probabilities, manuscript table data and reader-study analysis code.

## Find the weights

Open models/: clinical.pkl, traditional.pkl, habitat.pkl, cnn2d_ep36.pth and cnn25d_ep16.pth are the five final risk branches. The five risk-model weights are included directly in this repository. The larger KiTS21 automatic-segmentation checkpoint is a separately downloadable release asset; run Download_Segmentation_Model.py to place it under models/kits21_fold0. Its plans, metadata and licence remain in the repository. MODEL_FILES.json lists their files and fingerprints. configs/, pipeline/frozen_tabular_state.json, pipeline/frozen_habitat_knn.joblib and pipeline/habitat_feature_names.json contain the matching preprocessing and selected-feature information.

The final risk is the arithmetic mean of the five branch probabilities. Positive classification uses probability >= 0.3530539465287861. model_contract.json records the fields and model settings. These files contain the final inference models, rather than every training checkpoint.

## Anonymous predictions

data/predictions/all_cohorts_predictions.csv contains 671 anonymous cases: 296 development OOF, 126 internal validation, 156 external and 93 prospectively enrolled cases. It includes the binary OHP label, five branch probabilities, final probability and threshold-based classification. The four cohort files are views of the same records, not additional cases. reader_subset_107_predictions.csv links the 107 reader-demonstration cases to the same anonymous IDs and predictions. Original identifiers, mapping keys, source clinical records and DICOM are not included.

Run analysis/reproduce_prediction_results.py to reproduce discrimination, average precision, Brier score, confusion counts, calibration point estimates, decision curves, paired DeLong comparisons and exact probability decomposition from the released predictions. Its output goes to a new directory chosen by the user. It does not train or recalibrate a model.

On Windows with all three modules together, run Reproduce_Prediction_Results.cmd. It uses the self-contained Python environment in ../02_End_to_End. Source-only users can install analysis/requirements-prediction-analysis.txt in their own environment. Full ready-to-run CT inference is available in the [OHP End-to-End Windows release](https://github.com/urologistChenyu/OHP-End-to-End/releases/tag/v1.0.0); the inference source and final model assets are also retained here.

## Manuscript and reader results

data/manuscript_tables contains the English data for Table 1, Table 2 and Supplementary Tables S1-S5 and S7-S19; table identifiers follow the manuscript. data/reader_aggregate_results contains the published reader-study aggregate results with its filename index. The numerical candidate-fusion comparison collection is outside this distribution.

analysis/analyse_readers.py implements the reported paired 16-reader statistics. It requires the approved original reader exports and grouping/reference inputs described in analysis/README.md; private responses are not included in this release. New public demonstration responses are separate from the research data.

See LICENSE and THIRD_PARTY_NOTICES.md for software and model attribution.

## Related software

- [Complete Windows end-to-end software](https://github.com/urologistChenyu/OHP-End-to-End)
- [Browser reader-study platform](https://github.com/urologistChenyu/OHP-Reader-Study-Platform)

Analysis version: v1.0.1. Trained model assets: v1.0.0. Patient-derived source CT, clinical records and private reader responses require institutional approval. Anonymous labels and model probabilities are public in this repository.
