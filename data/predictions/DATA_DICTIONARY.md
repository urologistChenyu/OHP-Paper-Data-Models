# Anonymous OHP model predictions

- case_id: keyed anonymous study code, consistent across the cohort and reader-demo files. The mapping is retained privately.
- cohort: development_oof, internal_validation, external or prospective. The development probabilities are out-of-fold estimates.
- ohp_label: postoperative binary reference; 1 indicates at least one of the seven adverse pathological features and 0 indicates none.
- clinical_probability, conventional_radiomics_probability, habitat_probability, cnn_2d_probability, cnn_2_5d_probability: saved branch estimates on the 0-1 scale.
- ohp_probability: saved arithmetic mean of the five branch probabilities.
- classification_threshold: fixed full-precision value 0.3530539465287861.
- predicted_ohp_positive: 1 when the final probability is at or above the threshold.

Use all_cohorts_predictions.csv for analyses; the separate cohort and reader-subset files repeat selected rows for convenience. Do not concatenate these files as if they were additional patients. No patient demographic or blood-test observations are released in this table.
