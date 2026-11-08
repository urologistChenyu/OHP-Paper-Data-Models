# Reproduce OHP analyses

## Released model probabilities

Run reproduce_prediction_results.py with --input ../data/predictions/all_cohorts_predictions.csv and --output-dir NEW_OUTPUT_DIRECTORY. With the modules in their supplied arrangement, the root Reproduce_Prediction_Results.cmd uses the bundled environment in 02_End_to_End.

The CSV columns are documented in ../data/predictions/DATA_DICTIONARY.md. Calculations preserve the published threshold and cohort assignments. A development OOF estimate is distinct from a final-model training estimate. Calibration coordinates use six within-cohort probability-quantile groups and Wilson 95% intervals, matching Figure 4e.

## Reader study

analyse_readers.py requires the approved R01-R16 original XLSX exports and reader grouping CSV. Use --input-dir APPROVED_EXPORT_DIRECTORY --reference APPROVED_MODEL_REFERENCE.csv --output-dir NEW_OUTPUT_DIRECTORY. The reference CSV fields are cohort,ID,y,clin,trad,hab,d2,d25,age,sex,size,ohp_probability. Use the analysis/requirements-reader-analysis.txt environment for that analysis. Private original answers and clinical covariates are not embedded in the public demonstration.

Public aggregate tables are under ../data/reader_aggregate_results. Their index.json maps each released filename to the corresponding analysis output.
