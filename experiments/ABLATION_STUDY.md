# XGBoost Ablation Study

## Objective
Evaluate the contribution of selected features and feature groups
to the Edge-IIoTset intrusion detection model.

## Dataset
- Dataset: Edge-IIoTset
- Model: XGBoost
- Features: 12 network/transport-level behavioral features
- Evaluation: Scenario-aware validation split

## Experiments
1. Full model with all 12 features.
2. Without tcp.dstport.
3. Without tcp.ack.
4. Without tcp.flags.
5. Without tcp.len.
6. Without selected TCP connection-flag features.
7. Without UDP features.
8. Without the DNS feature.

## Metrics
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Average Precision (AP)

## Method
For each configuration, remove the specified features, train an
XGBoost classifier using the same hyperparameters, and evaluate it
on the same validation split. Compare each result with the full
12-feature baseline.

## Reproducibility
Random seed: 42.

The training and validation data must be generated using the
project's documented preprocessing and scenario-aware splitting
procedure.

## Results
The measured results will be saved to:
`ablation_study_validation_results.csv`

Only report results produced by successfully executed experiments.
