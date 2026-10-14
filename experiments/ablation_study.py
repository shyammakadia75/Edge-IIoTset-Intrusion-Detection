"""
XGBoost ablation study for Edge-IIoTset.

Prerequisites:
Prepare X_train_g, X_val_g, y_train_g, and y_val_g using the
project's existing preprocessing and scenario-aware split pipeline.

Run this script in an environment where those variables are loaded,
or integrate this code into the notebook that creates them.
"""

import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

FEATURES = [
    "tcp.ack",
    "tcp.connection.fin",
    "tcp.connection.rst",
    "tcp.connection.syn",
    "tcp.connection.synack",
    "tcp.dstport",
    "tcp.flags",
    "tcp.flags.ack",
    "tcp.len",
    "udp.port",
    "udp.time_delta",
    "dns.qry.qu",
]

ABLATIONS = {
    "Full model (12 features)": [],
    "Without tcp.dstport": ["tcp.dstport"],
    "Without tcp.ack": ["tcp.ack"],
    "Without tcp.flags": ["tcp.flags"],
    "Without tcp.len": ["tcp.len"],
    "Without TCP connection flags": [
        "tcp.connection.fin",
        "tcp.connection.rst",
        "tcp.connection.syn",
        "tcp.connection.synack",
        "tcp.flags.ack",
    ],
    "Without UDP features": ["udp.port", "udp.time_delta"],
    "Without DNS feature": ["dns.qry.qu"],
}


def run_ablation_study(
    X_train_g, y_train_g, X_val_g, y_val_g
):
    """Train each ablation model and return validation metrics."""
    missing = set(FEATURES) - set(X_train_g.columns)
    missing |= set(FEATURES) - set(X_val_g.columns)
    if missing:
        raise ValueError(f"Missing required features: {sorted(missing)}")

    results = []

    for experiment, removed in ABLATIONS.items():
        remaining = [f for f in FEATURES if f not in removed]

        clf = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss",
        )

        clf.fit(X_train_g[remaining], y_train_g)

        predictions = clf.predict(X_val_g[remaining])
        probabilities = clf.predict_proba(X_val_g[remaining])[:, 1]

        results.append({
            "Experiment": experiment,
            "Features Used": len(remaining),
            "Accuracy": accuracy_score(y_val_g, predictions),
            "Precision": precision_score(
                y_val_g, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_val_g, predictions, zero_division=0
            ),
            "F1-score": f1_score(
                y_val_g, predictions, zero_division=0
            ),
            "ROC-AUC": roc_auc_score(y_val_g, probabilities),
            "Average Precision": average_precision_score(
                y_val_g, probabilities
            ),
        })

    results_df = pd.DataFrame(results)
    output_path = "ablation_study_validation_results.csv"
    results_df.to_csv(output_path, index=False)

    print(results_df.round(5).to_string(index=False))
    print(f"\nSaved results to: {output_path}")
    return results_df


# This file deliberately does not load or split the dataset automatically.
# Call run_ablation_study(...) after loading your verified split variables.
