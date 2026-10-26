"""Stage 2: hazard classifier per skill level, trained on out-of-fold Stage 1 corrected values.

Run:  python -m src.train_stage2   (after train_stage1)
Writes: models/stage2.joblib, data/processed/thresholds.json (first run), stage2_metrics.csv, stage2_summary.json
"""
import json

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import precision_recall_fscore_support

from config import MODELS, PROC, SKILLS
from src import hazard
from src.cv import time_folds
from src.dataset import load_training_table

S2_FEATURES = ["hs_corr", "tm_corr", "wind_corr", "f_wavedir_sin", "f_wavedir_cos", "f_swell_frac",
               "tide_h", "tide_rate", "hour_sin", "hour_cos", "season"]
MIN_CLASS_ROWS = 30


def make_clf():
    return LGBMClassifier(n_estimators=250, learning_rate=0.04, num_leaves=15, min_child_samples=30,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.8, class_weight="balanced",
                          random_state=0, verbose=-1)


def main():
    df = load_training_table()
    oof = pd.read_parquet(PROC / "stage1_oof.parquet")
    df = df.join(oof)
    df["hs_corr"] = df["hs_corr"].fillna(df["f_hs"])
    df["tm_corr"] = df["tm_corr"].fillna(df["f_tm"])
    df["wind_corr"] = df["wind_corr"].fillna(df["f_wind"])
    df = df[df["obs_hs"].notna() & df["obs_tz"].notna() & df["hs_corr"].notna()]
    thr = hazard.load_thresholds() if hazard.PATH.exists() else hazard.make_thresholds(df["obs_hs"], df["obs_tz"])
    feats = [c for c in S2_FEATURES if c in df and df[c].notna().mean() > 0.5]
    rows, bundle, summary = [], {}, {}
    for skill in SKILLS:
        y = hazard.hazard_level(df["obs_hs"], df["obs_tz"], skill, thr).astype(int)
        counts = pd.Series(y).value_counts().reindex([0, 1, 2], fill_value=0).to_dict()
        merge = counts[2] < MIN_CLASS_ROWS            # too few HIGH hours: merge HIGH into MODERATE
        y_use = np.where(y == 2, 1, y) if merge else y
        labels = [0, 1] if merge else [0, 1, 2]
        raw_pred = hazard.hazard_level(df["f_hs"], df["f_tm"], skill, thr)
        raw_pred = np.nan_to_num(raw_pred, nan=0).astype(int)
        raw_pred = np.where(raw_pred == 2, 1, raw_pred) if merge else raw_pred
        X = df[feats]
        for k, (tr, te) in enumerate(time_folds(len(df))):
            clf = make_clf()
            if len(np.unique(y_use[tr])) < 2:
                continue
            clf.fit(X.iloc[tr], y_use[tr])
            for name, pred in (("model", clf.predict(X.iloc[te])), ("raw_forecast_thresholds", raw_pred[te])):
                p, r, f, s = precision_recall_fscore_support(y_use[te], pred, labels=labels, zero_division=0)
                for i, lab in enumerate(labels):
                    rows.append({"skill": skill, "fold": k, "method": name, "class": lab,
                                 "precision": p[i], "recall": r[i], "f1": f[i], "support": int(s[i])})
        res = pd.DataFrame([r for r in rows if r["skill"] == skill])
        top = labels[-1]
        rec = res[res["class"] == top].groupby("method")["recall"].mean()
        prec = res[res["class"] == top].groupby("method")["precision"].mean()
        summary[skill] = {"class_counts": counts, "merged_high_into_moderate": bool(merge),
                          "mean_recall_top_class": {k: round(float(v), 3) for k, v in rec.items()},
                          "mean_precision_top_class": {k: round(float(v), 3) for k, v in prec.items()},
                          "recall_beats_raw_baseline": bool(rec.get("model", 0) > rec.get("raw_forecast_thresholds", 0))}
        bundle[skill] = {"model": make_clf().fit(X, y_use), "features": feats, "merged": bool(merge)}
    pd.DataFrame(rows).to_csv(PROC / "stage2_metrics.csv", index=False)
    (PROC / "stage2_summary.json").write_text(json.dumps(summary, indent=2))
    joblib.dump(bundle, MODELS / "stage2.joblib")
    print("Thresholds in use:", json.dumps(thr, indent=2))
    for skill, s in summary.items():
        print(f"\n[{skill}] class counts {s['class_counts']} merged_high={s['merged_high_into_moderate']}")
        print("   top class recall   :", s["mean_recall_top_class"])
        print("   top class precision:", s["mean_precision_top_class"])
        print("   (recall alone can be inflated by over-warning, so always read it with precision)")


if __name__ == "__main__":
    main()
