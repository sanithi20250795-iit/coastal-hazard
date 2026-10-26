"""Stage 1: learn the local correction (measured minus model) for wave height, wave period and wind.

Run:  python -m src.train_stage1
Writes: models/stage1.joblib, data/processed/stage1_oof.parquet, stage1_metrics.csv, stage1_summary.json
"""
import json

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import MODELS, PROC
from src.cv import time_folds
from src.dataset import S1_FEATURES, TARGETS, load_training_table

TARGET_IMPROVEMENT = 0.10     # Section 9: at least 10% lower RMSE than the raw forecast


def make_models():
    return {
        "ridge": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0)),
        "lgbm": LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=40,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                              random_state=0, verbose=-1),
    }


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = np.isfinite(y) & np.isfinite(p)
    e = p[m] - y[m]
    return {"rmse": float(np.sqrt(np.mean(e ** 2))), "mae": float(np.mean(np.abs(e))),
            "bias": float(np.mean(e)), "n": int(m.sum())}


def main():
    df = load_training_table()
    rows, bundle, oof = [], {}, pd.DataFrame(index=df.index)
    summary = {}
    for tgt, (obs_col, raw_col) in TARGETS.items():
        d = df[df[obs_col].notna() & df[raw_col].notna()]
        feats = [c for c in S1_FEATURES if c in d and d[c].notna().mean() > 0.5]
        y, raw = d[obs_col].to_numpy(), d[raw_col].to_numpy()
        resid = y - raw
        X = d[feats]
        ys = pd.Series(y, index=d.index)
        stored = {name: np.full(len(d), np.nan) for name in make_models()}
        for k, (tr, te) in enumerate(time_folds(len(d))):
            preds = {"raw": raw[te], "bias_only": raw[te] + resid[tr].mean()}
            clim = pd.Series(y[tr]).groupby(d.index.month[tr]).mean()
            preds["climatology"] = pd.Series(d.index.month[te]).map(clim).fillna(y[tr].mean()).to_numpy()
            preds["persistence"] = ys.reindex(d.index[te] - pd.Timedelta(hours=24)).to_numpy()
            for name, model in make_models().items():
                model.fit(X.iloc[tr], resid[tr])
                p = raw[te] + model.predict(X.iloc[te])
                stored[name][te] = p
                preds[name] = p
            for name, p in preds.items():
                rows.append({"target": tgt, "fold": k, "method": name, **metrics(y[te], p)})
        res = pd.DataFrame([r for r in rows if r["target"] == tgt])
        mean_rmse = res.groupby("method")["rmse"].mean()
        chosen = min(("ridge", "lgbm"), key=lambda m: mean_rmse[m])
        gain_raw = 1 - mean_rmse[chosen] / mean_rmse["raw"]
        gain_bias = 1 - mean_rmse[chosen] / mean_rmse["bias_only"]
        per_fold = res[res.method == chosen].set_index("fold")["rmse"] / res[res.method == "raw"].set_index("fold")["rmse"]
        summary[tgt] = {"chosen": chosen, "n_rows": len(d), "features": feats,
                        "mean_rmse": {k: round(float(v), 4) for k, v in mean_rmse.items()},
                        "improvement_vs_raw": round(float(gain_raw), 3),
                        "improvement_vs_bias_only": round(float(gain_bias), 3),
                        "fold_rmse_ratio_to_raw": [round(float(v), 3) for v in per_fold],
                        "target_met": bool(gain_raw >= TARGET_IMPROVEMENT and gain_bias > 0)}
        oof[f"{tgt}_corr"] = pd.Series(stored[chosen], index=d.index)
        final = make_models()[chosen].fit(X, resid)
        bundle[tgt] = {"model": final, "features": feats, "raw_col": raw_col, "name": chosen}
    pd.DataFrame(rows).to_csv(PROC / "stage1_metrics.csv", index=False)
    (PROC / "stage1_summary.json").write_text(json.dumps(summary, indent=2))
    oof.to_parquet(PROC / "stage1_oof.parquet")
    joblib.dump(bundle, MODELS / "stage1.joblib")
    for tgt, s in summary.items():
        print(f"\n[{tgt}] rows={s['n_rows']} chosen={s['chosen']}")
        print("   mean RMSE by method:", s["mean_rmse"])
        print(f"   improvement vs raw {s['improvement_vs_raw']:+.1%}, vs bias-only {s['improvement_vs_bias_only']:+.1%}, "
              f"target met: {s['target_met']}")
        print("   fold RMSE / raw RMSE:", s["fold_rmse_ratio_to_raw"])


if __name__ == "__main__":
    main()
