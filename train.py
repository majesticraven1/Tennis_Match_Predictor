"""Train and evaluate tennis match predictors.

    python train.py                    # download data, build features, train all models
    python train.py --test-year 2019   # reproduce the paper's split
    python train.py --models logreg    # faster run
"""
import argparse
import datetime as dt
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from data import download, load_matches
from features import build_dataset


def make_model(name, c=0.1):
    imp = ("imp", SimpleImputer(strategy="median"))
    sc = ("sc", StandardScaler())
    if name == "logreg":       # the paper's main model: L2, liblinear
        clf = LogisticRegression(solver="liblinear", C=c, max_iter=1000)
        return Pipeline([imp, sc, ("clf", clf)])
    if name == "random_forest":
        clf = RandomForestClassifier(n_estimators=300, criterion="entropy",
                                     min_samples_leaf=10, n_jobs=-1, random_state=0)
        return Pipeline([imp, ("clf", clf)])
    if name == "gradient_boosting":
        clf = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=300,
                                             l2_regularization=1.0, random_state=0)
        return Pipeline([imp, ("clf", clf)])
    if name == "svm":          # linear SVM with calibrated probabilities (kernel SVMs are too slow)
        clf = CalibratedClassifierCV(LinearSVC(C=0.05, dual=False), cv=3)
        return Pipeline([imp, sc, ("clf", clf)])
    raise ValueError(name)


def win_probs(model, X):
    """P(winner wins) per match, averaging the two mirrored rows (winner-first, loser-first)."""
    p = model.predict_proba(X)[:, 1]
    return 0.5 * (p[0::2] + 1.0 - p[1::2])


def score(p_win):
    p = np.clip(p_win, 1e-6, 1 - 1e-6)
    return {"accuracy": float((p_win > 0.5).mean()),
            "log_loss": float(-np.log(p).mean()),
            "brier": float(((1 - p_win) ** 2).mean())}


def pick_test_year(matches):
    counts = matches[matches.tour == "atp"].groupby("year").size()
    full = counts[(counts >= 1500) & (counts.index < dt.date.today().year)]
    return int((full if len(full) else counts).index.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=1997)
    ap.add_argument("--end", type=int, default=dt.date.today().year)
    ap.add_argument("--test-year", type=int, default=None)
    ap.add_argument("--models", nargs="+", default=["logreg", "random_forest", "gradient_boosting", "svm"])
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--refresh", action="store_true", help="re-download recent seasons")
    ap.add_argument("--rebuild", action="store_true", help="ignore the cached feature table")
    a = ap.parse_args()

    print("Loading data")
    download(a.data_dir, a.start, a.end, a.refresh)
    matches = load_matches(a.data_dir, a.start, a.end)
    test_year = a.test_year or pick_test_year(matches)
    print(f"{len(matches):,} matches, test season {test_year}")

    cache = Path(a.data_dir) / f"features_{a.start}_{a.end}.pkl"
    if cache.exists() and not a.rebuild:
        X, fb = joblib.load(cache)
    else:
        print("Building features")
        X, fb = build_dataset(matches)
        joblib.dump((X, fb), cache)

    mid = X["mid"].to_numpy()
    year, tour = matches["year"].to_numpy()[mid], matches["tour"].to_numpy()[mid]
    feats = [c for c in X.columns if c not in ("label", "mid")]
    y = X["label"].to_numpy()

    def rows(mask):
        return X.loc[mask, feats], y[mask]

    train = year < test_year
    test = (year == test_year) & (tour == "atp")
    Xtr, ytr = rows(train)
    Xte, _ = rows(test)

    # choose the logistic-regression regularisation on the season before the test year
    fit, val = year < test_year - 1, (year == test_year - 1) & (tour == "atp")
    Xf, yf = rows(fit)
    Xv, _ = rows(val)
    cs = (0.003, 0.01, 0.03, 0.1, 1.0)
    losses = [score(win_probs(make_model("logreg", c).fit(Xf, yf), Xv))["log_loss"] for c in cs]
    best_c = cs[int(np.argmin(losses))]
    print(f"logreg C={best_c} (validation log loss {min(losses):.4f})")

    # baselines on the same test matches
    m = mid[test][0::2]
    wr, lr = matches["winner_rank"].to_numpy()[m], matches["loser_rank"].to_numpy()[m]
    ok = ~np.isnan(wr) & ~np.isnan(lr) & (wr != lr)
    baseline_rank = float((wr[ok] < lr[ok]).mean())
    baseline_elo = float((Xte["d_elo"].to_numpy()[0::2] > 0).mean())

    models, results = {}, {}
    for name in a.models:
        print(f"Training {name}")
        model = make_model(name, best_c).fit(Xtr, ytr)
        models[name] = model
        results[name] = score(win_probs(model, Xte))
        print("  ", {k: round(v, 4) for k, v in results[name].items()})
    print(f"Baseline, higher-ranked player wins: {baseline_rank:.4f}")
    print(f"Baseline, higher Elo wins:           {baseline_elo:.4f}")

    out = Path(a.out)
    out.mkdir(exist_ok=True)
    joblib.dump(models, out / "models.joblib")
    joblib.dump(fb, out / "state.joblib")
    (out / "features.json").write_text(json.dumps(feats))
    (out / "metrics.json").write_text(json.dumps({
        "test_year": test_year,
        "n_test_matches": int(test.sum() // 2),
        "n_train_matches": int(train.sum() // 2),
        "data_through": str(np.datetime64(int(fb.last_day), "D")),
        "baseline_rank": baseline_rank,
        "baseline_elo": baseline_elo,
        "logreg_C": best_c,
        "models": results,
    }, indent=2))
    print(f"Saved to {out}/")


if __name__ == "__main__":
    main()
