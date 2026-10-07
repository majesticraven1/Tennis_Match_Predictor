"""Inference helpers used by the app and the command line.

    python predict.py --a "Player One" --b "Player Two" --surface Clay
"""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from features import UNRANKED

PRETTY = {
    "elo": "Elo rating", "elo_surf": "Elo on this surface", "win_pct": "Career win rate",
    "surf_win_pct": "Win rate on this surface", "wins": "Career wins", "losses": "Career losses",
    "h2h_diff": "Head-to-head record", "h2h_surf_diff": "Head-to-head on this surface",
    "co_w_diff": "Wins against common opponents", "co_l_diff": "Losses against common opponents",
    "month_w": "Wins in the last 30 days", "month_l": "Losses in the last 30 days",
    "ace_rate": "Aces per serve point", "df_rate": "Double faults per serve point",
    "first_in": "First serves in", "first_won": "First-serve points won",
    "second_won": "Second-serve points won", "bp_saved": "Break points saved",
    "sv_games": "Service games played", "aces_total": "Career aces", "dfs_total": "Career double faults",
    "tour_wins": "Wins at this tournament", "tour_losses": "Losses at this tournament",
    "age": "Age", "best5": "Best-of-five match",
}


def pretty(name):
    if name.startswith("p1_") or name.startswith("p2_"):
        who = "Player A" if name.startswith("p1_") else "Player B"
        return f"{who} {name[3:].replace('_', ' ').replace('rank pts', 'ranking points')}"
    base = name[2:] if name.startswith("d_") else name
    if base in PRETTY:
        return PRETTY[base]
    for tag, label in (("sform", "on this surface"), ("form", "overall")):
        if base.startswith(tag) and base[len(tag):len(tag) + 1].isdigit():
            n, wl = base[len(tag):].split("_")
            return f"{'Wins' if wl == 'w' else 'Losses'} in last {n} matches {label}"
    return base.replace("_", " ").capitalize()


def load_artifacts(model_dir="artifacts"):
    d = Path(model_dir)
    return {"models": joblib.load(d / "models.joblib"),
            "fb": joblib.load(d / "state.joblib"),
            "features": json.loads((d / "features.json").read_text()),
            "metrics": json.loads((d / "metrics.json").read_text())}


def active_players(fb, min_matches=20, within_days=730):
    """Players who have played recently, best ranked first."""
    out = []
    for pid, s in fb.players.items():
        if s.w + s.l >= min_matches and fb.last_day - s.last_day <= within_days and s.info:
            out.append({"id": pid, "name": fb.names[pid], "rank": s.info["rank"]})
    return sorted(out, key=lambda p: (p["rank"], p["name"]))


def predict_match(art, a, b, surface="Hard", best5=False, tournament="", model="logreg"):
    """Win probability for player a against player b, plus what drove it."""
    fb = art["fb"]
    ctx = {"surface": surface, "tname": tournament, "day": fb.last_day, "best5": int(best5)}
    r_ab, r_ba = fb.pair_rows(a, b, ctx, fb.players[a].info, fb.players[b].info)
    X = pd.DataFrame([r_ab, r_ba])[art["features"]]
    pipe = art["models"][model]
    p = pipe.predict_proba(X)[:, 1]
    p_a = float(0.5 * (p[0] + 1.0 - p[1]))
    drivers = []
    if hasattr(pipe.named_steps["clf"], "coef_"):
        z = pipe[:-1].transform(X.iloc[[0]])[0]
        contrib = pipe.named_steps["clf"].coef_[0] * z
        order = sorted(range(len(contrib)), key=lambda i: -abs(contrib[i]))[:6]
        drivers = [(pretty(art["features"][i]), float(contrib[i])) for i in order]
    return {"p_a": p_a, "p_b": 1.0 - p_a, "drivers": drivers}


def snapshot(art, a, b, surface):
    """Side-by-side facts for two players: (label, value for A, value for B)."""
    fb = art["fb"]
    A, B = fb.players[a], fb.players[b]

    def rec(w, l):
        return f"{w}\u2013{l}"

    def last10(P):
        seq = list(P.recent)[-10:]
        return rec(sum(seq), len(seq) - sum(seq))

    def month(P):
        m = [r for d, r in P.month if d >= fb.last_day - 30]
        return rec(sum(m), len(m) - sum(m))

    def rank(P):
        r = P.info.get("rank", UNRANKED)
        return f"#{int(r)}" if r < UNRANKED else "Unranked"

    hw, hl = A.opp.get(b, (0, 0))
    sw, sl = A.opp_surf.get((b, surface), (0, 0))
    aw, al, bw, bl, n = fb.common(A, B)
    return [
        ("Ranking", rank(A), rank(B)),
        ("Elo rating", f"{A.elo:.0f}", f"{B.elo:.0f}"),
        (f"Elo on {surface.lower()}", f"{A.elo_surf.get(surface, 1500.0):.0f}",
         f"{B.elo_surf.get(surface, 1500.0):.0f}"),
        ("Career record", rec(A.w, A.l), rec(B.w, B.l)),
        (f"Record on {surface.lower()}", rec(*A.surf.get(surface, (0, 0))), rec(*B.surf.get(surface, (0, 0)))),
        ("Last 10 matches", last10(A), last10(B)),
        ("Last 30 days", month(A), month(B)),
        ("Head-to-head", rec(hw, hl), rec(hl, hw)),
        (f"Head-to-head on {surface.lower()}", rec(sw, sl), rec(sl, sw)),
        (f"Against {n} common opponents", rec(aw, al), rec(bw, bl)),
    ]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--surface", default="Hard", choices=["Hard", "Clay", "Grass"])
    ap.add_argument("--best-of", type=int, default=3, choices=[3, 5])
    ap.add_argument("--model", default="logreg")
    args = ap.parse_args()

    art = load_artifacts()
    names = {v.lower(): k for k, v in art["fb"].names.items()}

    def find(q):
        hits = [names[q.lower()]] if q.lower() in names else [k for n, k in names.items() if q.lower() in n]
        if not hits:
            raise SystemExit(f"No player matching '{q}'")
        return max(hits, key=lambda k: art["fb"].players[k].last_day)

    a, b = find(args.a), find(args.b)
    res = predict_match(art, a, b, args.surface, args.best_of == 5, model=args.model)
    print(f"{art['fb'].names[a]}: {res['p_a']:.1%}   {art['fb'].names[b]}: {res['p_b']:.1%}")
