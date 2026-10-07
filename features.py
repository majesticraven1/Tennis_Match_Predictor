"""Leak-free feature engineering for pre-match tennis prediction.

Matches are processed in chronological order. Features for a match are read
from player state that only contains earlier matches, then the state is
updated with the result. Training and the app share this exact code path.

Feature groups follow the paper: tournament form, last-N-match form, last-month
form, surface form, overall record, head-to-head (overall and per surface),
common opponents, player attributes, and cumulative serve statistics.
Elo ratings are an extra.
"""
import math
from collections import deque

import numpy as np
import pandas as pd

WINDOWS = (5, 10, 15, 25)
SURFACES = ("Hard", "Clay", "Grass")
STAT_KEYS = ("svpt", "ace", "df", "in1", "won1", "won2", "bps", "bpf", "svgms", "n")
STAT_COLS = {"svpt": "svpt", "ace": "ace", "df": "df", "in1": "1stIn", "won1": "1stWon",
             "won2": "2ndWon", "bps": "bpSaved", "bpf": "bpFaced", "svgms": "SvGms"}
UNRANKED = 1500.0
UNSEEDED = 40.0


def _num(x, default=np.nan):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(x) else x


def _ratio(n, d):
    return n / d if d > 0 else np.nan


def attrs_from_row(r, side):
    """Player attributes known before the match. side is 'winner' or 'loser'."""
    rank = max(_num(r.get(f"{side}_rank"), UNRANKED), 1.0)
    return {
        "rank": rank,
        "log_rank": math.log(rank),
        "rank_pts": _num(r.get(f"{side}_rank_points"), 0.0),
        "seed": _num(r.get(f"{side}_seed"), UNSEEDED),
        "ht": _num(r.get(f"{side}_ht")),
        "age": _num(r.get(f"{side}_age")),
        "left": 1.0 if r.get(f"{side}_hand") == "L" else 0.0,
    }


class PlayerState:
    def __init__(self):
        self.w = 0
        self.l = 0
        self.surf = {}           # surface -> [wins, losses]
        self.tour = {}           # tournament name -> [wins, losses]
        self.opp = {}            # opponent id -> [wins, losses]
        self.opp_surf = {}       # (opponent id, surface) -> [wins, losses]
        self.recent = deque(maxlen=25)      # 1 = win, 0 = loss
        self.recent_surf = {}               # surface -> deque
        self.month = deque()                # (day, result) for the last 30 days
        self.elo = 1500.0
        self.elo_surf = {}
        self.st = dict.fromkeys(STAT_KEYS, 0.0)
        self.info = {}
        self.last_day = 0


class FeatureBuilder:
    def __init__(self):
        self.players = {}
        self.names = {}
        self.tournaments = {}    # main-tour tournament name -> last day played
        self.last_day = 0

    def _get(self, pid):
        s = self.players.get(pid)
        if s is None:
            s = self.players[pid] = PlayerState()
        return s

    # ---------------------------------------------------------------- features
    @staticmethod
    def player_feats(s, ctx):
        surface, day = ctx["surface"], ctx["day"]
        n = s.w + s.l
        f = {"wins": s.w, "losses": s.l, "win_pct": s.w / n if n else 0.5}
        sw, sl = s.surf.get(surface, (0, 0))
        f.update(surf_wins=sw, surf_losses=sl, surf_win_pct=sw / (sw + sl) if sw + sl else 0.5)
        tw, tl = s.tour.get(ctx["tname"], (0, 0))
        f.update(tour_wins=tw, tour_losses=tl)
        for tag, seq in (("form", list(s.recent)),
                         ("sform", list(s.recent_surf.get(surface, ())))):
            for k in WINDOWS:
                last = seq[-k:]
                f[f"{tag}{k}_w"] = sum(last)
                f[f"{tag}{k}_l"] = len(last) - sum(last)
        month = [r for d, r in s.month if d >= day - 30]
        f["month_w"], f["month_l"] = sum(month), len(month) - sum(month)
        st = s.st
        f["ace_rate"] = _ratio(st["ace"], st["svpt"])
        f["df_rate"] = _ratio(st["df"], st["svpt"])
        f["first_in"] = _ratio(st["in1"], st["svpt"])
        f["first_won"] = _ratio(st["won1"], st["in1"])
        f["second_won"] = _ratio(st["won2"], st["svpt"] - st["in1"])
        f["bp_saved"] = _ratio(st["bps"], st["bpf"])
        f["sv_games"] = st["svgms"]
        f["aces_total"] = st["ace"]
        f["dfs_total"] = st["df"]
        f["elo"] = s.elo
        f["elo_surf"] = s.elo_surf.get(surface, 1500.0)
        return f

    @staticmethod
    def common(A, B):
        """Records of A and B against opponents both have played."""
        keys = A.opp.keys() & B.opp.keys()
        aw = sum(A.opp[k][0] for k in keys)
        al = sum(A.opp[k][1] for k in keys)
        bw = sum(B.opp[k][0] for k in keys)
        bl = sum(B.opp[k][1] for k in keys)
        return aw, al, bw, bl, len(keys)

    def _row(self, X, Y, fx, fy, ix, iy, y_id, co, ctx):
        r = {f"d_{k}": fx[k] - fy[k] for k in fx}
        hw, hl = X.opp.get(y_id, (0, 0))
        r["h2h_diff"], r["h2h_n"] = hw - hl, hw + hl
        sw, sl = X.opp_surf.get((y_id, ctx["surface"]), (0, 0))
        r["h2h_surf_diff"] = sw - sl
        r["co_w_diff"], r["co_l_diff"], r["co_n"] = co[0] - co[2], co[1] - co[3], co[4]
        for tag, attrs in (("p1", ix), ("p2", iy)):
            for k in ("rank", "log_rank", "rank_pts", "seed", "ht", "left"):
                r[f"{tag}_{k}"] = attrs[k]
        r["d_age"] = ix["age"] - iy["age"]
        r["best5"] = float(ctx["best5"])
        for s_ in SURFACES:
            r[f"surf_{s_.lower()}"] = float(ctx["surface"] == s_)
        return r

    def pair_rows(self, a, b, ctx, ia, ib):
        """Two mirrored feature rows: (a as player 1, b as player 1)."""
        A, B = self._get(a), self._get(b)
        fa, fb = self.player_feats(A, ctx), self.player_feats(B, ctx)
        aw, al, bw, bl, n = self.common(A, B)
        r_ab = self._row(A, B, fa, fb, ia, ib, b, (aw, al, bw, bl, n), ctx)
        r_ba = self._row(B, A, fb, fa, ib, ia, a, (bw, bl, aw, al, n), ctx)
        return r_ab, r_ba

    # ------------------------------------------------------------------ update
    def update(self, r, ctx, aw, al):
        wid, lid = str(r["winner_id"]), str(r["loser_id"])
        W, L = self._get(wid), self._get(lid)
        surface, tname, day = ctx["surface"], ctx["tname"], ctx["day"]

        # Elo (overall and per surface), computed before either side changes
        for key in (None, surface):
            ew = W.elo if key is None else W.elo_surf.get(key, 1500.0)
            el = L.elo if key is None else L.elo_surf.get(key, 1500.0)
            p_w = 1.0 / (1.0 + 10 ** ((el - ew) / 400.0))
            kw = 250.0 / (W.w + W.l + 5) ** 0.4
            kl = 250.0 / (L.w + L.l + 5) ** 0.4
            if key is None:
                W.elo, L.elo = ew + kw * (1 - p_w), el - kl * (1 - p_w)
            else:
                W.elo_surf[key], L.elo_surf[key] = ew + kw * (1 - p_w), el - kl * (1 - p_w)

        for P, opp_id, res in ((W, lid, 1), (L, wid, 0)):
            if res:
                P.w += 1
            else:
                P.l += 1
            idx = 0 if res else 1
            P.surf.setdefault(surface, [0, 0])[idx] += 1
            P.tour.setdefault(tname, [0, 0])[idx] += 1
            P.opp.setdefault(opp_id, [0, 0])[idx] += 1
            P.opp_surf.setdefault((opp_id, surface), [0, 0])[idx] += 1
            P.recent.append(res)
            P.recent_surf.setdefault(surface, deque(maxlen=25)).append(res)
            P.month.append((day, res))
            while P.month and day - P.month[0][0] > 30:
                P.month.popleft()
            P.last_day = day

        for P, prefix in ((W, "w_"), (L, "l_")):
            vals = {k: _num(r.get(prefix + c)) for k, c in STAT_COLS.items()}
            if not any(math.isnan(v) for v in vals.values()):
                for k, v in vals.items():
                    P.st[k] += v
                P.st["n"] += 1

        W.info, L.info = dict(aw), dict(al)
        self.names[wid], self.names[lid] = r.get("winner_name", str(wid)), r.get("loser_name", str(lid))
        if r.get("tour") == "atp" and str(r.get("tourney_level")) in ("G", "M", "A", "F", "250", "500"):
            self.tournaments[tname] = day
        self.last_day = max(self.last_day, day)


def match_ctx(r):
    return {"surface": r["surface"], "tname": r.get("tourney_name", ""),
            "day": int(r["day"]), "best5": 1 if _num(r.get("best_of"), 3) == 5 else 0}


def build_dataset(matches, log_every=20000):
    """Return (feature DataFrame with 2 rows per match, fitted FeatureBuilder).

    Row 2i has the winner as player 1 (label 1); row 2i+1 has the loser as
    player 1 (label 0). Column `mid` is the position of the match in `matches`.
    """
    fb = FeatureBuilder()
    rows = []
    for i, r in enumerate(matches.to_dict("records")):
        ctx = match_ctx(r)
        aw, al = attrs_from_row(r, "winner"), attrs_from_row(r, "loser")
        r1, r2 = fb.pair_rows(str(r["winner_id"]), str(r["loser_id"]), ctx, aw, al)
        r1["label"], r2["label"] = 1, 0
        r1["mid"] = r2["mid"] = i
        rows += [r1, r2]
        fb.update(r, ctx, aw, al)
        if log_every and (i + 1) % log_every == 0:
            print(f"  processed {i + 1:,} matches")
    return pd.DataFrame(rows), fb
