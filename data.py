"""Download and clean ATP match data.

Default source: the TML-Database (github.com/Tennismylife/TML-Database), which uses
the same column layout as Jeff Sackmann's tennis_atp files (whose GitHub repository
has been taken down). Files are stored as data/YYYY.csv.

Files named atp_matches_YYYY.csv and atp_matches_qual_chall_YYYY.csv (Sackmann's
naming, for example from an older copy) are also read if present. Challenger files
are only available that way; the TML-Database contains main-tour matches only.
"""
import datetime as dt
import re
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

SOURCE = "https://raw.githubusercontent.com/Tennismylife/TML-Database/master/"
ROUND_ORDER = {"Q1": 0, "Q2": 1, "Q3": 2, "R128": 3, "R64": 4, "R32": 5, "R16": 6,
               "RR": 5, "QF": 7, "SF": 8, "BR": 8, "F": 9}
FILE_RE = re.compile(r"^(?:atp_matches_(?P<chall>qual_chall_)?)?(?P<year>\d{4})$")


def download(data_dir, start, end, refresh=False):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    this_year = dt.date.today().year
    for year in range(start, end + 1):
        path = data_dir / f"{year}.csv"
        have = path.exists() or (data_dir / f"atp_matches_{year}.csv").exists()
        if have and not (refresh and year >= this_year - 1):
            continue
        try:
            urllib.request.urlretrieve(f"{SOURCE}{year}.csv", path)
            print(f"  downloaded {path.name}")
        except (urllib.error.HTTPError, urllib.error.URLError) as e:
            print(f"  could not download {year}.csv ({e})")


def load_matches(data_dir, start, end):
    frames, seen = [], set()
    for path in sorted(Path(data_dir).glob("*.csv")):
        m = FILE_RE.match(path.stem)
        if not m:
            continue                      # ignores ATP_Database.csv, futures files, etc.
        year, chall = int(m.group("year")), bool(m.group("chall"))
        if not start <= year <= end or (year, chall) in seen:
            continue
        seen.add((year, chall))
        df = pd.read_csv(path, low_memory=False)
        df["tour"] = "chall" if chall else "atp"
        frames.append(df)
    if not frames:
        raise SystemExit(
            f"No match files found in '{data_dir}'. The automatic download failed or was blocked.\n"
            "Fix: open https://github.com/Tennismylife/TML-Database, download the yearly files "
            f"(1997.csv ... 2026.csv) and put them in '{data_dir}/', then run again.")
    df = pd.concat(frames, ignore_index=True)

    df = df.dropna(subset=["winner_id", "loser_id", "surface", "tourney_date"])
    df["surface"] = df["surface"].astype(str).str.capitalize()
    df = df[df["surface"].isin(["Hard", "Clay", "Grass", "Carpet"])]
    df = df[~df["score"].fillna("").str.contains("W/O|DEF|Default|Walkover", case=False)]
    df["winner_id"], df["loser_id"] = df["winner_id"].astype(str), df["loser_id"].astype(str)
    df["date"] = pd.to_datetime(df["tourney_date"].astype(int).astype(str),
                                format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["date"])
    df["day"] = (df["date"] - pd.Timestamp("1970-01-01")).dt.days
    df["year"] = df["date"].dt.year
    df["round_order"] = df["round"].map(ROUND_ORDER).fillna(5)
    df["tourney_id"] = df["tourney_id"].astype(str)
    df = df.sort_values(["day", "tourney_id", "round_order", "match_num"], kind="stable")
    return df.reset_index(drop=True)
