"""Tennis match predictor UI.   Run: streamlit run app.py"""
import html

import pandas as pd
import streamlit as st

from predict import active_players, load_artifacts, predict_match, snapshot
from features import UNRANKED

st.set_page_config(page_title="Tennis match predictor", page_icon="\U0001F3BE", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;700;800&display=swap');
:root { --ink:#12202E; --muted:#5B6B7C; --a:#2458B8; --b:#C9D3E0; --ball:#D9E021; --line:#E3E8EF; --panel:#F4F6F9; }
.stApp, .stApp p, .stApp label, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp button, .stApp input { font-family:'Manrope',sans-serif; }
.block-container { max-width: 820px; padding-top: 2.5rem; }
#MainMenu, footer { visibility: hidden; }
h1 { font-weight:800; letter-spacing:-0.02em; margin-bottom:0.2rem; }
.lede { color:var(--muted); font-size:1.02rem; line-height:1.55; max-width:60ch; margin-bottom:1.6rem; }
.result { display:flex; justify-content:space-between; align-items:flex-end; margin:1.6rem 0 0.7rem; gap:1rem; }
.side .nm { font-weight:700; font-size:1.05rem; color:var(--ink); }
.side .pct { font-weight:800; font-size:3.4rem; line-height:1; letter-spacing:-0.03em; color:var(--muted); }
.side .pct.lead { color:var(--a); }
.side .odds { color:var(--muted); font-size:0.85rem; margin-top:0.25rem; }
.side.right { text-align:right; }
.bar { height:14px; background:var(--b); border-radius:7px; overflow:hidden; }
.bar .fill { height:100%; background:var(--a); border-right:4px solid var(--ball); box-sizing:border-box; }
.note { color:var(--muted); font-size:0.85rem; margin-top:0.6rem; }
.facts { width:100%; border-collapse:collapse; margin-top:0.4rem; }
.facts td { padding:0.55rem 0.4rem; border-bottom:1px solid var(--line); font-size:0.95rem; }
.facts td.v { width:28%; font-weight:700; color:var(--ink); }
.facts td.l { text-align:center; color:var(--muted); }
.facts td.r { text-align:right; }
.drv { display:flex; align-items:center; gap:0.8rem; padding:0.4rem 0; font-size:0.93rem; }
.drv .t { flex:0 0 46%; color:var(--ink); }
.drv .track { flex:1; height:8px; background:var(--panel); border-radius:4px; position:relative; }
.drv .mark { position:absolute; top:0; height:8px; border-radius:4px; }
.drv .who { flex:0 0 22%; text-align:right; color:var(--muted); font-size:0.85rem; }
.stTabs [data-baseweb="tab"] { font-weight:700; }
@media (max-width: 640px) { .side .pct { font-size:2.4rem; } .drv .t { flex-basis:40%; } }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_artifacts():
    return load_artifacts()


st.markdown("# Tennis match predictor")

try:
    art = get_artifacts()
except FileNotFoundError:
    st.info("No trained model found yet. Run `python train.py` in this folder, then reload this page.")
    st.stop()

fb, metrics = art["fb"], art["metrics"]
data_through = pd.Timestamp(metrics["data_through"]).strftime("%d %b %Y")
st.markdown(
    f'<p class="lede">Choose two players and a surface to see each player\'s chance of winning. '
    f'Estimates use results, form, surface record, head-to-head and serve statistics up to {data_through}.</p>',
    unsafe_allow_html=True)

if (pd.Timestamp.today() - pd.Timestamp(metrics["data_through"])).days > 60:
    st.caption(f"The match data ends on {data_through}. Later results, rankings and form are not included.")

tab_predict, tab_results, tab_about = st.tabs(["Predict", "Model results", "How it works"])

MODEL_LABELS = {"logreg": "Logistic regression", "random_forest": "Random forest",
                "gradient_boosting": "Gradient boosting", "svm": "Linear SVM"}

with tab_predict:
    players = active_players(fb)
    by_id = {p["id"]: p for p in players}

    def label(pid):
        p = by_id[pid]
        return f"{p['name']} (#{int(p['rank'])})" if p["rank"] < UNRANKED else p["name"]

    ids = [p["id"] for p in players]
    c1, c2 = st.columns(2)
    a = c1.selectbox("Player A", ids, index=0, format_func=label)
    b = c2.selectbox("Player B", ids, index=1, format_func=label)

    c3, c4, c5 = st.columns([1.2, 0.8, 1.4])
    surface = c3.radio("Surface", ["Hard", "Clay", "Grass"], horizontal=True)
    best_of = c4.radio("Best of", [3, 5], horizontal=True)
    tournaments = ["Not specified"] + sorted(t for t, d in fb.tournaments.items() if fb.last_day - d <= 400)
    tournament = c5.selectbox("Tournament", tournaments,
                              help="Adds each player's history at this event to the estimate.")

    model_names = [m for m in MODEL_LABELS if m in art["models"]]
    model = st.selectbox("Model", model_names, format_func=MODEL_LABELS.get)

    if a == b:
        st.warning("Pick two different players.")
    else:
        res = predict_match(art, a, b, surface, best_of == 5,
                            "" if tournament == "Not specified" else tournament, model)
        name_a, name_b = html.escape(by_id[a]["name"]), html.escape(by_id[b]["name"])
        pa, pb = res["p_a"], res["p_b"]
        st.markdown(f"""
        <div class="result">
          <div class="side"><div class="nm">{name_a}</div>
            <div class="pct {'lead' if pa >= 0.5 else ''}">{pa:.0%}</div>
            <div class="odds">Fair odds {1 / pa:.2f}</div></div>
          <div class="side right"><div class="nm">{name_b}</div>
            <div class="pct {'lead' if pb > 0.5 else ''}">{pb:.0%}</div>
            <div class="odds">Fair odds {1 / pb:.2f}</div></div>
        </div>
        <div class="bar"><div class="fill" style="width:{pa * 100:.1f}%"></div></div>
        <div class="note">Fair odds are the decimal odds implied by the model, before any bookmaker margin.</div>
        """, unsafe_allow_html=True)

        st.markdown("### Match-up")
        body = "".join(
            f'<tr><td class="v">{html.escape(x)}</td><td class="l">{html.escape(lbl)}</td>'
            f'<td class="v r">{html.escape(y)}</td></tr>'
            for lbl, x, y in snapshot(art, a, b, surface))
        st.markdown(f'<table class="facts">{body}</table>', unsafe_allow_html=True)

        if res["drivers"]:
            st.markdown("### What moved the estimate")
            top = max(abs(v) for _, v in res["drivers"]) or 1.0
            rows = ""
            for name, v in res["drivers"]:
                w = abs(v) / top * 50
                pos = f"left:50%;width:{w:.1f}%;background:var(--a)" if v >= 0 \
                    else f"left:{50 - w:.1f}%;width:{w:.1f}%;background:#8896A8"
                who = name_a if v >= 0 else name_b
                rows += (f'<div class="drv"><div class="t">{html.escape(name)}</div>'
                         f'<div class="track"><div class="mark" style="{pos}"></div></div>'
                         f'<div class="who">Favours {who.split()[-1]}</div></div>')
            st.markdown(rows, unsafe_allow_html=True)
            st.caption("Shown for logistic regression: each bar is the feature's push on the log-odds.")

with tab_results:
    test_year = metrics["test_year"]
    st.markdown(
        f"Models were trained on matches before {test_year} and tested on the "
        f"{metrics['n_test_matches']:,} main-tour matches of {test_year}.")
    table = pd.DataFrame([
        {"Model": "Higher-ranked player wins", "Accuracy": metrics["baseline_rank"], "Log loss": None, "Brier score": None},
        {"Model": "Higher Elo wins", "Accuracy": metrics["baseline_elo"], "Log loss": None, "Brier score": None},
        *[{"Model": MODEL_LABELS.get(k, k), "Accuracy": v["accuracy"], "Log loss": v["log_loss"],
           "Brier score": v["brier"]} for k, v in metrics["models"].items()],
    ])
    st.dataframe(table, hide_index=True, width="stretch", column_config={
        "Accuracy": st.column_config.NumberColumn(format="percent"),
        "Log loss": st.column_config.NumberColumn(format="%.3f"),
        "Brier score": st.column_config.NumberColumn(format="%.3f"),
    })
    st.caption("Lower log loss and Brier score mean better-calibrated probabilities. "
               "The two baselines only pick a winner, so they have no probability scores.")

with tab_about:
    st.markdown("""
**Data.** ATP main-tour match results from 1997, from the TML-Database, which follows the column layout of
Jeff Sackmann's tennis_atp files. Challenger and qualifying files are also used if they are added to the data folder,
so players new to the tour can have a longer history.

**Features.** For every match, both players are described using only earlier matches:
tournament history, form over the last 5, 10, 15 and 25 matches, the last 30 days, surface form,
career record, head-to-head (overall and on the surface), results against common opponents,
rank, ranking points, seed, height, handedness and career serve statistics.
Elo ratings are added on top of the paper's set.

**Method.** Each match becomes two mirrored rows, one per player order. The model scores both,
and the two scores are averaged so the result does not depend on who is listed first.
Training uses earlier seasons only and testing uses a later season, so there is no look-ahead.

**Limits.** Injuries, weather and motivation are not in the data. Treat the output as an estimate,
not a certainty.
""")
