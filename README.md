# Tennis match prediction

Pre-match ATP winner prediction with logistic regression, random forest,
gradient boosting and a linear SVM, following the CS229 paper's feature set
(plus Elo).

**Data:** Jeff Sackmann's `tennis_atp` repository, used by the paper, has been removed from GitHub.
The project now downloads the TML-Database (github.com/Tennismylife/TML-Database), which has the same
columns. Non-commercial use only; credit TML and Sackmann's original work (CC BY-NC-SA 4.0).

## Run

```bash
pip install -r requirements.txt
python train.py                     # downloads data, builds features, trains, writes artifacts/
streamlit run app.py                # opens the UI
python predict.py --a "Player One" --b "Player Two" --surface Clay   # optional CLI
```

Useful options: `--test-year 2019` (paper split), `--models logreg`, `--rebuild`, `--refresh`.
If the automatic download is blocked, download the yearly files (`1997.csv` ... `2026.csv`) from the
TML-Database page in a browser and put them in `data/`. Old Sackmann-style files
(`atp_matches_YYYY.csv`, `atp_matches_qual_chall_YYYY.csv`) also work if you have them.
The GitHub copy of the current season can lag behind; the app warns when data is over 60 days old.

## Files

| File | Purpose |
|---|---|
| `data.py` | download and clean match files |
| `features.py` | leak-free features, shared by training and the app |
| `train.py` | time-based split, C selection, evaluation, saving |
| `predict.py` | inference, match-up snapshot, CLI |
| `app.py` | Streamlit UI |

## Method notes

- Every match becomes two mirrored rows (winner first, loser first); probabilities from both are averaged.
- Features only use matches played before the one being predicted.
- Train on seasons before the test year (plus Challenger files if you supply them), test on the ATP matches of the test year.

## Results on real data (test season 2025, 2,839 matches)

Higher-ranked wins 64.3%, higher Elo wins 63.8%, logistic regression 65.8%, gradient boosting 65.6%, linear SVM 65.7%.
