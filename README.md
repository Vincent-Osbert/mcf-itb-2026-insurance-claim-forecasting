# MCF ITB 2026 — Health Insurance Claim Forecasting

Forecast monthly claim frequency, claim severity and total claim cost for August to December 2025, from 19 months of paid claims for one health insurance portfolio. Built for the Data Science Competition of MCF ITB 2026 on Kaggle as team **Analyx**.

| | |
|---|---|
| Private leaderboard | 88th, MAPE 16.19847 |
| Public leaderboard | 28th, MAPE 0.55542 |
| Best internal holdout (v25, no leakage) | MAPE 5.23% |
| Targets | 15 values: 3 series × 5 months |

## The problem

The data has 4,627 paid claims (January 2024 to July 2025) and 4,096 policies. Each month becomes one row with three numbers: the claim count, the mean claim cost and the total cost. That leaves 19 monthly points to forecast five months ahead, so every modelling choice has to survive very little data.

## Approach

The notebook keeps four versions side by side, because the team iterated on one file:

| Cell | Version | What it does |
|---|---|---|
| 0 | v22/v23 | Kalman filter, damped Holt-Winters, SARIMAX, STL + Ridge and Prophet per series, combined with inverse-MAPE² weights |
| 1 | v24 | v22 plus LightGBM and Extra Trees, adaptive weights and per-month bias correction |
| 2 | v25 | Clean rebuild: walk-forward CV, global bias correction only, 5-month holdout simulation |
| 3 | — | Plain Holt-Winters baseline |

`teammate-v16/insurance_claim_v16.py` is the teammate's v16 script, the reference that cells 0 to 2 are compared against.

All versions are scored the same way: hold out the last five known months (March to July 2025), forecast them from the months before, and compute MAPE per series.

| Version | Frequency | Severity | Total | Average |
|---|---|---|---|---|
| Teammate v16 | 5.87% | 2.44% | 7.12% | 5.14% |
| v23 (cell 0) | 5.98% | 2.38% | 7.05% | 5.14% |
| v24 (cell 1) | 5.50% | 2.64% | 3.69% | 3.94% |
| **v25 (cell 2)** | **5.96%** | **2.58%** | **7.15%** | **5.23%** |

v24 looks best, but its bias factors are fitted on the full series, holdout months included, and then reused inside the holdout simulation. v25 removes that and lands back at the teammate's level. In walk-forward CV, v25 scores 6.69% on frequency, 7.40% on severity and 13.87% on total cost.

## Leaderboard

The public leaderboard covers only August and September 2025, six numbers in all. It can be probed, and the public scores near 0.0002 at the top show that many teams did. A teammate probed it to keep the team near the finals cutoff. The 0.555% public score reflects that probing, not the models in this repository, which is why this README does not treat it as a result.

The private leaderboard (October to December) is the real test of the forecasts. There the team finished 88th with MAPE 16.20%, three times the internal holdout estimate.

## Limitations

- **The holdout was too optimistic.** The last five known months are not the same task as forecasting three to five months past the end of the data. The private score shows the gap.
- **Very short series.** 19 monthly points cannot support the number of models and weights tried here. Most of the ensemble machinery is noise at this sample size.
- **Not re-run for this repository.** The outputs are from the original runs. The cell 3 baseline aggregates the months differently, so its printed holdout score cannot be compared with the table above.

## Repository structure

```
notebooks/mcf-itb-2026-claim-forecasting.ipynb
teammate-v16/insurance_claim_v16.py
requirements.txt
```

The code comments are in Indonesian. The competition data, the problem statement PDF and all submission files are not included.

## How to run

Put `Data_Klaim (1).csv`, `Data_Polis.csv` and `sample_submission.csv` next to the notebook, install `requirements.txt`, and run the cell you want. Each cell is self-contained.

## Credits

Team Analyx, two members. The notebook is Vincent Osbert's work. The v16 script and its holdout baseline are the teammate's. The competition, data and leaderboard belong to the MCF ITB organisers.
