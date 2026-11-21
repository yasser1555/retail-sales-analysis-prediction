# Recorded notebook metrics (not recomputed)

The following values are copied from **saved execution outputs** of `notebooks/Walmart_proj.ipynb`. They are not new benchmarks or rerun results.

| Method | Features | MAE | RMSE |
| --- | --- | ---: | ---: |
| Constant-mean baseline | Target mean only | 15,177.283092117861 | Not recorded |
| Linear Regression | 6 | 14,614.447108543185 | Not recorded |
| Random Forest (50 trees) | 6 | 2,713.099422787748 | 7,349.398761689806 |
| Random Forest (50 trees) | 10 | 1,914.5209765849363 | 5,483.603255266457 |

- **R²** was not calculated or reported in the saved notebook.
- Linear Regression and both Random Forest models used a random 80/20 train/test split (`random_state=42`), not a chronological holdout.
- The mean-baseline MAE was computed over the **entire cleaned dataset**, not the held-out test subset, so it is not a directly like-for-like test-set comparison.
- The original competition evaluates **weighted MAE** with greater weight on holiday weeks; this project reported unweighted MAE and RMSE, and no competition WMAE or leaderboard score.
- The original code did not record package versions or evaluate model behavior on future unseen weeks.
