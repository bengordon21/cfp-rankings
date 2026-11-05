# Committee ranking model: REPORT

_Auto-generated. Current champion: `w10-pairwise-combined-0.05` (trained through release week 10; 1 committee release(s) of training data)._

## 1. What the committee appears to value

The champion is a **pairwise** model on the **combined** feature set. Largest drivers (standardized coefficients; positive = helps a team's ranking; signs are flipped for lower-is-better stats such as SP+ rank of best win): average margin of victory (capped at 28) (+2.67); strength of schedule (opponents' SP+) (+1.62); games behind conference leader (+1.41); best win (SP+ rank of best team beaten) (+0.93); number of Power 4 opponents played (+0.79).
Features that work against a team: offensive explosiveness (-0.14).

**Résumé vs. power ratings.** Best score per philosophy (mean of Kendall tau and within-Top-25 pairwise accuracy, held-out):

| Philosophy | Best score |
|---|---|
| combined | 0.696 |
| pure résumé | 0.688 |
| pure power-rating | 0.622 |

combined leads pure résumé by 0.008; pure power-rating trails by 0.074. Differences under about 0.05 are not distinguishable from noise with this little data.

## 2. How that has shifted over the season

Only one committee release is in the data so far, so no shift can be measured yet. From the second release on, `results/feature_drift.csv` and `feature_drift.png` track per-week refits of the same feature set.

## 3. How accurate the forecasts have been

No forecasts have been scored yet. The first live forecast is Week 11. Until then, the numbers in `results/model_comparison.csv` are cross-validation (LOTO) on the Week 10 release: Kendall tau 0.55, within-Top-25 pairwise accuracy 0.84, 23/25 teams in the right Top 25 (11/12 for the playoff field). These are optimistic: hyper-parameters and the champion were chosen on the same data.

## 4. Caveats (please read)

- **Sample size.** 1 release(s) x 138 teams. Only 25 teams per week carry real rank information; pairwise comparisons multiply rows but are not independent.
- **Collinearity.** SP+, SRS, Elo, FPI and the efficiency stats overlap heavily with margin and record. Features were pruned by VIF (> 10), preferring to drop duplicative ratings first. Which of two near-identical features survives is arbitrary, so read coefficients as 'this family of information', not as causal weights.
- Dropped in the champion's framing: win percentage, number of wins, SP+ offense, SP+ defense, FPI, SRS, Elo, offensive PPA/play, defensive PPA/play allowed, Power 4 membership, conference win %, SP+ rating, number of losses.
- **Selection optimism.** The champion and hyper-parameters are chosen on the same validation folds, so reported CV is optimistic. Live scored weeks are the honest test.
- **Snapshot timing.** SP+/SRS/FPI/PPA have no 'as of week' option, so each week's ratings are the snapshot taken when `predict` ran. Run `predict` before Tuesday-night games.
- **Release-week alignment.** Release week N uses games through CFBD week N-1.
- **FCS opponents** are counted separately and excluded from SOS averages; a loss to an FCS team is a 'bad loss'.
- **Bracket** conference champions are projected from current conference records (no tiebreakers, no title games). Seeding rules are assumed.
- **Circularity.** 'Wins vs. committee Top 25' only uses *last* week's committee list (not available at Week 10).
