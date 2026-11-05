# Features: how each is calculated

Unit of analysis: every FBS team (138 in 2026) as of a committee release.
**Release week N uses only games from CFBD weeks <= N-1** (the Tuesday "Week 10" show follows CFBD week 9).
Before modeling, every numeric feature is **z-scored within its release week** across all FBS teams.
`n_fcs_opp` counts non-FBS opponents; those games count in record and margin, but not in FBS-only features (SOS, quality wins).

## Record
| Feature | Definition |
|---|---|
| wins, losses, win_pct | Completed games, including vs. FCS. |
| avg_margin_capped | Mean of (points for - points against), each game clipped to [-28, +28]. |

## Strength of schedule
| Feature | Definition |
|---|---|
| sos_opp_sp | Mean SP+ rating of FBS opponents played (SP+ snapshot at release). |
| n_fbs_opp, n_p4_opp, n_fcs_opp | Counts of FBS / Power 4 (SEC, Big Ten, Big 12, ACC) / non-FBS opponents played. |

## Resume
| Feature | Definition |
|---|---|
| quality_wins_sp25 | Wins over teams in the **SP+ top 25**. SP+ is a stand-in for "currently good": using the committee's own Top 25 for the target week would be circular. |
| wins_vs_prev_top25 | Wins over teams in **last week's** committee Top 25 (0 for Week 10: no earlier list). |
| wins_vs_winning | Wins over FBS opponents with win% > .500 (all their games, through the cutoff). |
| bad_losses | Losses to a non-FBS team, a team with win% < .500, or a team outside the SP+ top 40. |
| best_win_sp_rank | Best (lowest) SP+ rank among FBS teams beaten (139 if none). **Lower is better.** |

## Power ratings (CFBD snapshots; all four are available on the free tier)
`sp` (SP+ overall), `sp_off`, `sp_def` (**lower = better**), `srs`, `elo`, `fpi`.
Ratings have no "as of week" option, so each is stored under its release stamp when `predict` runs.
SRS lists a few names twice (e.g. Charlotte, Troy); the row with a conference is kept.

## Efficiency (`/stats/season/advanced`, `endWeek=N-1`, garbage time excluded)
`off_ppa`, `def_ppa_allowed`, `off_success`, `def_success_allowed`, `off_explosiveness`, `def_explosiveness_allowed`. Defensive features are what the defense **allows** (lower = better).

## Context
| Feature | Definition |
|---|---|
| is_p4, is_independent | Conference group (G5 is the reference). |
| conf_wins, conf_losses, conf_win_pct | FBS conference games only (independents: win% falls back to overall). |
| conf_games_behind | ((leader wins - my wins) + (my losses - leader losses)) / 2, leader = best conference record. Lower is better. No tiebreakers, no divisions. |
| conf_contender | 1 if conf_games_behind <= 1 and not independent. A rough proxy: it ignores games remaining and head-to-head tiebreakers. |
| weeks_since_loss | N-1 minus the CFBD week of the last loss (0 = lost last week); unbeaten = N. |
| prev_committee_rank | Last week's committee rank (26 if unranked); empty for Week 10. Only used by the `combined_inertia` framing. |

## Head-to-head
Not a team feature. In the pairwise model each pair carries `h2h` = +1 if A beat B this season (FBS games before the cutoff), -1 if B beat A, 0 otherwise. Its coefficient is applied after scoring: adjacent teams in the top 40 are swapped when the head-to-head result outweighs the score gap.

## Framings
- **resume**: record + SOS + resume features.
- **power**: SP+/SRS/Elo/FPI + efficiency.
- **combined**: everything above + context.
- **combined_inertia**: combined + prev_committee_rank (activates once Week 11 exists).

## Collinearity handling
Each framing is pruned by VIF (> 10). At each step, among features over the limit, the first one in this priority list is dropped:
win_pct, wins, sp_off, sp_def, fpi, srs, elo, off_ppa, def_ppa_allowed, off_success, def_success_allowed, off_explosiveness, def_explosiveness_allowed, n_fbs_opp, is_p4, conf_win_pct.
Otherwise the highest-VIF feature goes. This keeps committee-style résumé items and drops duplicative ratings first.
Everything dropped is logged in `results/collinearity.csv`.
