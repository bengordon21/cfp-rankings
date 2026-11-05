# Changelog

Every model/feature/process change, with the reason. Manual tweaks are tagged **[manual]**.

## 2026-11-05T02:33:35+00:00 - Week 10 update: champion w10-pairwise-combined-0.05
- Trained on weeks [10]; validation = LOTO. Previous champion: none.
- Champion selection score 0.696 (Kendall 0.553, within-Top-25 pairwise acc 0.840).
- Features kept (16): avg_margin_capped, n_fbs_opp, n_p4_opp, sos_opp_sp, quality_wins_sp25, wins_vs_winning, bad_losses, best_win_sp_rank, off_success, def_success_allowed, off_explosiveness, def_explosiveness_allowed, is_independent, conf_games_behind, conf_contender, weeks_since_loss.
- VIF-dropped in combined: win_pct, wins, sp_off, sp_def, fpi, srs, elo, off_ppa, def_ppa_allowed, is_p4, conf_win_pct, sp, losses.
**Reason:** Automatic retrain; champion = best mean leave-one-team-out score over all past weeks (pairwise preferred within 0.01).
