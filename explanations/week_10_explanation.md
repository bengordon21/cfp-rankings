# Week 10 committee explanation

_Explainer: ridge regression (alpha 10) fit on the committee's actual Top 25 for this release, 16 VIF-pruned features. In-sample fit: Kendall tau 0.52, within-Top-25 pairwise accuracy 0.75, 22/25 teams in the right Top 25. This describes what the committee appeared to value; it is separate from the forecasting model._

## What the committee valued this week

| Factor | Group | Weight | 90% band | Standalone corr | Share of influence | Change vs last week |
|---|---|---|---|---|---|---|
| wins over SP+ top-25 teams | Resume | +1.57 | +0.67 to +2.45 | +0.56 | 21% | n/a |
| weeks since last loss | Context | +0.93 | -0.15 to +1.71 | +0.39 | 12% | n/a |
| in conference-title contention | Context | +0.81 | +0.11 to +1.61 | +0.43 | 11% | n/a |
| strength of schedule (opponents' SP+) | Schedule | +1.68 | +1.02 to +2.51 | +0.30 | 10% | n/a |
| games behind conference leader | Context | -1.36 | -2.14 to -0.70 | -0.46 | 8% | n/a |
| number of FBS opponents played | Schedule | +0.61 | +0.04 to +1.11 | +0.17 | 7% | n/a |
| average margin of victory (capped at 28) | Resume | +1.43 | +0.71 to +2.17 | +0.56 | 7% | n/a |
| bad losses | Resume | +1.69 | +0.96 to +2.30 | -0.42 ⚠ sign differs | 4% | n/a |
| number of Power 4 opponents played | Schedule | +1.43 | +0.71 to +2.04 | +0.39 | 4% | n/a |
| offensive success rate | Efficiency | +0.50 | -0.07 to +1.04 | +0.46 | 4% | n/a |

Sign: positive = a higher value of the stat helps a team; negative = a higher value hurts (so lower is better). Weight = change in the explainer's score per 1 SD of the stat, holding the other stats fixed. Standalone corr = how the stat lines up with the committee's order by itself (across all FBS teams; positive = higher value goes with a better rank, so compare its sign with the weight). Share of influence = share of total mean |contribution| across the committee's ranked teams.

**Sign conflicts** (⚠: the weight's sign contradicts the stat's standalone correlation, with a band excluding zero): bad losses (weight +1.69, standalone -0.42), best win (SP+ rank of best team beaten) (weight +0.80, standalone -0.47). This happens when stats overlap: after accounting for the correlated stats, this one picks up leftover signal. Read its weight as 'given the other stats', not as 'the committee dislikes it'.

By group: Resume 37%, Context 33%, Schedule 21%, Efficiency 9%.

## Why each Top 25 team sits where it does

Contributions are relative to the average team in the committee's Top 25.

| Rank | Team | Helped most | Hurt most |
|---|---|---|---|
| 1 | Miami | weeks since last loss +2.89; average margin of victory (capped at 28) +1.28; games behind conference leader +1.10 | strength of schedule (opponents' SP+) -1.23; number of FBS opponents played -0.51; wins over SP+ top-25 teams -0.39 |
| 2 | Georgia | games behind conference leader +1.10; average margin of victory (capped at 28) +1.10; in conference-title contention +0.92 | number of FBS opponents played -0.51; weeks since last loss -0.47; wins over SP+ top-25 teams -0.39 |
| 3 | Utah | weeks since last loss +2.89; games behind conference leader +1.10; average margin of victory (capped at 28) +0.99 | strength of schedule (opponents' SP+) -1.19; number of FBS opponents played -0.51; wins over SP+ top-25 teams -0.39 |
| 4 | Indiana | wins over SP+ top-25 teams +2.04; in conference-title contention +0.92; number of FBS opponents played +0.77 | offensive explosiveness -0.69; weeks since last loss -0.47; offensive success rate -0.40 |
| 5 | Alabama | strength of schedule (opponents' SP+) +1.30; in conference-title contention +0.92; number of FBS opponents played +0.77 | wins over SP+ top-25 teams -2.81; weeks since last loss -0.47; offensive explosiveness -0.43 |
| 6 | Kentucky | weeks since last loss +1.63; games behind conference leader +1.10; strength of schedule (opponents' SP+) +0.95 | average margin of victory (capped at 28) -0.88; number of FBS opponents played -0.51; wins over SP+ top-25 teams -0.39 |
| 7 | Texas | wins over SP+ top-25 teams +4.46; strength of schedule (opponents' SP+) +1.60; number of FBS opponents played +0.77 | in conference-title contention -0.99; offensive success rate -0.63; games behind conference leader -0.49 |
| 8 | UCLA | weeks since last loss +2.89; games behind conference leader +1.10; in conference-title contention +0.92 | strength of schedule (opponents' SP+) -1.46; wins over SP+ top-25 teams -0.39; bad losses -0.26 |
| 9 | BYU | wins over SP+ top-25 teams +2.04; in conference-title contention +0.92; bad losses +0.67 | weeks since last loss -0.89; number of FBS opponents played -0.51; average margin of victory (capped at 28) -0.46 |
| 10 | Ole Miss | wins over SP+ top-25 teams +4.46; in conference-title contention +0.92; number of FBS opponents played +0.77 | weeks since last loss -0.89; best win (SP+ rank of best team beaten) -0.31; bad losses -0.26 |
| 11 | Notre Dame | independent status +2.36; average margin of victory (capped at 28) +1.16; games behind conference leader +1.10 | wins over SP+ top-25 teams -2.81; strength of schedule (opponents' SP+) -1.50; in conference-title contention -0.99 |
| 12 | Ohio State | wins over SP+ top-25 teams +2.04; in conference-title contention +0.92; number of FBS opponents played +0.77 | weeks since last loss -1.31; strength of schedule (opponents' SP+) -0.34; bad losses -0.26 |
| 13 | Tennessee | offensive explosiveness +0.51; strength of schedule (opponents' SP+) +0.16; number of Power 4 opponents played +0.14 | in conference-title contention -0.99; weeks since last loss -0.89; number of FBS opponents played -0.51 |
| 14 | LSU | average margin of victory (capped at 28) +1.03; offensive success rate +0.51; strength of schedule (opponents' SP+) +0.39 | wins over SP+ top-25 teams -2.81; in conference-title contention -0.99; number of FBS opponents played -0.51 |
| 15 | Texas Tech | wins over SP+ top-25 teams +2.04; in conference-title contention +0.92; bad losses +0.67 | weeks since last loss -0.89; strength of schedule (opponents' SP+) -0.63; number of FBS opponents played -0.51 |
| 16 | Wake Forest | weeks since last loss +1.21; in conference-title contention +0.92; number of FBS opponents played +0.77 | strength of schedule (opponents' SP+) -1.55; offensive explosiveness -0.49; wins over SP+ top-25 teams -0.39 |
| 17 | Houston | strength of schedule (opponents' SP+) +0.94; offensive explosiveness +0.50; offensive success rate +0.49 | games behind conference leader -1.02; in conference-title contention -0.99; weeks since last loss -0.89 |
| 18 | Missouri | bad losses +0.67; best win (SP+ rank of best team beaten) +0.26; number of Power 4 opponents played +0.14 | wins over SP+ top-25 teams -2.81; in conference-title contention -0.99; offensive success rate -0.69 |
| 19 | Iowa | bad losses +0.67; offensive explosiveness +0.19; best win (SP+ rank of best team beaten) +0.18 | wins over SP+ top-25 teams -2.81; games behind conference leader -1.02; in conference-title contention -0.99 |
| 20 | USC | number of FBS opponents played +2.05; offensive success rate +0.29; strength of schedule (opponents' SP+) +0.28 | in conference-title contention -0.99; average margin of victory (capped at 28) -0.98; weeks since last loss -0.89 |
| 21 | South Carolina | bad losses +0.67; strength of schedule (opponents' SP+) +0.60; weeks since last loss +0.37 | games behind conference leader -1.02; in conference-title contention -0.99; average margin of victory (capped at 28) -0.92 |
| 22 | Arizona | best win (SP+ rank of best team beaten) +0.63; average margin of victory (capped at 28) +0.36; defensive success rate allowed +0.10 | wins over SP+ top-25 teams -2.81; strength of schedule (opponents' SP+) -1.44; weeks since last loss -1.31 |
| 23 | Oregon | wins over SP+ top-25 teams +4.46; strength of schedule (opponents' SP+) +1.84; bad losses +0.67 | average margin of victory (capped at 28) -1.34; games behind conference leader -1.02; in conference-title contention -0.99 |
| 24 | Florida | wins over SP+ top-25 teams +2.04; strength of schedule (opponents' SP+) +1.94; number of Power 4 opponents played +0.14 | average margin of victory (capped at 28) -1.69; weeks since last loss -1.31; games behind conference leader -1.02 |
| 25 | Boise State | weeks since last loss +2.05; games behind conference leader +1.10; in conference-title contention +0.92 | wins over SP+ top-25 teams -2.81; number of Power 4 opponents played -2.68; strength of schedule (opponents' SP+) -1.41 |

## What separates neighbors (Top 12)

- **#1 Miami over #2 Georgia**: Miami ahead on weeks since last loss +3.36, best win (SP+ rank of best team beaten) +0.37; Georgia ahead on strength of schedule (opponents' SP+) +1.07, bad losses +0.93
- **#2 Georgia over #3 Utah**: Georgia ahead on strength of schedule (opponents' SP+) +1.03, bad losses +0.93; Utah ahead on weeks since last loss +3.36, best win (SP+ rank of best team beaten) +0.31 (The factors alone score them the other way.)
- **#3 Utah over #4 Indiana**: Utah ahead on weeks since last loss +3.36, offensive success rate +0.92; Indiana ahead on wins over SP+ top-25 teams +2.42, strength of schedule (opponents' SP+) +1.57
- **#4 Indiana over #5 Alabama**: Indiana ahead on wins over SP+ top-25 teams +4.85, wins over teams above .500 +0.54; Alabama ahead on strength of schedule (opponents' SP+) +0.93, number of Power 4 opponents played +0.56
- **#5 Alabama over #6 Kentucky**: Alabama ahead on number of FBS opponents played +1.28, average margin of victory (capped at 28) +1.20; Kentucky ahead on wins over SP+ top-25 teams +2.42, weeks since last loss +2.10. Head-to-head: Alabama won. (The factors alone score them the other way.)
- **#6 Kentucky over #7 Texas**: Kentucky ahead on weeks since last loss +2.10, in conference-title contention +1.91; Texas ahead on wins over SP+ top-25 teams +4.85, number of FBS opponents played +1.28 (The factors alone score them the other way.)
- **#7 Texas over #8 UCLA**: Texas ahead on wins over SP+ top-25 teams +4.85, strength of schedule (opponents' SP+) +3.06; UCLA ahead on weeks since last loss +3.36, in conference-title contention +1.91 (The factors alone score them the other way.)
- **#8 UCLA over #9 BYU**: UCLA ahead on weeks since last loss +3.78, number of FBS opponents played +1.28; BYU ahead on wins over SP+ top-25 teams +2.42, strength of schedule (opponents' SP+) +1.75
- **#9 BYU over #10 Ole Miss**: BYU ahead on bad losses +0.93, defensive success rate allowed +0.16; Ole Miss ahead on wins over SP+ top-25 teams +2.42, number of FBS opponents played +1.28 (The factors alone score them the other way.)
- **#10 Ole Miss over #11 Notre Dame**: Ole Miss ahead on wins over SP+ top-25 teams +7.27, strength of schedule (opponents' SP+) +2.08; Notre Dame ahead on independent status +2.46, average margin of victory (capped at 28) +1.38
- **#11 Notre Dame over #12 Ohio State**: Notre Dame ahead on independent status +2.46, games behind conference leader +1.06; Ohio State ahead on wins over SP+ top-25 teams +4.85, in conference-title contention +1.91 (The factors alone score them the other way.)

## Teams the factors cannot explain

Committee and explainer differ by 5+ places (positive = the committee ranked the team higher than the factors suggest):

- Arizona: committee 22, explainer 66 (+44)
- Missouri: committee 18, explainer 39 (+21)
- Iowa: committee 19, explainer 36 (+17)
- Oregon: committee 23, explainer 11 (-12)
- Tennessee: committee 13, explainer 24 (+11)
- Alabama: committee 5, explainer 15 (+10)
- Ole Miss: committee 10, explainer 1 (-9)
- Louisville: committee NR, explainer 19 (-7)
- Florida: committee 24, explainer 18 (-6)
- UCLA: committee 8, explainer 2 (-6)
- Boise State: committee 25, explainer 20 (-5)
- Georgia: committee 2, explainer 7 (+5)
- Houston: committee 17, explainer 22 (+5)
- Texas Tech: committee 15, explainer 10 (-5)

## Biggest movers vs last week

No earlier committee release is in the data yet.

## Caveats

- Each week has only 25 ranked teams; weights are noisy, so rely on the 90% bands and on the pooled fit (`kind=pooled` in the weights CSV) as more weeks arrive.
- Collinear stats (dropped by VIF) cannot be separated from the kept stats that proxy them.
- Fit metrics are in-sample. Head-to-head results are shown as facts; they have no fitted weight.
