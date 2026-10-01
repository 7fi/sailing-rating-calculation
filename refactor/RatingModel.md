# How the rating model works

This document explains the rating system in `refactor/whr.py` and `refactor/whrTR.py`:
what it computes, why it is built this way, and which parts of what the site displays
are model output versus presentation.

It assumes you know the previous system — the sequential openskill filter — and uses it
as the point of comparison throughout.

Every number quoted as "measured" was computed against the real race history
(~1.5M score rows, 2013–2026). Where a number comes from one rating type only, that
type is named; `sr` is open skipper, which is the largest graph and the one most
measurements were taken on.

---

## 1. The one difference everything follows from

The old system is a **filter**. It walks races in chronological order. Each race reads
the current `(mu, sigma)` of its competitors, applies a Plackett-Luce update, and writes
back new values. Once a race has been processed its effect is baked into the state and
never reconsidered.

The new system is a **fit**. No race is "processed". Instead the model declares one
unknown number per sailor per regatta — call the whole collection `theta` — and asks a
single question:

> Which assignment of `theta` makes the entire observed race history most probable,
> subject to skill changing slowly over time?

and then solves that optimisation problem. There is no "before" or "after" a race.
Every race in the history is a simultaneous constraint on one answer.

Two consequences worth stating immediately, because they explain most of the surprises:

- **The fit revises history.** If a sailor you beat in 2024 turns out to be excellent in
  2026, the fit raises what that 2024 win was worth, and your 2024 rating moves. A
  filter cannot do this; a fit cannot avoid it.
- **There is no "starting rating".** A filter needs an initial value to update away
  from. A fit has no sequence and therefore no starting point. This is the main thing
  the new model gives up, and section 5 explains why it is unavoidable rather than a
  design choice.

---

## 2. The model

### 2.1 Nodes and theta

A **node** is one `(sailor, time bucket)` pair, and it holds exactly one number:
`theta`, a strength on a log-odds scale. Not a mean and a variance — just one number.
Uncertainty is computed separately and afterwards (section 6); it is not part of the
rating and does not feed back into it.

`timeResolution` sets what a bucket is, and therefore how finely the fitted career curve
can bend:

| resolution | bucket | nodes (`sr`) |
|---|---|---|
| `season` | one per sailor per season | 35,141 |
| `regatta` | one per sailor per regatta | 76,183 |
| `race` | one per sailor per rated field | 616,950 (= one per row) |

The pipeline uses `regatta` for the published curve and `season` for the uncertainty
fit. `race` is available and was measured; it is worse on both axes (section 9.2).

### 2.2 The likelihood: Plackett-Luce over whole fields

This is the same model family openskill uses, so the shape will be familiar. For one
rated field — a single race, every boat in it at once — finishing in order
`i_1, i_2, ..., i_n`:

```
log P(this finishing order) = sum over k=1..n-1 of
        theta_{i_k} - log( sum over m>=k of exp(theta_{i_m}) )
```

Read it left to right as a sequence of conditional choices: the winner had to be the
best of all `n` boats; given that, second place had to be the best of the remaining
`n-1`; and so on. The denominator is the "risk set" — everyone still unplaced at that
point.

The crucial property is that this is the **exact** likelihood of a whole field, not a
decomposition into pairs. A 20-boat race is one 20-way constraint, not 190 pairwise ones.

The difference from the old system is **not** the likelihood. It is that openskill
applies this incrementally, with a Gaussian approximation to the posterior at each step,
while here the exact expression is summed over all 45,765 fields (`sr`) and maximised
jointly.

Implementation: `_plObjective` in `whr.py`. Fields are bucketed by competitor count so
each bucket is one dense `(nFields, n)` array, which makes the whole objective and its
gradient a handful of vectorised numpy operations instead of a Python loop over races.
Per field it is O(n) via one suffix sum and one cumulative sum. `exp` is taken on
`theta` shifted by the per-field maximum for numerical stability, which is exact because
Plackett-Luce is invariant to adding a constant within a field.

### 2.3 There is no beta

In the old system `beta` controls how a rating gap translates into a win probability,
and you have to choose it. Here there is no such parameter. The scale **is** the spread
of `theta`, and the spread of `theta` is fitted from the data. So:

```
P(sailor i beats sailor j) = sigmoid(theta_i - theta_j)
```

holds exactly, by construction, with nothing to tune.

This matters beyond convenience. It is why the model is calibrated by default, and it is
also why every attempt to measure region offsets *on top of openskill* was unreliable:
fitting a correction onto a base model whose held-out log-loss is 0.94 absorbs the base
model's miscalibration into the correction, and the fitted offsets swung with `beta`.
Two independent implementations of that measurement disagreed on sign for MCSA.

### 2.4 The prior: a random walk in time

The likelihood alone is not enough — it has no opinion about time, and it does not
determine a unique answer. Two prior terms fix both problems.

**Wiener (random walk) between a sailor's consecutive nodes:**

```
theta[next] - theta[this]  ~  Normal(0, w^2 * dt)
```

where `dt` is **elapsed days** at regatta and race resolution (and the season index at
season resolution), floored at `dtFloorDays = 0.1` so simultaneous nodes cannot produce
an infinite precision.

This is the term that replaces `sigma`. It says skill drifts gradually, so a sailor's
rating at one regatta is informed by their neighbouring regattas, with strength
depending on how far apart in time they are. A one-week gap couples two nodes tightly; a
summer couples them loosely. That is a genuine improvement over `sigma`, which only
tracked *how many* races you had sailed and not *when*.

`w` (`whrWCurve = 0.055`) is the single knob on how lively the curve is, and it was
chosen by held-out predictive likelihood, not by eye. See section 9.1.

**Initial prior on each sailor's first node:**

```
theta[first]  ~  Normal(0, sigma0^2)
```

with `sigma0 = 3.0`. This does one essential job: Plackett-Luce is invariant to adding
the same constant to every sailor on earth, so without this term the problem has
infinitely many solutions. Pinning first appearances to a zero-centred prior removes
that freedom. It also makes **disconnected components well posed** — a cluster of
sailors who never met anyone else still gets a finite answer, centred on the prior,
rather than floating off.

A warning recorded in `config.py` and worth repeating: **do not tighten `sigma0` to
suppress thin records.** Shrinkage toward the prior bites hardest on sailors with the
least data, and race volume correlates with region (PCCSC sailors average 13 races and 2
regattas a season against NEISA's 9 and 1). Measured, dropping `sigma0` from 3.0 to 0.75
cut NEISA+MAISA's share of the top 100 from 78 to 72 and *raised* the region-offset
spread from 0.18 to 0.38. Thin records are handled by the SE gate instead, which keys
off how well a record actually pins a sailor down rather than off how much they raced.

### 2.5 Why the answer is unique

The Plackett-Luce log-likelihood is concave in `theta`. The two prior terms are negative
quadratics. So the objective (`negLogPost`) is **strictly convex**: one global optimum,
no local minima, no sensitivity to initialisation, and no "run it again and see if it
settles somewhere else". This is a real robustness property that the sequential filter
did not have — there, the answer depended on the order races were swept and on where
each sailor started.

---

## 3. Why a joint fit, and not a better filter

This is the heart of it, and it is worth being precise because the conclusion is
*structural*, not a tuning failure.

### 3.1 Mean conservation

A Plackett-Luce update redistributes strength among the boats in the rated field. The
sum of `mu` over the field is approximately conserved: points move *between*
competitors, and the field's total barely changes.

Now consider a PCCSC regatta: twenty local sailors and two NEISA visitors. Whatever the
result, almost all of the redistribution happens among the twenty locals, because
they are who the points are being traded with. The two visitors carry away a tiny
fraction of the level information. Repeat this across a season and each conference
**self-normalises to whatever mean it started at**. Mean conservation is not neutral
here — it actively opposes the propagation you need.

Measured, by simulation: two pools with a genuine 760-point difference in true skill,
rated by the sequential filter, land **5 points apart**. Adding 200 cross-pool races
recovers only 30% of the real gap.

Re-sweeping the history (iterating the filter to convergence) re-applies the same weak
coupling and does not help — measured, it does not. Four separate tuning attempts failed
to fix this: adjusting `beta`, iterated replay, a `sigma` floor, and time-varying `tau`.

### 3.2 What the joint fit does instead

A joint fit has no "redistribute" step, so it has nothing to conserve. Those two
visitors' results are hard constraints tying the two regions' `theta` together, and the
optimiser must satisfy them **simultaneously** with every other constraint in the
system. Information does not decay with distance through the comparison graph; it
propagates until the whole thing is consistent.

### 3.3 Measured result

Whole-regatta holdout — entire regattas removed from the fit, then predicted:

| | openskill | joint fit |
|---|---|---|
| log-loss | 0.9405 | **0.5259** (−44%) |
| cross-region log-loss | 1.0374 | **0.5250** (−49%) |
| pairwise accuracy | 0.7217 | **0.7311** |
| regional bias, max abs z | 22.15 | **6.28** |

Model-free check (actual cross-region regatta results, no model involved): residual
spread across conferences 0.2296 → 0.0599.

One finding worth keeping in mind because it rules out a whole class of simpler fixes:
PCCSC's 90th percentile fell 283 points (2396 → 2113) while its median barely moved. The
inflation was concentrated in the elite tail, which **no single additive per-conference
offset could ever have corrected**. That is why the "fit 6 region offsets" approach was
abandoned in favour of changing the model.

---

## 4. Solving it

`fitWHR` runs L-BFGS-B on the convex objective. Notes that matter:

- **The tolerances are deliberately much tighter than scipy's defaults.** At the default
  `ftol=2.2e-9` L-BFGS stops after roughly 1,400 of the ~2,500–3,700 iterations this
  problem needs, leaving individual ratings up to 0.086 sd — about 35 display points —
  away from the optimum. At `ftol=1e-11, gtol=1e-7` the worst node is within 0.004 sd
  (~1.5 points). If you ever see ratings that look "almost right but jittery between
  runs", check this first.
- **`freeIdx`** optimises a subset of nodes and holds the rest fixed. Settled history
  barely moves when new races arrive, so freezing it shrinks the live parameter count
  enormously.
- **`warmStart`** carries a previous fit's `theta` onto a rebuilt node set. Node indices
  are positional and shift whenever races are added, so the mapping goes through stable
  text labels (`"<sailorID>" + NODE_SEP + "<bucket>"`). Nodes that are new to this fit
  start at that sailor's most recent known value. Measured: `sr` refit 53.7s cold →
  10.4s warm.

### 4.1 A note on NODE_SEP

Node labels use `\x1f` (ASCII unit separator), not `\x00`. This is not cosmetic:
**pandas silently drops `\x00` from string concatenation**, which left every fleet node
label as a bare concatenation (`"A A-Hamilton" + "s14"` → `"A A-Hamiltons14"`). Nothing
was ever mis-rated by this — checked explicitly, zero collisions between distinct
`(sailor, bucket)` pairs at both season and regatta resolution — but `warmStart` splits
the label to recover the sailor ID, so its "fall back to this sailor's latest theta"
path never fired and new nodes always cold-started at zero. `whrTR` built its labels
with Python f-strings, where `\x00` survives, so it was unaffected.

---

## 5. From theta to a published number, and why there is no starting rating

`theta` is on a log-odds scale centred near zero. `anchorScale` maps it to display
points with an affine transform:

```
display = theta * scale + offset
```

chosen so that a reference population has mean `whrTargetMean` (1400) and standard
deviation `whrTargetSd` (400).

### 5.1 Why the scale has to be imposed from outside

The likelihood constrains **differences** of `theta`, and the prior pins the overall
centre, but nothing in the model fixes the *units*. Ranking is completely invariant to
`scale` and `offset`: multiplying every `theta` by 1.5 changes no ordering and no
predicted probability ordering. So the display scale is genuinely a free choice, and the
model cannot make it for you.

Two consequences:

- **`whrTargetSd` is free.** Setting it to 600 instead of 400 widens every published
  rating and every career span by 50% at *exactly zero* accuracy cost. If you want
  livelier-looking graphs, this is the honest lever, not `wCurve`.
- **There is no common starting rating, and there cannot be.** The old system gave every
  sailor exactly 1000 to begin with because a filter needs something to update away
  from. A fit has no sequence, so a sailor's earliest node is simply wherever the
  evidence and the `sigma0` prior put it. Measured: 1,697 distinct first-race values
  against openskill's single 1000. This is the one requirement the new model does not
  meet.

### 5.2 Re-anchoring every run

The anchor is recomputed each run on the current population. Without that, the whole
distribution drifts slightly between runs and every sailor's number moves even when
nobody's standing changed.

**The anchor population must be the same for every fit of a rating type.** This caused a
real, visible bug. The two fits (section 7) used different anchor populations:

- the curve fit anchored on every `(sailor, regatta)` node in the target seasons
- the level fit anchored on one node per sailor

Both force their own population to mean 1400, but those populations differ in
composition: the first weights each sailor by how many regattas they sailed, and heavy
racers are stronger. Measured on `sr`, target seasons `s26`/`f26`:

| population | n | mean |
|---|---|---|
| all `(sailor, regatta)` nodes | 3,186 | 1400.0 (by construction) |
| one node per sailor, latest | 1,485 | **1320.9** |
| sailors with 1 regatta | 704 | 1220.8 |
| sailors with 4+ regattas | 239 | 1509.6 |

The 79.1-point difference between the first two rows became a pure offset between the
two fits: **the published leaderboard rating sat 79 points above the end of the sailor's
own fitted curve.** Rankings were unaffected (spearman 0.9974), but the big number on a
sailor's page disagreed with where their graph ended.

Fixed by `currentNodesBySailor`, which both fits now use: one node per sailor, their
most recent. Measured after the fix, leaderboard minus curve end: mean **−0.00**
(was +79.05), sd 28.2 (was 32.6).

The residual sd of 28 points is not a bug — it is the genuine disagreement between two
different models of the same sailor (regatta resolution at `w=0.055` versus season
resolution at `w=0.6`). If you want them to agree exactly, the only way is to publish
the curve's endpoint as the rating, which is a separate decision.

A related trap, recorded in `fitWithSE`: **do not anchor on the ranked subset.** The
eligibility threshold is expressed in display points, so a larger scale inflates every
SE, which shrinks the ranked set, which shrinks the `theta` spread, which inflates the
scale again. That feedback does not settle — it cut the women's skipper list from 203 to
73 on its own. Anchor on a fixed population and express the threshold relative to the
scale instead (`maxRatingSEFraction`).

---

## 6. Uncertainty

The rating itself carries no uncertainty, so it is computed separately, from the
curvature of the objective at the optimum.

### 6.1 The Laplace approximation

At the MAP optimum, approximate the posterior as Gaussian with precision equal to the
Hessian of the negative log posterior. `buildHessian` assembles it sparsely:

- Plackett-Luce contributes, per field, `sum over k of [ diag(p_.k) - p_.k p_.k^T ]`
  over the risk sets, with `p_mk = exp(theta_m) / S_k`. This is positive semi-definite.
- The Wiener prior adds a tridiagonal block per sailor.
- The initial prior adds to the diagonal.

Together these make the matrix **positive definite even where the comparison graph is
disconnected**, which is what lets every sailor get a finite SE.

### 6.2 What quantity is actually reported

Not the marginal variance of a single `theta`, which would be dominated by the
unidentifiable global constant. Instead the SE of a **contrast**:

```
se = sqrt( v' H^-1 v )   where v = e_sailor - referenceWeights
```

with `referenceWeights` the race-count-weighted ranked population. In words: *how well
do we know where this sailor sits relative to the national field?* That is the quantity
`outLinks` was a crude proxy for.

`contrastSE` solves `H x = v` by preconditioned conjugate gradient rather than forming
`H^-1`, because the rankings only need one solve per sailor of interest, not one per
node. `fitWithSE` uses a sparse LU (`splu`) and solves in blocks of 400 columns, which
is faster when thousands of sailors are needed at once.

### 6.3 Why SEs come from a season-resolution fit

Two reasons. The quantity of interest is a sailor's *current level*, not their rating at
one specific regatta. And the season-resolution Hessian factorises in about a minute,
where the regatta-resolution one is roughly 20× larger and risks a very large LU factor.

This is the dominant cost in the pipeline: SE computation is 61% of a 656-second run
(`cr` 258.7s, `sr` 139.7s). Peak RSS is 3.3GB of 16GB and it is single-threaded, so it
is CPU-bound, not memory-bound.

### 6.4 The per-regatta noise term

A sailor's races within one regatta share a weekend, a venue, a boat and a partner, so
they are not independent samples of skill — but the likelihood treats them as if they
were. `whrRegattaNoiseFraction` adds the missing term:

```
se_total^2 = se_model^2 + (frac * targetSd)^2 / nRegattas
```

Measured regatta-to-regatta scatter of a sailor's own rating is about 68 points on a
400-point spread, i.e. 0.17 sd, which is where the suggested value comes from.

**It is currently set to 0.0** — off. It was reverted because it is effectively a volume
rule, and at values large enough to matter it put USC at #1. If you turn it on, re-check
the regional mix of the top 30.

### 6.5 The gate, not a subtraction

Eligibility is `se < maxRatingSEFraction * whrTargetSd` = `0.25 * 400` = **100 points**.
Expressing it as a fraction means changing the display scale cannot silently change who
qualifies.

This replaced `requiredOutLinks`, which measured connectivity backwards: PCCSC had the
*lowest* pass rate (10.5%) and simultaneously the *lowest* median SE — the
best-determined conference by the model's own reckoning. At the 100-point gate, 79% of
sailors are rankable against 24% under `outLinks`, and PCCSC goes from worst-served
(10.2%) to best (86.7%).

**The leaderboard publishes `rating` and ranks on `rating`**, so the displayed order
always agrees with the displayed number. Ranking on `rating - 1.96*se` while publishing
`rating` puts a sailor showing 1900 below one showing 1850, which is precisely the
objection `RatingSystem.md` raises. Measured: with the gate active the two order almost
identically (spearman 0.9964, mean 20 ranks of movement) versus 45 ranks and a maximum
of 684 with no gate. **The gate does the work, not the subtraction.**

---

## 7. Shrinkage, and exactly where it applies

### 7.1 The single-level version

Empirical-Bayes shrinkage toward the population mean, by reliability:

```
shrunk = m + (rating - m) * tau2 / (tau2 + se^2)
tau2   = max( var(rating) - mean(se^2), 1e-6 )
```

`tau2` is the method-of-moments estimate of the true between-sailor variance: observed
spread minus the part that is measurement noise. There is **no free constant**. This is
the posterior mean under a normal prior, so a sailor with one regatta regresses most of
the way to average while a well-measured sailor keeps their rating.

### 7.2 The two-level version (`_hierarchicalShrink`)

Shrinking every sailor independently toward the *global* mean systematically favours a
roster that concentrates its sailing in a few heavy racers over one that spreads it
across many, because each thin record is dragged to average in isolation. Nesting the
shrinkage inside the school fixes that:

```
se_t^2   = 1 / sum_i (1/se_i^2)                 precision of the team's evidence
m_t      = sum_i (rating_i / se_i^2) * se_t^2   precision-weighted team mean
tau2_T   = var(m_t) - mean(se_t^2)              between-team variance
m_t'     = m + (m_t - m) * tau2_T/(tau2_T + se_t^2)
tau2_S   = var(rating_i - m_t) - mean(se_i^2)   within-team variance
shrunk_i = m_t' + (rating_i - m_t') * tau2_S/(tau2_S + se_i^2)
```

Six sailors with one regatta each pool into one reasonably-determined team level, which
is then itself shrunk toward global by how much the school has actually proved. Every
quantity is method-of-moments; again no free constant.

Measured against the single-level version: Michigan 26 → 33, Boston University 33 → 31,
Charleston 4 → 7, Harvard 6 → 8, Yale 7 → 5. Top-30 conference mix went from
`{NEISA 13, MAISA 8, SAISA 4, PCCSC 3, SEISA 1, MCSA 1}` to
`{NEISA 13, MAISA 8, SAISA 5, PCCSC 3, SEISA 1}` — MCSA out, NEISA unchanged.

It is **volume-neutral at team level**, which is what every per-sailor rule was not.

### 7.3 Where each statistic is actually used

This is easy to get backwards, so:

| what | statistic | config key |
|---|---|---|
| sailor leaderboard order | `rating` (raw, anchored) | `rankingStatistic` |
| sailor number on the site | `rating` | `publishedStatistic` |
| team **fleet** rating | sum of top-N `shrunk` | `teamRatingStatistic` |
| team **race** rating | sum of top-N `rating` | `teamRatingStatisticTR` |

So the hierarchical shrinkage affects **team fleet ratings only**. Individual sailors are
published and ranked on the raw anchored rating, with the SE used purely as a gate.

Team racing **must** use the unshrunk rating. A 3v3 match only ever observes the combined
strength of three sailors, so an individual `theta` has SE about 1.00 sd while the mean
of three is about 0.36 sd — the aggregate is identified even though the split is not.
Because shrinkage is computed per sailor and for TR `var(rating) < mean(se^2)`, `tau2`
clamps to its floor and the shrink factor comes out around 4e-12: every sailor collapses
onto the population mean and every team published exactly 1400. Shrinking individuals
first destroys precisely the information the top-N sum would have recovered.

Two deliberate differences in `Teams.getOrderedSailors`, both measured:

- **No SE gate on team ratings.** Applying it made cross-region bias worse (PCCSC +0.112
  versus +0.084) because it strips a team's weaker sailors and leaves only their best.
  The gate decides who is confidently *ranked*, not how strong a team is.
- **A missing top-N slot counts as the population mean**, not zero and not absent.
  Dividing by `numTops` regardless scored a thin roster as if its third sailor rated 0;
  dividing by however many were found rewarded thin rosters instead.

---

## 8. The per-race numbers, and what is real about them

This is the part where model and presentation diverge most, so it is worth reading
carefully before relying on these columns.

A joint fit produces a **curve over regattas**, not a sequence of per-race updates. The
site, inherited from the old system, displays a rating change per race. Reconciling the
two takes two distinct quantities, and `raceLadder` emits both.

### 8.1 `influence` — real, and a property of the model

```
influence = (d logL_race / d theta_n) / H_nn
```

At the optimum the total gradient is zero, so every individual race's likelihood
gradient is exactly balanced by the prior and by the other races. That gradient
therefore measures which way this one race is pulling this sailor's rating, and dividing
by the local curvature converts it into rating units.

This is the standard influence-function approximation to **"how far would this sailor's
fitted rating move if this race were dropped from the fit?"** It is the joint-fit
analogue of the sequential per-race delta, and unlike that delta it accounts for all the
information in the fit rather than only what came before. It is a real, defensible,
per-race quantity.

Measured on `sr`: sd 12.5 points, mean absolute 6.1 points.

### 8.2 `credit` — what the displayed ladder steps by

`influence` does **not** sum to a sailor's actual movement between regattas, and the gap
is large. Measured over 62,860 consecutive-regatta pairs on `sr`:

| quantity | sd |
|---|---|
| real move between adjacent regatta nodes | **61.5** pts |
| sum of `influence` within the regatta | **17.6** pts |

with correlation 0.585 and mean absolute disagreement **29.2 points**.

The gap is real movement that no single race can be credited with. It comes from the
Wiener prior pulling this node toward the sailor's neighbouring regattas, and from every
other result in the system shifting the field around them. A sailor can genuinely gain
rating at a regatta partly because people they beat last month did well this month.

The earlier version of `raceLadder` ignored this and opened each regatta at
`nodeRating - sum(influence)`, which **silently teleported the displayed line by those
29 points at every regatta boundary**. The graph looked continuous and was not.

It now distributes the remainder evenly across the regatta's races:

```
step     = nodeRating - previous nodeRating        (real, from the fit)
credit_i = influence_i + (step - sum(influence)) / nRaces
```

so that all three of these hold exactly (verified to 2e-13 on all 616,950 `sr` rows):

1. `newRating - oldRating == credit` for every race
2. the last race of a regatta lands exactly on that regatta's fitted rating
3. the first race of a regatta opens exactly where the previous regatta closed

A sailor's first regatta has no previous node, so it keeps the old convention of opening
at `nodeRating - sum(influence)`.

Measured after the change on `sr`: `credit` sd 14.7, mean absolute 7.7. `credit` and
`influence` correlate 0.883, and **in 12.9% of races they disagree in sign** — the
between-regatta drift is larger than the race's own influence, so the displayed step
goes one way while the race itself pushed the other.

### 8.3 What this means for what you display

Honest readings of each column:

- **The curve across regattas is fully real.** Every point on it is a fitted parameter.
- **`influence` is fully real** as "what this race did for my rating".
- **`credit` is real in total but partly unattributable in detail.** The sum over a
  regatta is exactly the sailor's real movement; the split across races within the
  regatta is `influence` plus an equal share of drift that belongs to no single race.
- **Within-regatta resolution is presentational.** All races in one regatta share one
  fitted `theta`. The intermediate points on the line inside a regatta are an
  attribution of real total movement, not independently fitted skill.

If you want the graph to be strictly model output and nothing else, plot one point per
regatta. If you want per-race movement, `credit` is now a correct decomposition of a
real total — which is a much better position than before this change, when the running
total did not close at all.

---

## 9. How the hyperparameters were chosen

### 9.1 `wCurve`

Swept `{0.02, 0.055, 0.12, 0.25, 0.5}` against held-out whole regattas:

| w | heldout nll/field | heldout acc | span (pts) | net/span |
|---|---|---|---|---|
| 0.020 | 20.3289 | 0.7194 | 46 | 0.95 |
| **0.055** | **20.1793** | **0.7233** | 129 | 0.78 |
| 0.120 | 20.4538 | 0.7177 | 221 | 0.62 |
| 0.250 | 20.8730 | 0.7080 | 345 | 0.61 |
| 0.500 | 21.5108 | 0.6933 | 482 | 0.65 |

0.055 is the accuracy optimum. A paired 5-seed comparison confirms `w=0.12` loses 5/5
seeds by 0.49pp and `w=0.25` by 1.44pp. **Liveliness is not free** — larger `w` gives
bigger career swings and measurably worse predictions.

For context on curve shape: openskill's median career span was 615 points but 82% of it
was one-way drift from the `sigma` artifact (`net/span` 0.82+). The joint fit's span is
smaller but `net/span` is 0.46–0.78, i.e. an actual trajectory rather than a ramp.

### 9.2 `timeResolution`

`race` resolution is worse on **both** axes — lower accuracy *and* smaller spans:

| w | race-res nll/field | race-res acc | span |
|---|---|---|---|
| 0.055 | 21.0533 | 0.7130 | 61 |
| 0.120 | 20.4449 | 0.7189 | 119 |

7× the parameters against the same data means heavier shrinkage, and the prior gains
many short-`dt` edges that pin consecutive races together. Two independent
investigations measured this.

### 9.3 `sigma0`

Chosen on overall held-out log-loss (tied best at 2.0) with cross-region offset spread
as the tie-break, giving 3.0. See the warning in section 2.4 about tightening it.

### 9.4 Rules that were tried and reverted

All of these optimised a proxy metric while making the published board worse. Recorded
so they are not re-attempted blindly:

| rule | why reverted |
|---|---|
| `minRegattasRanked = 2` | volume rule; cut NEISA+MAISA from 72 to 57 of eligible top 100, and PCCSC ended up with the *largest* eligible pool |
| `whrRegattaNoiseFraction` large enough to matter | put USC at #1 |
| per-team-regatta noise | barely moved anything |
| WHR region offsets | 41-point spread, i.e. nothing left to correct |
| ranking on `lcb` | fixes Michigan (→35) but moves USC the wrong way (30→25), because USC has the smallest team SE on the board; also rises as a sailor races more at constant skill, reintroducing the participation reward that made `mu - 3*sigma` unusable |
| tighter `sigma0` | shifts the board west (section 2.4) |

---

## 10. Team racing

A different likelihood, the same machinery. Team racing is 3-on-3 with a win/loss
outcome, so Bradley-Terry on the combined strength of each side:

```
P(A beats B) = sigmoid( sum(theta over A's three sailors) - sum(theta over B's) )
```

applied separately to skippers and crews, matching how the old pass treated a team as
the three same-position sailors. Ties (4 rows in the entire history) enter as `y = 0.5`,
which the Bernoulli likelihood handles directly.

Same Wiener prior between a sailor's consecutive regattas, same node construction, same
warm start, same Hessian/SE machinery, same ladder structure (including the continuity
fix in section 8.2). On a like-for-like whole-regatta holdout it beats openskill on
every team-race type — `tsr` accuracy 0.696 versus 0.625.

**The identifiability caveat is important.** A 3v3 match only observes a *sum*. The
individual split is barely identified: individual SE about 1.00 sd against 0.36 sd for
the mean of three. This is why team-race team ratings use the unshrunk rating
(section 7.3), and why individual team-race ratings should be read with much more
caution than fleet ratings.

Before this was implemented, team racing had no per-match ladder at all: every sailor
showed a constant 1400 in every match, because the openskill pass was skipped and
nothing filled the columns. After: 79,492 distinct values where there had been one, and
100% of rows move.

---

## 11. Pipeline shape

`runFleetPipeline` runs **two fits per fleet rating type**, because they answer different
questions:

| fit | resolution | `w` | produces |
|---|---|---|---|
| curve | regatta | `whrWCurve = 0.055` | published curve, `influence`, `credit`, the ladder |
| level | season | `whrWLevel = 0.6` | posterior SE, eligibility, `shrunk` |

Four fleet types (`sr`, `cr`, `wsr`, `wcr`) plus four team-race types
(`tsr`, `tcr`, `wtsr`, `wtcr`), each an independent graph, fitted separately.

Outputs:

- **`whr_races.parquet`** — per race row: `influence`, `credit`, `oldRating`,
  `newRating`, `nodeRating`, `theta`, `predicted`, `regAvg`
- **`whr_sailors.parquet`** — per sailor per type: `rating`, `se`, `shrunk`, `lcb`,
  `popmean`, `races`, `regattas`, `team`, `region`

`regAvg` (the average rating of everyone entered in a regatta, shown next to a sailor's
own rating) is recomputed from the joint fit rather than from the old sweep, averaged
across both positions, matching `main.getRegAvgFR`.

With `runOpenskill = False` the race sweep still runs — it builds partners, penalties,
ratio, cross-region links and every non-rating column — but no `rate()` or
`predict_rank()` call is made, and the rating columns are filled from the joint fit
afterwards. The openskill code is intact and comes back by setting `runOpenskill = True`
and `useWHR = False`.

### 11.1 Resume and warm start

The old system's `calcAll = False` path tried to resume by replaying only new races. A
joint fit cannot work that way, and does not need to: it is not append-only because it
revises history, so there is no valid "start from where we left off".

Instead you **warm-start from the previous fit**. Measured: `sr` refit 53.7s cold versus
10.4s warm; a full live cycle (build + warm refit, no SEs) about 37 seconds, which is
faster than the incremental openskill path ever was. Two pieces are still missing for a
real live loop: `theta` is not persisted between processes, and upload is not incremental
(measured per-race churn between runs is only 0.01–0.10 points, so a threshold filter
would skip almost every row).

---

## 12. Config reference

Every knob that affects the model, in `refactor/config.py`.

| key | value | what it does |
|---|---|---|
| `useWHR` | `True` | joint fit is authoritative; `False` reverts to openskill |
| `runOpenskill` | `False` | skip openskill rating math (sweep still builds other columns) |
| `runWHRFit` | `True` | run the fit; `False` reuses existing `whr_*.parquet` |
| `whrWCurve` | `0.055` | Wiener drift per day, curve fit. **The** knob on curve liveliness |
| `whrWLevel` | `0.6` | Wiener drift per season, level fit used for SEs |
| `whrSigma0` | `3.0` | prior sd on a sailor's first node. Do not tighten (§2.4) |
| `whrWTR` | `0.03` | team-racing drift, tuned separately |
| `whrSigma0TR` | `1.0` | team-racing initial prior |
| `whrTargetMean` | `1400.0` | display anchor mean |
| `whrTargetSd` | `400.0` | display anchor sd. **Free parameter** (§5.1) |
| `whrAnchor` | `{}` | optional per-type anchor override |
| `maxRatingSEFraction` | `0.25` | eligibility gate as a fraction of `targetSd` → 100 pts |
| `whrRegattaNoiseFraction` | `0.0` | per-regatta form noise added to SE. Currently **off** (§6.4) |
| `hierarchicalShrinkage` | `True` | two-level sailor→team→global shrinkage (§7.2) |
| `minRegattasRanked` | `0` | **off** deliberately; volume rule (§9.4) |
| `rankingStatistic` | `'rating'` | what the sailor leaderboard sorts on |
| `publishedStatistic` | `'rating'` | what the site displays for a sailor |
| `teamRatingStatistic` | `'shrunk'` | fleet team rating input |
| `teamRatingStatisticTR` | `'rating'` | team-race team rating input. Must stay unshrunk (§7.3) |
| `timeResolution` | `regatta` / `season` | set per fit in the pipeline, not a single config key |

Note: the comment block above `rankingStatistic` contains a stale paragraph claiming
individual rankings are "back on `lcb`". They are not — the value is `'rating'`, and the
reasoning for that is in the paragraph immediately below it. Same for
`Sailors.rankingKey`'s docstring, which says it sorts on the shrunk rating; it sorts on
`config.rankingStatistic`. And `publishedRating`'s docstring says team-race types have
no joint fit, which is no longer true. Worth cleaning up.

---

## 13. Known limitations

Things that are genuinely imperfect, as distinct from things that were measured and
chosen.

1. **No common starting rating.** Structural (§5.1). 1,697 distinct first-race values on
   `sr`. This requirement is not met and cannot be met by this model class.
2. **The fit revises history.** Past points shift slightly on every refit — measured
   median 0.01–0.10 display points, so visually invisible, but it means a sailor's 2024
   rating is not a frozen historical fact.
3. **Within-regatta rating movement is attribution, not fitted skill** (§8.3).
4. **The leaderboard number and the curve endpoint differ by sd 28 points**, because
   they come from two different fits of the same sailor (§5.2). The systematic 79-point
   offset is fixed; this residual is inherent to running two fits.
5. **Thin records still over-rank occasionally.** Queen's at #24–25 off two regattas is
   the standing example. One-regatta sailors have model SE around 59, so they keep ~93%
   of their raw rating. Pushing this harder requires a volume rule, and every volume
   rule tried shifted the board by region instead (§9.4). The honest fix is presentation:
   publish the team SE so that near-ties read as ties. Queen's is 1868 ± 73 next to
   Northeastern's 1847 ± 26 — ranks alone cannot convey that.
6. **USC vs BU is within noise.** USC − BU is about 7 points against an SE of 29
   (z = 0.24). The model cannot order them, and any change that reliably did would be
   fitting judgment rather than data.
7. **6,747 team-race rows per type have no `sailorID`**, and `buildTRData` collapses all
   keyless boats in a regatta into one shared node — 212 phantom composite "sailors".
   These are scrape gaps, not model errors, but they pollute team-race nodes.
8. **Unrated teams publish literal `0`.** 86 team-race, 122 women's team-race, 92
   women's fleet teams have no eligible sailors. The front end must render `0` as
   "unrated" or those teams will display a rating of 0.
9. **Unfitted sailors publish `1000`.** Same class of front-end issue.
10. **SE computation dominates runtime** (61% of 656s, single-threaded). The four
    independent fleet types could be fitted in parallel (398s → 259s measured for SEs),
    and CHOLMOD would likely beat `splu` on an SPD matrix. Deferred, not done.

---

## 14. How to re-verify any of this

The claims above are reproducible. Useful recipes:

**Does the fit converge properly?** Watch for `converged` in the fit line and an
iteration count in the low thousands for the curve fit. If `iters` is near 1,400 and it
reports success, suspect loosened tolerances (§4).

**Is the ladder self-consistent?** On `whr_races.parquet`:
```python
(d.newRating - d.oldRating - d.credit).abs().max()          # ~1e-13
g = d.groupby(["sailorID","regatta"], sort=False)
(g.last().newRating - g.last().nodeRating).abs().max()       # ~1e-13
```
and continuity: the first `oldRating` of each regatta against the previous regatta's
`nodeRating`, in the row order the parquet is written in (which is already sailor/time
order — do not re-sort by regatta name, it interleaves seasons).

**Does the leaderboard match the curve end?** Join `whr_sailors.rating` against each
sailor's last `nodeRating` in the target seasons. Mean should be ~0 and sd ~28.

**Is any conference systematically off?** `evaluation.report` and
`evaluation.fitRegionOffsets`. Expect PCCSC within noise and MCSA as the largest
residual. A model-free version — actual cross-region regatta results, no ratings
involved — is the stronger check.

**Is the display scale doing something unexpected?** Change `whrTargetSd` and confirm
that *no* ranking moves. If any does, something is reading an absolute rating where it
should read a difference.

---

## 15. Summary for someone who reads one section

The old system walked races in order and nudged ratings after each one. Because a
Plackett-Luce nudge conserves the field's total, a conference that mostly sails itself
could never move its own mean, and the regions drifted apart by hundreds of points —
provably, not fixably by tuning.

The new system throws away the sequence. It treats every race in history as one
simultaneous constraint on one unknown per sailor per regatta, adds a random-walk prior
so skill drifts smoothly in time, and solves the resulting strictly convex problem
exactly. Cross-region log-loss halves. Uncertainty comes from the curvature of that
solution and is used as an eligibility gate, replacing `outLinks`, which was measuring
connectivity backwards.

The costs: no common starting rating, history gets revised on each refit, and per-race
rating changes have to be reconstructed from the fitted curve rather than being the
primitive the model works in.
