"""Fitted per-region rating offsets.

Why this exists
---------------
Plackett-Luce conserves the sum of mu within each race, so a nearly closed regional
pool cannot move its own mean however strong it really is. Simulated with a genuine
760-point gap between two pools that never meet, the model puts them within 5 points of
each other. Worse, when a travelling sailor's rating is corrected upward, conservation
pushes their stay-at-home team-mates *down*, when the correct inference is that they
should also rise. Tuning cannot fix this - beta, replaying the chronological pass and a
floor on sigma were all measured and all fail - because the obstruction is the
conservation law, not the update gain.

The fix is one explicit level parameter per region, fitted from the races that actually
cross regions.

Model
-----
For every within-race pair (i, j),

    P(i beats j) = sigmoid( b * (r_i - r_j) + (d_gi - d_gj) )

where r is the PRE-race display rating, g is the sailor's region in that race and d is a
region offset constrained to sum to zero. Within-region pairs have d_gi - d_gj = 0, so
they contribute only to the slope b, which is what pins the rating scale; cross-region
pairs are what identify the offsets. Offsets in rating points are d / b.

Nothing here is hand-set. If the regions were already calibrated the fit returns zeros
and nothing moves; if a region wins more than its ratings predict, its offset comes out
positive. That is a measurement, not a conference penalty.

Two limitations, both measured, both worth knowing before switching this on
--------------------------------------------------------------------------
1. Selection. Only travelling sailors appear in cross-region races, and they are 192 to
   500 rating points stronger than their non-travelling conference-mates. So the fit
   describes how miscalibrated a region's *travellers* are, and applying it to the whole
   region assumes the miscalibration is uniform. It is a coarse correction.
2. Stability. Fitted per season the offsets move by 43-62 points (sd), against a
   within-window standard error of 3.5-6.7 points. A single additive per-region constant
   is therefore misspecified to some degree. Pool several seasons (the default) rather
   than fitting each one, which is the version that was validated out of sample.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, log_expit

from config import Config
from calculationsFR import regionOf

# Fixed order so the returned arrays are interpretable and reproducible.
REGIONS = ['NEISA', 'MAISA', 'PCCSC', 'SAISA', 'MCSA', 'SEISA']
_RIDX = {r: i for i, r in enumerate(REGIONS)}

EXCUSED = ("DNS", "BKD", "RDG", "BYE")


def buildPairs(df_races, df_source, ratingType, seasons):
    """Within-race pairs (ratingDiff, regionA, regionB, aWon) for the given seasons.

    df_races   post-calc rows (needs raceID, sailorID, season, score, oldRating,
               ratingType, penalty)
    df_source  scraped rows, for each sailor's contemporaneous Team
    """
    import pandas as pd

    d = df_races[(df_races['ratingType'] == ratingType)
                 & df_races['season'].isin(seasons)
                 & ~df_races['penalty'].isin(EXCUSED)]
    d = d.dropna(subset=['oldRating', 'score'])
    if d.empty:
        return np.empty(0), np.empty((0, 2), dtype=int), np.empty(0)

    src = df_source.copy()
    src['reg'] = [regionOf(t) for t in src['Team']]
    src = src.dropna(subset=['reg']).drop_duplicates(['raceID', 'key'])
    regLookup = src.set_index(['raceID', 'key'])['reg']

    d = d.assign(reg=pd.MultiIndex.from_arrays([d['raceID'], d['sailorID']]).map(regLookup))
    d = d[d['reg'].isin(REGIONS)]

    diffs, groups, wins = [], [], []
    for _, g in d.groupby('raceID', sort=False):
        if len(g) < 2:
            continue
        r = g['oldRating'].to_numpy()
        s = g['score'].to_numpy()
        gi = g['reg'].map(_RIDX).to_numpy()
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                if s[i] == s[j]:
                    continue
                diffs.append(r[i] - r[j])
                groups.append((gi[i], gi[j]))
                wins.append(1.0 if s[i] < s[j] else 0.0)

    return np.asarray(diffs), np.asarray(groups, dtype=int).reshape(-1, 2), np.asarray(wins)


def _design(groups, nFree):
    """Reference coding: the last region is pinned at 0, leaving nFree free offsets."""
    X = np.zeros((len(groups), nFree))
    for k, (a, b) in enumerate(groups):
        if a < nFree:
            X[k, a] += 1.0
        if b < nFree:
            X[k, b] -= 1.0
    return X


def fitOffsets(diffs, groups, wins, initialScale=380.0):
    """Maximum-likelihood slope and sum-to-zero region offsets, in rating points.

    Returns (offsets, standardErrors, slope, nPairs) with offsets indexed like REGIONS.
    """
    n = len(REGIONS)
    if len(diffs) < 500:
        return np.zeros(n), np.full(n, np.inf), 1.0 / initialScale, len(diffs)

    X = _design(groups, n - 1)

    def nll(p):
        z = p[0] * diffs + X @ p[1:]
        return -(wins * log_expit(z) + (1 - wins) * log_expit(-z)).sum()

    p0 = np.zeros(n)
    p0[0] = 1.0 / initialScale
    res = minimize(nll, p0, method='L-BFGS-B',
                   options={'maxiter': 4000, 'ftol': 1e-15, 'gtol': 1e-12})

    slope = res.x[0]
    if not np.isfinite(slope) or abs(slope) < 1e-9:
        return np.zeros(n), np.full(n, np.inf), 1.0 / initialScale, len(diffs)

    # Observed information, then recentre to sum-to-zero and propagate the covariance.
    z = slope * diffs + X @ res.x[1:]
    w = expit(z) * (1 - expit(z))
    J = np.column_stack([diffs, X])
    try:
        cov = np.linalg.inv(J.T @ (w[:, None] * J))
    except np.linalg.LinAlgError:
        return np.zeros(n), np.full(n, np.inf), slope, len(diffs)

    lift = np.zeros((n, n))
    lift[:n - 1, 1:] = np.eye(n - 1)          # params -> d, with the last region at 0
    centre = np.eye(n) - np.ones((n, n)) / n  # sum-to-zero

    d = centre @ np.concatenate([res.x[1:], [0.0]])
    covD = centre @ (lift @ cov @ lift.T) @ centre.T

    return d / slope, np.sqrt(np.clip(np.diag(covD), 0, None)) / slope, slope, len(diffs)


def computeRegionOffsets(df_frAfter, df_source, config: Config):
    """Fit offsets for each fleet rating type. Returns {ratingType: {region: points}}."""
    out = {}
    for ratingType in ('sr', 'cr', 'wsr', 'wcr'):
        diffs, groups, wins = buildPairs(df_frAfter, df_source, ratingType,
                                         config.regionOffsetSeasons)
        offsets, se, slope, nPairs = fitOffsets(diffs, groups, wins)

        # Shrink toward zero by the ratio of sampling variance to observed spread, so a
        # weakly determined offset is damped rather than trusted.
        spread = offsets.std()
        if spread > 0 and np.all(np.isfinite(se)):
            shrink = spread ** 2 / (spread ** 2 + se ** 2)
        else:
            shrink = np.zeros(len(offsets))
        offsets = offsets * shrink * config.regionOffsetWeight

        out[ratingType] = dict(zip(REGIONS, offsets))
        print(f"  {ratingType}: {nPairs:,} pairs, scale {1/slope:.0f} pts/logit, "
              + ", ".join(f"{r}{v:+.0f}" for r, v in zip(REGIONS, offsets)))
    return out


def applyRegionOffsets(people, offsets, config: Config):
    """Store each sailor's offset for their current region.

    Set on the Sailor rather than folded into mu, because mu/sigma are persisted and
    reloaded on a resume - baking the offset in would compound it on every run. New
    sailors keep an offset of 0 until they have a region, so everyone still starts at
    exactly the same displayed rating.
    """
    applied = 0
    for p in people.values():
        region = regionOf(p.teams[-1]) if p.teams else None
        if region is None:
            continue
        p.regionOffsets = {rt: offsets.get(rt, {}).get(region, 0.0) for rt in offsets}
        applied += 1
    print(f"  applied region offsets to {applied:,} sailors")
    return people
