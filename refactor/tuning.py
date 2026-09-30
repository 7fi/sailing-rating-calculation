"""Hyperparameter fitting for the rating model.

``beta`` was hand-set to 25/120 (= sigma0/40, against openskill's default sigma0/2)
with no metric behind it, and it is the parameter that sets how sharp the likelihood
is. At beta = sigma0/40 the Fisher information per comparison is 0.25/beta^2 = 5.76,
so pairwise predictions saturate near 0/1 and sigma collapses far too fast relative to
the mu scale. A nearly closed regional pool then self-normalizes to its own initial
mean regardless of true strength, because Plackett-Luce updates are approximately
mean-preserving within a rated field.

This module replaces the hand-set value with one chosen by out-of-sample predictive
likelihood. It rebuilds the chronological sweep over flat numpy arrays so a trial
costs seconds instead of a full ``main.py`` run, while calling the same
``PlackettLuce.rate`` as production so the arithmetic cannot drift.

Protocol: prequential (rolling-origin). Every race is predicted from its own pre-race
mu/sigma before being used to update, so every row is already out-of-sample and no
holdout is required. Whole-regatta holdout is only needed once ratings stop being a
forward-only filter (Stage 3).
"""

from dataclasses import dataclass, replace
import itertools
import time

import numpy as np
import pandas as pd
from openskill.models import PlackettLuce, PlackettLuceRating

from config import Config
from regions import teamRegions
import evaluation

EXCUSED = ["DNS", "BKD", "RDG", "BYE"]


@dataclass(frozen=True)
class HyperParams:
    beta: float
    tau: float
    sigma0: float
    limitSigma: bool = False
    sigmaFloor: float = 0.0

    def model(self):
        return PlackettLuce(mu=25.0, sigma=self.sigma0, beta=self.beta,
                            tau=self.tau, limit_sigma=self.limitSigma)

    def label(self):
        return (f"beta={self.beta:.4f}(s0/{self.sigma0/self.beta:.1f}) "
                f"tau={self.tau:.4f} sigma0={self.sigma0:.3f} "
                f"limit={int(self.limitSigma)} floor={self.sigmaFloor:.2f}")


@dataclass
class SweepData:
    entity: np.ndarray        # int32, row -> index into the mu/sigma state vector
    place: np.ndarray         # float64, finishing place (lower is better)
    active: np.ndarray        # bool, False for excused penalties
    fieldStart: np.ndarray    # int32
    fieldEnd: np.ndarray      # int32
    fieldRatable: np.ndarray  # bool, passes calculateFR's skip conditions
    fieldRegatta: np.ndarray  # int32, regatta index per field (for holdout)
    fieldCross: np.ndarray    # bool, field spans more than one region
    nEntities: int
    rows: pd.DataFrame        # per-row metadata for evaluation (region, score, field)
    evalMask: np.ndarray      # bool, rows inside the evaluation window


def buildSweepArrays(rootDir="", frFile="racesfr.parquet",
                     postFile="postcalcFRraces.parquet",
                     evalSeasons=("f24", "s25", "f25", "s26")):
    """Flatten the fleet-race history into arrays a sweep can walk without pandas.

    The rated field is (adjusted_raceID, position). The rating type comes from the
    persisted post-calc output so the womens/open split matches production exactly
    rather than being re-derived. Every sailor holds one rating per rating type, so
    the state vector is indexed by (sailor, ratingType).
    """
    df = pd.read_parquet(rootDir + frFile,
                         columns=["raceID", "adjusted_raceID", "key", "Sailor", "Team",
                                  "Position", "Score", "penalty", "Date", "Regatta",
                                  "raceNum", "Div"])
    df["sailorID"] = df["key"].where(
        df["key"].notna() & (df["key"] != "Unknown"),
        df["Sailor"].astype(str) + "-" + df["Team"].astype(str))
    # Apply the same identity merges handleMerges applies, so a merged sailor is one
    # entity here too.
    df["sailorID"] = df["sailorID"].replace(Config.merges)

    post = pd.read_parquet(rootDir + postFile,
                           columns=["raceID", "position", "ratingType"]).drop_duplicates()
    rt = post.set_index(["raceID", "position"])["ratingType"]
    df["ratingType"] = df.set_index(["raceID", "Position"]).index.map(rt)
    df = df[df["ratingType"].notna()]          # keep only fields production rated

    df["region"] = df["Team"].map(teamRegions).replace(evaluation.REGION_MERGE)
    df["season"] = df["raceID"].str.split("/").str[0]
    df["field"] = (df["adjusted_raceID"].astype(str) + "|" + df["Position"].astype(str))

    # Reproduce main.py's iteration order exactly, or the ratings diverge:
    #   1. main.load sorts the whole frame by ['Date', 'raceNum', 'Div'].
    #   2. calcAllRacesForRT groups by Regatta with sort=False, so regattas are walked
    #      in order of first appearance - every race of one regatta before the next,
    #      even when dates interleave across a weekend.
    #   3. Within a regatta it groups by adjusted_raceID with sort=False, again first
    #      appearance - NOT alphabetically (which would put race 10 before race 2).
    df = df.sort_values(["Date", "raceNum", "Div"], kind="stable").reset_index(drop=True)
    df["_seq"] = np.arange(len(df))
    regattaOrder = df.groupby("Regatta")["_seq"].transform("min")
    fieldOrder = df.groupby("field")["_seq"].transform("min")
    df["_regattaOrder"] = regattaOrder
    df["_fieldOrder"] = fieldOrder
    df = df.sort_values(["_regattaOrder", "_fieldOrder", "_seq"],
                        kind="stable").reset_index(drop=True)

    sailors, sIdx = np.unique(df["sailorID"].to_numpy(), return_inverse=True)
    rtypes, rIdx = np.unique(df["ratingType"].to_numpy(), return_inverse=True)
    entity = (sIdx * len(rtypes) + rIdx).astype(np.int32)
    nEntities = len(sailors) * len(rtypes)

    place = df["Score"].to_numpy(float)
    active = ~df["penalty"].isin(EXCUSED).to_numpy()

    sizes = df.groupby("field", sort=False).size().to_numpy()
    ends = np.cumsum(sizes)
    starts = np.concatenate([[0], ends[:-1]])

    ratable = np.zeros(len(sizes), bool)
    for k, (s, e) in enumerate(zip(starts, ends)):
        if (e - s) < 2:                       # fewer than two sailors
            continue
        if not np.isfinite(place[s]):         # B division did not complete the set
            continue
        if active[s:e].sum() < 2:             # fewer than two unpenalised sailors
            continue
        ratable[k] = True

    # Regatta identity per field, for whole-regatta holdout.
    regattas, gIdx = np.unique(df["Regatta"].to_numpy(), return_inverse=True)
    fieldRegatta = gIdx[starts].astype(np.int32)
    regionCode = df["region"].fillna("?").to_numpy()
    fieldCross = np.array([len(set(regionCode[s:e]) - {"?"}) > 1
                           for s, e in zip(starts, ends)])

    rows = df[["field", "sailorID", "region", "Score", "season", "Position",
               "ratingType", "Regatta"]].rename(
        columns={"Score": "score", "Position": "position", "Regatta": "regatta"})
    evalMask = df["season"].isin(evalSeasons).to_numpy()

    return SweepData(entity, place, active, starts.astype(np.int32),
                     ends.astype(np.int32), ratable, fieldRegatta, fieldCross,
                     nEntities, rows, evalMask)


def makeRegattaHoldout(sweep: SweepData, frac=0.10, seed=0):
    """Hold out whole regattas, stratified on whether the regatta is cross-regional.

    Whole regattas, never individual races or rows: sailors race the same people
    repeatedly inside one regatta, so a race-level split leaks results.
    """
    rng = np.random.default_rng(seed)
    crossByRegatta = {}
    for g, c in zip(sweep.fieldRegatta, sweep.fieldCross):
        crossByRegatta[g] = crossByRegatta.get(g, False) or bool(c)

    held = []
    for isCross in (True, False):
        pool = np.array([g for g, c in crossByRegatta.items() if c == isCross])
        if len(pool) == 0:
            continue
        n = max(1, int(round(frac * len(pool))))
        held.append(rng.choice(pool, n, replace=False))
    heldRegattas = set(np.concatenate(held).tolist())
    return np.array([g in heldRegattas for g in sweep.fieldRegatta])


def iteratedSweep(sweep: SweepData, params: HyperParams, epochs=1, shrink=1.0,
                  damping=1.0, holdoutFields=None):
    """Repeat the chronological sweep, carrying mu forward between epochs.

    Held-out fields are predicted but never used to update, in every epoch, so the
    recorded predictions stay out-of-sample even though iteration lets information
    flow backward in time.

    Restarting each epoch from the previous epoch's mu conditions on the same data
    repeatedly; `shrink` (rho) pulls mu back toward the prior mean each epoch, which
    makes the fixed point a MAP under an explicit ridge prior instead of the
    whole-history MLE (which diverges for never-beaten / never-won records).
    """
    model = params.model()
    mu0 = model.mu
    mu = np.full(sweep.nEntities, mu0)
    sigma = np.full(sweep.nEntities, params.sigma0)
    preMu = np.empty(len(sweep.entity))
    preSigma = np.empty(len(sweep.entity))
    floor = params.sigmaFloor
    if holdoutFields is None:
        holdoutFields = np.zeros(len(sweep.fieldStart), bool)

    for ep in range(epochs):
        prior = mu.copy()
        mu = mu0 + shrink * (mu - mu0)
        sigma[:] = max(params.sigma0, floor) if floor else params.sigma0
        last = (ep == epochs - 1)

        for k in range(len(sweep.fieldStart)):
            s, e = sweep.fieldStart[k], sweep.fieldEnd[k]
            ent = sweep.entity[s:e]
            if last:
                preMu[s:e] = mu[ent]
                preSigma[s:e] = sigma[ent]
            if not sweep.fieldRatable[k] or holdoutFields[k]:
                continue
            act = sweep.active[s:e]
            aEnt = ent[act]
            teams = [[PlackettLuceRating(mu[i], sigma[i])] for i in aEnt]
            out = model.rate(teams, ranks=list(sweep.place[s:e][act]))
            for m, i in enumerate(aEnt):
                mu[i] = out[m][0].mu
                sigma[i] = max(out[m][0].sigma, floor) if floor else out[m][0].sigma

        if damping != 1.0 and not last:
            mu = prior + damping * (mu - prior)
        yield ep, preMu, preSigma, mu


def fastSweep(sweep: SweepData, params: HyperParams):
    """One chronological pass. Returns per-row pre-race (mu, sigma).

    Uses the same PlackettLuce.rate as production, with places passed as ranks=
    (lower is better). openskill's scores= parameter is the opposite convention.
    """
    model = params.model()
    mu = np.full(sweep.nEntities, model.mu)
    sigma = np.full(sweep.nEntities, params.sigma0)
    preMu = np.empty(len(sweep.entity))
    preSigma = np.empty(len(sweep.entity))
    floor = params.sigmaFloor

    for s, e, ratable in zip(sweep.fieldStart, sweep.fieldEnd, sweep.fieldRatable):
        ent = sweep.entity[s:e]
        preMu[s:e] = mu[ent]
        preSigma[s:e] = sigma[ent]
        if not ratable:
            continue
        act = sweep.active[s:e]
        aEnt = ent[act]
        teams = [[PlackettLuceRating(mu[i], sigma[i])] for i in aEnt]
        out = model.rate(teams, ranks=list(sweep.place[s:e][act]))
        for k, i in enumerate(aEnt):
            mu[i] = out[k][0].mu
            sigma[i] = max(out[k][0].sigma, floor) if floor else out[k][0].sigma

    return preMu, preSigma


def evalFrame(sweep: SweepData, preMu, preSigma, ratingTypes=("sr",)):
    """Assemble the evaluation DataFrame the harness in evaluation.py expects."""
    d = sweep.rows.loc[sweep.evalMask].copy()
    d["oldMu"] = preMu[sweep.evalMask]
    d["oldSigma"] = preSigma[sweep.evalMask]
    if ratingTypes:
        d = d[d["ratingType"].isin(ratingTypes)]
    return d


def trial(sweep: SweepData, params: HyperParams, alpha=24.0, ratingTypes=("sr",)):
    """Run one hyperparameter setting and score it."""
    t0 = time.time()
    preMu, preSigma = fastSweep(sweep, params)
    d = evalFrame(sweep, preMu, preSigma, ratingTypes=ratingTypes)

    allPairs = evaluation.buildPairs(d, crossOnly=False)
    crossPairs = allPairs[allPairs["regionI"] != allPairs["regionJ"]]
    withinPairs = allPairs[allPairs["regionI"] == allPairs["regionJ"]]
    fit = evaluation.fitRegionOffsets(crossPairs, params.beta, alpha)

    # Offsets in display points are NOT comparable across hyperparameters, because
    # changing beta rescales mu and therefore the whole display scale. Normalising by
    # the population spread of ratings makes the bias scale-free.
    ratingSd = float(np.std(d["oldMu"].to_numpy() * alpha))
    spreadPts = float(fit.offsets.max() - fit.offsets.min())
    calib = evaluation.regionCalibration(crossPairs, params.beta)

    return {
        "beta": params.beta, "tau": params.tau, "sigma0": params.sigma0,
        "limitSigma": params.limitSigma, "sigmaFloor": params.sigmaFloor,
        "beta_ratio": params.sigma0 / params.beta,
        "logloss_all": evaluation.logLoss(allPairs, params.beta),
        "logloss_cross": evaluation.logLoss(crossPairs, params.beta),
        "acc_all": evaluation.pairAccuracy(allPairs),
        "acc_within": evaluation.pairAccuracy(withinPairs),
        "acc_cross": evaluation.pairAccuracy(crossPairs),
        "max_abs_offset_pts": float(np.max(np.abs(fit.offsets))),
        "offset_spread_pts": spreadPts,
        "rating_sd_pts": ratingSd,
        "offset_spread_sd": spreadPts / ratingSd if ratingSd else float("nan"),
        "max_abs_z": fit.maxAbsZ,
        # Scale-free: worst per-region gap between actual and predicted win rate.
        "max_abs_calib": float(calib["calib"].abs().max()),
        "max_abs_order_bias": float(calib["order_bias"].abs().max()),
        "secs": time.time() - t0,
    }


def randomSearch(sweep, nTrials=40, seed=0, alpha=24.0, ratingTypes=("sr",),
                 sigma0=25.0 / 3.0):
    """Random search over the likelihood-sharpness knobs.

    beta and sigma0 are nearly collinear (only their ratio drives the likelihood), so
    sigma0 is held fixed and beta is searched as a ratio of it; the display scale is
    absorbed by Config.alpha downstream.
    """
    rng = np.random.default_rng(seed)
    results = []
    for k in range(nTrials):
        ratio = float(np.exp(rng.uniform(np.log(1.0), np.log(60.0))))   # sigma0/beta
        params = HyperParams(
            beta=sigma0 / ratio,
            tau=float(sigma0 * rng.uniform(0.0, 0.05)),
            sigma0=sigma0,
            limitSigma=bool(rng.integers(0, 2)),
            sigmaFloor=float(sigma0 * rng.uniform(0.0, 0.4)),
        )
        r = trial(sweep, params, alpha=alpha, ratingTypes=ratingTypes)
        results.append(r)
        print(f"[{k+1}/{nTrials}] {params.label()}  "
              f"ll_all={r['logloss_all']:.4f} ll_cross={r['logloss_cross']:.4f} "
              f"acc={r['acc_all']:.4f} maxoff={r['max_abs_offset_pts']:.1f}pts "
              f"({r['secs']:.0f}s)", flush=True)
    return pd.DataFrame(results).sort_values("logloss_all").reset_index(drop=True)
