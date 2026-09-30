"""Joint Whole-History Rating for team racing.

Team racing is 3-on-3 with a win/loss outcome, so the likelihood is Bradley-Terry on
the combined strength of each side rather than Plackett-Luce over a finishing order:

    P(A beats B) = sigmoid( sum(theta over A's three sailors) - sum over B's )

applied separately to skippers and crews, matching how the openskill pass treated a
team as the three same-position sailors. The prior is the same Wiener random walk
between a sailor's consecutive regattas used for fleet racing, so this module reuses
whr's node construction, warm start and standard-error machinery.

Ties (4 rows in the whole history) enter as y = 0.5, which the Bernoulli likelihood
handles directly.
"""

from dataclasses import dataclass
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, log_expit

from config import Config
from regions import teamRegions
import whr
import whrClassify
import evaluation

TR_RATING_TYPES = ("tsr", "tcr", "wtsr", "wtcr")


@dataclass
class TRData:
    aNodes: np.ndarray      # (M, 3) int32 - one side's sailors for this position
    bNodes: np.ndarray      # (M, 3) int32
    y: np.ndarray           # (M,) 1.0 A won, 0.0 A lost, 0.5 tie
    nNodes: int
    nodeSailor: np.ndarray
    priorA: np.ndarray
    priorB: np.ndarray
    priorDt: np.ndarray
    firstNode: np.ndarray
    nodeLabel: np.ndarray
    matches: pd.DataFrame   # per-match metadata (regatta, season, teams, outcome)
    rows: pd.DataFrame      # per (match, side, sailor), for output
    matchRegatta: np.ndarray


def buildTRData(rootDir="", trFile="racesTR.parquet", ratingType="tsr",
                seasons=None, timeResolution="regatta", dtFloorDays=0.1):
    """Flatten team races into paired node arrays plus the Wiener prior structure."""
    pos = "skipper" if ratingType.endswith("sr") else "crew"
    wantWomens = ratingType.startswith("w")
    keyField = f"{pos}Key"

    df = pd.read_parquet(rootDir + trFile)
    df = df[(df["teamABoats"].apply(len) == 3) & (df["teamBBoats"].apply(len) == 3)]
    df["season"] = df["raceID"].str.split("/").str[0]
    if seasons:
        df = df[df["season"].isin(seasons)]

    genders = whrClassify.genderMap(rootDir, merges=Config.merges)
    womens = whrClassify.trWomensByRegatta(df, genders)
    df = df[df["Regatta"].map(womens).fillna(False) == wantWomens]
    if df.empty:
        return None

    df = df.sort_values(["Date", "raceNum"], kind="stable").reset_index(drop=True)
    merges = Config.merges

    def side(boats):
        return [merges.get(b[keyField], b[keyField]) for b in boats]

    aKeys = df["teamABoats"].apply(side)
    bKeys = df["teamBBoats"].apply(side)

    bucketCol = "Regatta" if timeResolution == "regatta" else "season"
    bucket = df[bucketCol].astype(str)
    days = pd.to_datetime(df["Date"]).astype("int64") / 86_400_000_000_000.0

    # one node per (sailor, bucket), as in the fleet fit
    labels, nodeOf = [], {}
    def nodeFor(k, b):
        lbl = f"{k}\x00{b}"
        idx = nodeOf.get(lbl)
        if idx is None:
            idx = nodeOf[lbl] = len(labels)
            labels.append(lbl)
        return idx

    aN = np.empty((len(df), 3), np.int32)
    bN = np.empty((len(df), 3), np.int32)
    nodeTime = {}
    for i, (ak, bk, b, d) in enumerate(zip(aKeys, bKeys, bucket, days)):
        for j in range(3):
            aN[i, j] = na = nodeFor(ak[j], b)
            bN[i, j] = nb = nodeFor(bk[j], b)
            nodeTime.setdefault(na, d); nodeTime.setdefault(nb, d)

    nNodes = len(labels)
    nodeLabel = np.array(labels, dtype=object)
    sailorOf = np.array([l.split("\x00", 1)[0] for l in labels])
    _, nodeSailor = np.unique(sailorOf, return_inverse=True)
    t = np.array([nodeTime.get(i, 0.0) for i in range(nNodes)], float)

    order = np.lexsort((t, nodeSailor))
    ns, se = nodeSailor[order], t[order]
    same = ns[1:] == ns[:-1]
    priorA = order[:-1][same].astype(np.int32)
    priorB = order[1:][same].astype(np.int32)
    priorDt = np.maximum((se[1:][same] - se[:-1][same]).astype(float), dtFloorDays)
    firstNode = order[np.concatenate([[True], ~same])].astype(np.int32)

    y = df["teamAOutcome"].map({"win": 1.0, "lose": 0.0, "tie": 0.5}).to_numpy(float)

    regs, matchRegatta = np.unique(df["Regatta"].to_numpy(), return_inverse=True)
    rows = []
    for sideName, keys, nodes in (("A", aKeys, aN), ("B", bKeys, bN)):
        for j in range(3):
            rows.append(pd.DataFrame({
                "raceID": df["raceID"].to_numpy(), "regatta": df["Regatta"].to_numpy(),
                "season": df["season"].to_numpy(), "side": sideName,
                "sailorID": [k[j] for k in keys], "node": nodes[:, j],
                "team": df[f"team{sideName}Name"].to_numpy(),
                "outcome": df[f"team{sideName}Outcome"].to_numpy(),
            }))
    rows = pd.concat(rows, ignore_index=True)
    rows["ratingType"] = ratingType
    rows["position"] = pos.title()

    matches = df[["raceID", "Regatta", "season", "teamAName", "teamBName",
                  "teamAOutcome", "Date"]].reset_index(drop=True)
    return TRData(aN, bN, y, nNodes, nodeSailor, priorA, priorB, priorDt,
                  firstNode, nodeLabel, matches, rows, matchRegatta)


def negLogPostTR(theta, data: TRData, w, sigma0, activeMask=None):
    """Negative log posterior and gradient for the Bradley-Terry team likelihood."""
    aN, bN, y = data.aNodes, data.bNodes, data.y
    if activeMask is not None:
        aN, bN, y = aN[activeMask], bN[activeMask], y[activeMask]

    z = theta[aN].sum(axis=1) - theta[bN].sum(axis=1)
    f = -float(np.sum(y * log_expit(z) + (1.0 - y) * log_expit(-z)))
    r = y - expit(z)                       # dloglik/dz
    g = np.zeros_like(theta)
    np.add.at(g, aN.ravel(), np.repeat(-r, 3))
    np.add.at(g, bN.ravel(), np.repeat(r, 3))

    d = theta[data.priorB] - theta[data.priorA]
    prec = 1.0 / (w * w * data.priorDt)
    f += 0.5 * float(np.sum(prec * d * d))
    np.add.at(g, data.priorB, prec * d)
    np.add.at(g, data.priorA, -prec * d)

    t0 = theta[data.firstNode]
    f += 0.5 * float(np.sum(t0 * t0) / (sigma0 * sigma0))
    np.add.at(g, data.firstNode, t0 / (sigma0 * sigma0))
    return f, g


def fitTR(data: TRData, w=0.055, sigma0=3.0, activeMask=None, theta0=None,
          maxiter=4000, ftol=1e-11, gtol=1e-7, verbose=True):
    x0 = np.zeros(data.nNodes) if theta0 is None else theta0.copy()
    t0 = time.time()
    res = minimize(negLogPostTR, x0, args=(data, w, sigma0, activeMask), jac=True,
                   method="L-BFGS-B",
                   options={"maxiter": maxiter, "maxcor": 20, "ftol": ftol, "gtol": gtol})
    if verbose:
        print(f"  TR fit w={w:.3f} sigma0={sigma0:.2f}  nll={res.fun:.1f}  "
              f"iters={res.nit}  {'converged' if res.success else res.message}  "
              f"{time.time()-t0:.0f}s", flush=True)
    return res.x, res


def rowCreditTR(theta, data: TRData, w, sigma0):
    """Per-match credit: how much each match pulls one sailor's team-race rating.

    Same influence-function approximation the fleet fit uses in whr.rowCredit,

        credit = (dlogL_match / dtheta_n) / H_nn

    specialised to the 3v3 Bradley-Terry likelihood. With
    ``z = sum(theta_A) - theta_B`` and ``p = sigmoid(z)``, the per-match residual
    ``r = y - p`` is dlogL/dz, and dz/dtheta is +1 for every sailor on side A and -1 for
    every sailor on side B. So all three sailors on a side receive the same credit -
    which is the honest answer, because a 3v3 result only ever observes the combined
    strength of the three and carries no information about which of them earned it.

    Returns an array aligned to ``data.rows``, whose block order is
    (A, boat0), (A, boat1), (A, boat2), (B, boat0), (B, boat1), (B, boat2).
    """
    aN, bN, y = data.aNodes, data.bNodes, data.y
    z = theta[aN].sum(axis=1) - theta[bN].sum(axis=1)
    p = expit(z)
    r = y - p

    # Curvature: each participating node has (dz/dtheta)^2 = 1, so contributes p(1-p).
    hDiag = np.zeros_like(theta)
    pv = p * (1.0 - p)
    np.add.at(hDiag, aN.ravel(), np.repeat(pv, 3))
    np.add.at(hDiag, bN.ravel(), np.repeat(pv, 3))

    # Prior curvature, exactly as in whr.rowCredit.
    prec = 1.0 / (w * w * data.priorDt)
    np.add.at(hDiag, data.priorA, prec)
    np.add.at(hDiag, data.priorB, prec)
    np.add.at(hDiag, data.firstNode, 1.0 / (sigma0 * sigma0))

    rowGrad = np.concatenate([np.tile(r, 3), np.tile(-r, 3)])
    rowNode = np.concatenate([aN[:, 0], aN[:, 1], aN[:, 2],
                              bN[:, 0], bN[:, 1], bN[:, 2]])
    return rowGrad / np.maximum(hDiag[rowNode], 1e-9), rowNode


def trLadder(data: TRData, theta, credit, rowNode, scale, offset):
    """Per-match rating ladder for team racing, mirroring whr.raceLadder.

    Walks each sailor through each regatta so that ``newRating - oldRating == credit``
    for every match and the walk lands exactly on that regatta's fitted rating. Without
    this the TR score rows carried whatever the (skipped) openskill pass left behind,
    which was a constant 1000 for every sailor in every match.
    """
    d = data.rows.copy()
    d["credit"] = credit * scale
    d["nodeRating"] = theta[rowNode] * scale + offset
    # Raw theta as well: pairwise probabilities are sigmoid of theta differences, and
    # with openskill off the oldMu column would otherwise sit at the prior.
    d["theta"] = theta[rowNode]

    # Did the fit expect this side to win? Kept NUMERIC (1.0 win / 0.0 lose) because
    # whr_races.parquet holds fleet rows in the same column as an integer finishing
    # place, and a mixed int/str column cannot be written to parquet. Sailors
    # .applyWHRToRaces maps it back to 'win'/'lose' for the team-race score rows.
    aSum = theta[data.aNodes].sum(axis=1)
    bSum = theta[data.bNodes].sum(axis=1)
    ownSum = np.concatenate([np.tile(aSum, 3), np.tile(bSum, 3)])
    oppSum = np.concatenate([np.tile(bSum, 3), np.tile(aSum, 3)])
    d["predicted"] = np.where(ownSum > oppSum, 1.0, 0.0)

    # A match number is not in the flattened rows, so order within a regatta by raceID,
    # which carries the sequence.
    d = d.sort_values(["sailorID", "regatta", "raceID"], kind="stable")

    g = d.groupby(["sailorID", "regatta"], sort=False)["credit"]
    total = g.transform("sum")
    creditBefore = g.cumsum() - d["credit"]
    d["oldRating"] = d["nodeRating"] - total + creditBefore
    d["newRating"] = d["oldRating"] + d["credit"]
    return d


def buildHessianTR(theta, data: TRData, w, sigma0, activeMask=None):
    """Laplace precision: sum_m p(1-p) (e_A - e_B)(e_A - e_B)^T, plus the prior."""
    import scipy.sparse as sp
    aN, bN = data.aNodes, data.bNodes
    if activeMask is not None:
        aN, bN = aN[activeMask], bN[activeMask]
    z = theta[aN].sum(axis=1) - theta[bN].sum(axis=1)
    p = expit(z)
    wgt = p * (1.0 - p)

    idx = np.concatenate([aN, bN], axis=1)                  # (M, 6)
    sign = np.concatenate([np.ones((len(idx), 3)), -np.ones((len(idx), 3))], axis=1)
    outer = sign[:, :, None] * sign[:, None, :] * wgt[:, None, None]
    ii = np.repeat(idx, 6, axis=1).ravel()
    jj = np.tile(idx, (1, 6)).ravel()
    vals = outer.ravel()

    prec = 1.0 / (w * w * data.priorDt)
    a, b = data.priorA, data.priorB
    ii = np.concatenate([ii, a, b, a, b])
    jj = np.concatenate([jj, a, b, b, a])
    vals = np.concatenate([vals, prec, prec, -prec, -prec])
    ii = np.concatenate([ii, data.firstNode])
    jj = np.concatenate([jj, data.firstNode])
    vals = np.concatenate([vals, np.full(len(data.firstNode), 1.0 / (sigma0 ** 2))])

    H = sp.coo_matrix((vals, (ii, jj)), shape=(data.nNodes, data.nNodes)).tocsc()
    H.sum_duplicates()
    return H


def fitTRWithSE(rootDir="", ratingType="tsr", w=0.03, sigma0=1.0,
                targetSeasons=("s26",), targetMean=1400.0, targetSd=400.0,
                verbose=True):
    """Fit one team-race graph and attach posterior SEs for the current sailors."""
    import scipy.sparse.linalg as spl

    data = buildTRData(rootDir=rootDir, ratingType=ratingType)
    if data is None:
        return pd.DataFrame(), pd.DataFrame()
    if verbose:
        print(f"[{ratingType}] {len(data.y):,} matches, {data.nNodes:,} nodes", flush=True)
    theta, _ = fitTR(data, w=w, sigma0=sigma0, verbose=verbose)

    cur = data.rows[data.rows["season"].isin(list(targetSeasons))]
    if cur.empty:
        return pd.DataFrame(), pd.DataFrame()
    info = (cur.groupby("node")
               .agg(sailorID=("sailorID", "first"), team=("team", "first"),
                    matches=("node", "size"), regattas=("regatta", "nunique"))
               .reset_index()
               .groupby("sailorID", as_index=False).last())
    nodes = info["node"].to_numpy()

    ref = np.zeros(data.nNodes)
    wts = info["matches"].to_numpy(float)
    ref[nodes] = wts / wts.sum()
    scale, offset = whr.anchorScale(theta, nodes, targetMean, targetSd)

    H = buildHessianTR(theta, data, w, sigma0)
    lu = spl.splu(H)
    ses = np.empty(len(nodes))
    for s in range(0, len(nodes), 400):
        sl = slice(s, min(s + 400, len(nodes)))
        V = np.zeros((data.nNodes, sl.stop - sl.start))
        V[nodes[sl], np.arange(sl.stop - sl.start)] = 1.0
        V -= ref[:, None]
        X = lu.solve(V)
        ses[sl] = np.sqrt(np.maximum((V * X).sum(axis=0), 0.0))

    info = info.drop(columns=["node"])
    info["ratingType"] = ratingType
    info["rating"] = theta[nodes] * scale + offset
    info["se"] = ses * scale
    # Empirical-Bayes shrinkage toward the population mean, by reliability:
    #     shrunk = m + (rating - m) * tau2 / (tau2 + se^2)
    # with tau2 = var(rating) - mean(se^2), the method-of-moments estimate of the true
    # between-sailor variance. No free constant. This is the posterior mean under a
    # normal prior, so it is the right point estimate to publish and to rank on:
    # a sailor with one regatta regresses most of the way to average, while a
    # well-measured sailor keeps their rating. It replaces ranking on rating - 1.96*SE,
    # which penalised by uncertainty rather than regressing toward the mean.
    m = float(info["rating"].mean())
    tau2 = max(float(info["rating"].var()) - float((info["se"] ** 2).mean()), 1e-6)
    info["shrunk"] = m + (info["rating"] - m) * tau2 / (tau2 + info["se"] ** 2)
    info["popmean"] = m
    info["lcb"] = info["rating"] - 1.96 * info["se"]
    info["region"] = info["team"].map(teamRegions).replace(evaluation.REGION_MERGE)
    info = info.rename(columns={"matches": "races"}).drop(columns=["team"])

    # Per-match ladder, so the TR score rows carry a real rating that moves between
    # matches instead of whatever the skipped openskill pass left behind.
    credit, rowNode = rowCreditTR(theta, data, w, sigma0)
    races = trLadder(data, theta, credit, rowNode, scale, offset)
    races["rating"] = races["nodeRating"]
    races = races.drop(columns=["node"], errors="ignore")
    return info, races


def runTRPipeline(rootDir="", w=0.03, sigma0=1.0, targetSeasons=("s26",),
                  targetMean=1400.0, targetSd=400.0, verbose=True):
    """Fit every team-race rating type. Mirrors whr.runFleetPipeline."""
    sailors, races = [], []
    for rt in TR_RATING_TYPES:
        if verbose:
            print(f"\n=== {rt} ===", flush=True)
        s, r = fitTRWithSE(rootDir=rootDir, ratingType=rt, w=w, sigma0=sigma0,
                           targetSeasons=targetSeasons, targetMean=targetMean,
                           targetSd=targetSd, verbose=verbose)
        if len(s):
            sailors.append(s); races.append(r)
    sailors = pd.concat(sailors, ignore_index=True) if sailors else pd.DataFrame()
    races = pd.concat(races, ignore_index=True) if races else pd.DataFrame()
    return sailors, races
