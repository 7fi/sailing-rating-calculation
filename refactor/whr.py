"""Joint Whole-History Rating fit for the sailing ratings.

Replaces the sequential openskill filter with a single joint MAP fit over all races.

Why: Plackett-Luce updates are approximately mean-preserving *within a rated field*,
so a sequential filter can only redistribute strength among the boats in one race.
When a PCCSC sailor meets two NEISA visitors among twenty locals, the level
information transferred is structurally tiny and each region self-normalizes to its
own initial mean. Re-sweeping the history (iteration) re-applies the same weak
coupling and does not help - measured. A joint fit instead enforces every comparison
as a simultaneous constraint, so cross-region information is not diluted.

Model
-----
Each sailor gets one latent strength per season they appear in: ``theta[s, t]``. A node
is a (sailor, season) pair.

* Likelihood: Plackett-Luce over each rated field (a whole race, all boats at once -
  not a pairwise approximation). For a field finishing in order i_1..i_n,

      log P = sum_{k=1..n-1} [ theta_{i_k} - log sum_{m>=k} exp(theta_{i_m}) ]

  This is concave in theta, so with the quadratic prior below the objective is
  strictly convex and has a unique optimum.
* Prior: a Wiener process (random walk) linking a sailor's consecutive seasons,
  ``theta[s,t+1] - theta[s,t] ~ N(0, w^2 * dt)``, plus ``N(0, sigma0^2)`` on each
  sailor's first season. The prior also breaks Plackett-Luce's invariance to adding a
  constant to all strengths, which is what makes disconnected components well posed.

There is no ``beta``. The Plackett-Luce scale is carried by the spread of theta itself
and is therefore *fitted*, which is why this model is calibrated by construction:
P(i beats j) = sigmoid(theta_i - theta_j) exactly.

Hyperparameters ``w`` and ``sigma0`` are chosen by held-out predictive likelihood over
whole regattas, never by hand.
"""

from dataclasses import dataclass
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from config import Config
from regions import teamRegions
import evaluation
import whrClassify

EXCUSED = ["DNS", "BKD", "RDG", "BYE"]

# Season strings sort as s10, f10, s11, ... - two per year.
def seasonIndex(season):
    """Map a season label like 's26' / 'f25' to a monotonic float index."""
    half = 0.0 if season[0] == "s" else 0.5
    return float(int(season[1:])) + half


@dataclass
class WHRData:
    """Fields bucketed by size so the gradient can be fully vectorized."""
    buckets: dict          # n -> (nodeIdx (F,n) int32, fieldId (F,) int32)
    nNodes: int
    nodeSailor: np.ndarray  # int32
    nodeSeason: np.ndarray  # float64, season index
    # Wiener prior edges between a sailor's consecutive seasons
    priorA: np.ndarray      # int32
    priorB: np.ndarray      # int32
    priorDt: np.ndarray     # float64
    firstNode: np.ndarray   # int32, each sailor's earliest node
    nodeLabel: np.ndarray   # str, "<sailorID>\x00<time bucket>" - stable across rebuilds
    rows: pd.DataFrame      # per-row metadata, aligned to rowNode
    rowNode: np.ndarray     # int32
    rowField: np.ndarray    # int32
    fieldRegatta: np.ndarray
    fieldCross: np.ndarray
    nFields: int


def buildWHRData(rootDir="", frFile="racesfr.parquet",
                 postFile="postcalcFRraces.parquet", ratingTypes=("sr",),
                 seasons=None, timeResolution="season", dtFloorDays=0.1,
                 postDf=None):
    """Flatten the history into per-field arrays plus the Wiener prior structure.

    ``timeResolution`` sets what a node is, and therefore how fine the fitted skill
    curve is:

    * ``"season"``  - one node per sailor per season. Fewest parameters.
    * ``"regatta"`` - one node per sailor per regatta.
    * ``"race"``    - one node per sailor per rated field, matching the original WHR
      paper's per-day nodes. This is what makes a genuine per-race rating delta exist,
      since the curve can then move across each race.

    For the regatta and race resolutions the Wiener prior uses elapsed days between a
    sailor's consecutive nodes, so a gap of one weekend is penalised far less than a
    gap of one summer. At season resolution it uses the season index.
    """
    df = pd.read_parquet(rootDir + frFile,
                         columns=["raceID", "adjusted_raceID", "key", "Sailor", "Team",
                                  "Position", "Score", "penalty", "Date", "Regatta",
                                  "raceNum"])
    df["sailorID"] = df["key"].where(
        df["key"].notna() & (df["key"] != "Unknown"),
        df["Sailor"].astype(str) + "-" + df["Team"].astype(str))
    df["sailorID"] = df["sailorID"].replace(Config.merges)

    # The open/womens split is derived here from entrant genders rather than read back
    # from a previous openskill run, which used to make the joint fit depend on that
    # pass being present AND current - a regatta scraped since the last run would have
    # been missing from the mapping and silently dropped. whrClassify reproduces
    # main.calculateAllRegattaInfo exactly (verified to 100% on all 1.5M rows).
    if postDf is not None:
        post = postDf[["raceID", "position", "ratingType"]].drop_duplicates()
        rt = post.set_index(["raceID", "position"])["ratingType"]
        df["ratingType"] = df.set_index(["raceID", "Position"]).index.map(rt)
    else:
        genders = whrClassify.genderMap(rootDir, merges=Config.merges)
        df["ratingType"] = whrClassify.fleetRatingTypes(df, genders)
    df = df[df["ratingType"].isin(ratingTypes)]

    df["season"] = df["raceID"].str.split("/").str[0]
    if seasons:
        df = df[df["season"].isin(seasons)]
    df["region"] = df["Team"].map(teamRegions).replace(evaluation.REGION_MERGE)
    df["field"] = (df["adjusted_raceID"].astype(str) + "|" + df["Position"].astype(str)
                   + "|" + df["ratingType"].astype(str))

    # Drop excused penalties and unscored rows; they carry no ordering information.
    df = df[~df["penalty"].isin(EXCUSED)]
    df = df[np.isfinite(df["Score"].to_numpy(float))]

    # Keep only fields that still have at least two competitors.
    sizes = df.groupby("field")["Score"].transform("size")
    df = df[sizes >= 2]

    # Best finisher first inside each field.
    df["seasonIdx"] = df["season"].map(seasonIndex)
    df["days"] = pd.to_datetime(df["Date"]).astype("int64") / 86_400_000_000_000.0
    df = df.sort_values(["field", "Score"], kind="stable").reset_index(drop=True)

    if timeResolution == "season":
        bucket = df["season"].astype(str)
        timeCol = df["seasonIdx"]
        dtFloor = 0.5
    elif timeResolution == "regatta":
        bucket = df["Regatta"].astype(str)
        timeCol = df["days"]
        dtFloor = dtFloorDays
    elif timeResolution == "race":
        bucket = df["field"].astype(str)
        timeCol = df["days"]
        dtFloor = dtFloorDays
    else:
        raise ValueError(f"unknown timeResolution {timeResolution!r}")

    nodeKey = df["sailorID"].astype(str) + "\x00" + bucket
    nodeLabels, rowNode = np.unique(nodeKey.to_numpy(), return_inverse=True)
    nNodes = len(nodeLabels)

    # Node attributes, taken from the rows that map to each node.
    tmp = pd.DataFrame({"node": rowNode, "sailorID": df["sailorID"].to_numpy(),
                        "t": timeCol.to_numpy()})
    agg = tmp.groupby("node").agg(sailorID=("sailorID", "first"), t=("t", "min"))
    agg = agg.reindex(range(nNodes))
    sailorUniq, nodeSailor = np.unique(agg["sailorID"].to_numpy(), return_inverse=True)
    seasonOf = agg["t"].to_numpy(float)

    # Wiener edges: a sailor's consecutive nodes in time.
    order = np.lexsort((seasonOf, nodeSailor))
    ns, se = nodeSailor[order], seasonOf[order]
    same = ns[1:] == ns[:-1]
    priorA = order[:-1][same].astype(np.int32)
    priorB = order[1:][same].astype(np.int32)
    priorDt = np.maximum((se[1:][same] - se[:-1][same]).astype(float), dtFloor)
    firstIdx = np.concatenate([[True], ~same])
    firstNode = order[firstIdx].astype(np.int32)

    fields, rowField = np.unique(df["field"].to_numpy(), return_inverse=True)
    nFields = len(fields)

    # Bucket fields by competitor count so each bucket is one dense (F, n) array.
    buckets = {}
    fieldSizes = np.bincount(rowField, minlength=nFields)
    for n in np.unique(fieldSizes):
        if n < 2:
            continue
        which = np.where(fieldSizes == n)[0]
        sel = np.isin(rowField, which)
        sub = rowNode[sel]
        # rows are already grouped and ordered by field then score
        fid = rowField[sel]
        orderSub = np.lexsort((np.arange(len(fid)), fid))
        buckets[int(n)] = (sub[orderSub].reshape(-1, int(n)).astype(np.int32),
                           fid[orderSub].reshape(-1, int(n))[:, 0].astype(np.int32))

    regattaOf = df.groupby("field")["Regatta"].first().reindex(fields).to_numpy()
    crossOf = df.groupby("field")["region"].nunique().reindex(fields).to_numpy() > 1

    rows = df[["field", "raceID", "sailorID", "region", "Score", "season", "Position",
               "ratingType", "Regatta", "raceNum"]].rename(
        columns={"Score": "score", "Position": "position", "Regatta": "regatta",
                 "raceNum": "raceNumber"})

    return WHRData(buckets, nNodes, nodeSailor.astype(np.int32), seasonOf,
                   priorA, priorB, priorDt, firstNode, nodeLabels, rows,
                   rowNode.astype(np.int32), rowField.astype(np.int32),
                   regattaOf, crossOf, nFields)


def _plObjective(theta, buckets, activeMask=None):
    """Negative Plackett-Luce log-likelihood and its gradient, vectorized per bucket.

    For a field ordered best-first with strengths E = exp(theta):
        S_k   = sum_{m>=k} E_m                     (suffix sums)
        logL  = sum_{k<n} (theta_k - log S_k)
        dlogL/dtheta_m = 1{m<=n-2} - E_m * sum_{k<=min(m, n-2)} 1/S_k
    which is O(n) per field via one suffix sum and one cumulative sum.

    exp() is taken on theta shifted by the field maximum for stability; Plackett-Luce
    is invariant to a per-field constant, and the shift is added back into log S_k.
    """
    negll = 0.0
    g = np.zeros_like(theta)
    for n, (nodeIdx, fieldId) in buckets.items():
        if activeMask is not None:
            keep = activeMask[fieldId]
            if not keep.any():
                continue
            nodeIdx = nodeIdx[keep]
        t = theta[nodeIdx]                                  # (F, n), best first
        tmax = t.max(axis=1, keepdims=True)
        E = np.exp(t - tmax)
        S = np.cumsum(E[:, ::-1], axis=1)[:, ::-1]          # suffix sums of E
        # log of the true suffix sum restores the shift
        logStrue = np.log(S[:, :n - 1]) + tmax
        negll -= float(np.sum(t[:, :n - 1] - logStrue))

        invS = 1.0 / S[:, :n - 1]
        C = np.cumsum(invS, axis=1)                         # (F, n-1)
        dll = np.empty_like(t)
        dll[:, :n - 1] = 1.0 - E[:, :n - 1] * C
        dll[:, n - 1] = -E[:, n - 1] * C[:, n - 2]
        np.add.at(g, nodeIdx.ravel(), -dll.ravel())
    return negll, g


def negLogPost(theta, data: WHRData, w, sigma0, activeMask=None):
    """Negative log posterior and gradient: -logPL + Wiener prior + initial prior."""
    f, g = _plObjective(theta, data.buckets, activeMask)
    g = g.copy()

    # Wiener prior between consecutive seasons.
    d = theta[data.priorB] - theta[data.priorA]
    prec = 1.0 / (w * w * data.priorDt)
    f += 0.5 * float(np.sum(prec * d * d))
    np.add.at(g, data.priorB, prec * d)
    np.add.at(g, data.priorA, -prec * d)

    # Prior on each sailor's first season, which also pins the global constant.
    t0 = theta[data.firstNode]
    f += 0.5 * float(np.sum(t0 * t0) / (sigma0 * sigma0))
    np.add.at(g, data.firstNode, t0 / (sigma0 * sigma0))
    return f, g


def fitWHR(data: WHRData, w=0.30, sigma0=1.5, activeMask=None, maxiter=4000,
           theta0=None, verbose=True, ftol=1e-11, gtol=1e-7, freeIdx=None):
    """Convex MAP fit by L-BFGS. Returns (theta, result).

    ftol/gtol are much tighter than scipy's defaults on purpose. At the default
    ftol=2.2e-9 L-BFGS stops after ~1400 of the ~3700 iterations this problem needs,
    leaving individual ratings up to 0.086 sd - roughly 35 display points - away from
    the optimum. At 1e-11/1e-7 the worst node is within 0.004 sd (~1.5 points).

    `freeIdx` optimizes only those nodes and holds the rest at their theta0 values.
    Settled history barely moves when new races arrive, so freezing it shrinks the
    parameter count enormously and is what makes a live refresh affordable.
    """
    x0 = np.zeros(data.nNodes) if theta0 is None else theta0.copy()
    t0 = time.time()
    if freeIdx is None:
        res = minimize(negLogPost, x0, args=(data, w, sigma0, activeMask),
                       jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "maxcor": 20,
                                "ftol": ftol, "gtol": gtol})
    else:
        base = x0.copy()
        def reduced(z):
            th = base.copy(); th[freeIdx] = z
            f, g = negLogPost(th, data, w, sigma0, activeMask)
            return f, g[freeIdx]
        res = minimize(reduced, x0[freeIdx], jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "maxcor": 20,
                                "ftol": ftol, "gtol": gtol})
        full = base.copy(); full[freeIdx] = res.x
        res.x = full
    if verbose:
        print(f"  fit w={w:.3f} sigma0={sigma0:.2f}  nll={res.fun:.1f}  "
              f"iters={res.nit}  {'converged' if res.success else res.message}  "
              f"{time.time()-t0:.0f}s", flush=True)
    return res.x, res


def thetaFrame(data: WHRData, theta):
    """Per-row fitted strength, for scoring with the evaluation harness."""
    d = data.rows.copy()
    d["oldMu"] = theta[data.rowNode]
    d["oldSigma"] = 0.0
    return d


def rowCredit(theta, data: WHRData, w, sigma0, activeMask=None):
    """Per-race credit: how much each individual race pulls a sailor's rating.

    At the MAP optimum the total gradient is zero, so every race's likelihood
    gradient is exactly balanced by the prior. The gradient contribution of one race
    to one sailor therefore measures which way that race pushes their rating, and
    dividing by the local curvature converts it into rating units:

        credit_race = (dlogL_race / dtheta_n) / H_nn

    This is the standard influence-function approximation to "how much would this
    sailor's rating move if we dropped this race". It is the joint-fit analogue of the
    sequential model's per-race delta, and unlike that delta it accounts for all the
    information in the fit rather than only what came before.

    Returns an array aligned to ``data.rows``.
    """
    # Per-row likelihood gradient, and the Hessian diagonal, in one pass per bucket.
    rowGrad = np.zeros(len(data.rowNode))
    hDiag = np.zeros(data.nNodes)

    # Rows of a field are contiguous and score-ordered, matching the column order of
    # the bucket arrays, so a field's per-row gradients scatter back by position.
    order = np.lexsort((np.arange(len(data.rowField)), data.rowField))
    sizes = np.bincount(data.rowField, minlength=data.nFields)
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    for n, (nodeIdx, fieldId) in data.buckets.items():
        if activeMask is not None:
            keep = activeMask[fieldId]
            if not keep.any():
                continue
            nodeIdx, fieldId = nodeIdx[keep], fieldId[keep]
        t = theta[nodeIdx]
        tmax = t.max(axis=1, keepdims=True)
        E = np.exp(t - tmax)
        S = np.cumsum(E[:, ::-1], axis=1)[:, ::-1]
        invS = 1.0 / S[:, :n - 1]
        C = np.cumsum(invS, axis=1)
        dll = np.empty_like(t)
        dll[:, :n - 1] = 1.0 - E[:, :n - 1] * C
        dll[:, n - 1] = -E[:, n - 1] * C[:, n - 2]

        # Hessian diagonal: sum_k 1{m in R_k} (p_mk - p_mk^2), p_mk = E_m / S_k
        P = E[:, :, None] * invS[:, None, :]              # (F, n, n-1) = p_{m,k}
        kIdx = np.arange(n - 1)[None, None, :]
        mIdx = np.arange(n)[None, :, None]
        inRisk = (kIdx <= mIdx)                           # m is in risk set R_k iff k <= m
        hcontrib = np.where(inRisk, P - P * P, 0.0).sum(axis=2)
        np.add.at(hDiag, nodeIdx.ravel(), hcontrib.ravel())

        # scatter per-row gradients back to their original row positions
        for f_, fid in enumerate(fieldId):
            s = starts[fid]
            rowGrad[order[s:s + n]] = dll[f_]

    # prior curvature at each node
    prec = 1.0 / (w * w * data.priorDt)
    np.add.at(hDiag, data.priorA, prec)
    np.add.at(hDiag, data.priorB, prec)
    np.add.at(hDiag, data.firstNode, 1.0 / (sigma0 * sigma0))

    return rowGrad / np.maximum(hDiag[data.rowNode], 1e-9)


def buildHessian(theta, data: WHRData, w, sigma0, activeMask=None):
    """Sparse Hessian of the negative log posterior at ``theta`` (the Laplace precision).

    The Plackett-Luce contribution of one field is
        sum_k [ diag(p_.k) - p_.k p_.k^T ]   over the risk set R_k
    with p_mk = exp(theta_m) / S_k, which is PSD. The Wiener prior adds a tridiagonal
    block per sailor and the initial prior adds to the diagonal, which together make
    the matrix positive definite even where the comparison graph is disconnected.

    At regatta resolution this is ~8M nonzeros before coalescing - tractable. At race
    resolution it is ~20x larger, which is one more reason regatta resolution is the
    right granularity.
    """
    import scipy.sparse as sp

    rowsI, colsJ, vals = [], [], []
    for n, (nodeIdx, fieldId) in data.buckets.items():
        if activeMask is not None:
            keep = activeMask[fieldId]
            if not keep.any():
                continue
            nodeIdx = nodeIdx[keep]
        t = theta[nodeIdx]
        tmax = t.max(axis=1, keepdims=True)
        E = np.exp(t - tmax)
        S = np.cumsum(E[:, ::-1], axis=1)[:, ::-1]
        invS = 1.0 / S[:, :n - 1]
        P = E[:, :, None] * invS[:, None, :]              # (F, n, n-1)
        mIdx = np.arange(n)[None, :, None]
        kIdx = np.arange(n - 1)[None, None, :]
        P = np.where(kIdx <= mIdx, P, 0.0)                # zero outside the risk set
        diagPart = P.sum(axis=2)                          # (F, n)
        outer = np.einsum("fmk,fnk->fmn", P, P)           # (F, n, n)
        block = -outer
        di = np.arange(n)
        block[:, di, di] += diagPart
        ii = np.repeat(nodeIdx, n, axis=1).ravel()
        jj = np.tile(nodeIdx, (1, n)).ravel()
        rowsI.append(ii); colsJ.append(jj); vals.append(block.ravel())

    # Wiener prior: +prec on both diagonals, -prec off-diagonal.
    prec = 1.0 / (w * w * data.priorDt)
    a, b = data.priorA, data.priorB
    rowsI += [a, b, a, b]
    colsJ += [a, b, b, a]
    vals += [prec, prec, -prec, -prec]

    # Prior on each sailor's first node.
    rowsI.append(data.firstNode); colsJ.append(data.firstNode)
    vals.append(np.full(len(data.firstNode), 1.0 / (sigma0 * sigma0)))

    H = sp.coo_matrix((np.concatenate(vals),
                       (np.concatenate(rowsI), np.concatenate(colsJ))),
                      shape=(data.nNodes, data.nNodes)).tocsc()
    H.sum_duplicates()
    return H


def contrastSE(H, contrasts, tol=1e-8, maxiter=2000):
    """Standard errors of linear contrasts v' theta, i.e. sqrt(v' H^-1 v).

    Solves H x = v by conjugate gradients with a Jacobi preconditioner rather than
    forming H^-1: the full inverse diagonal would need one solve per node, while the
    rankings only need one solve per sailor of interest.
    """
    import scipy.sparse as sp
    import scipy.sparse.linalg as spl

    d = H.diagonal()
    Minv = spl.LinearOperator(H.shape, matvec=lambda x: x / np.maximum(d, 1e-12))
    out = np.empty(contrasts.shape[1])
    for c in range(contrasts.shape[1]):
        v = contrasts[:, c]
        x, info = spl.cg(H, v, rtol=tol, maxiter=maxiter, M=Minv)
        out[c] = np.sqrt(max(float(v @ x), 0.0))
    return out


def nodeSE(H, nodes, reference=None, **kw):
    """SE of each node's rating relative to a reference combination of nodes.

    ``reference`` is a weight vector over nodes (it should sum to 1); pass the
    race-count-weighted ranked population to get "how well do we know where this
    sailor sits nationally", which is the quantity the outLinks gate was a proxy for.
    """
    nNodes = H.shape[0]
    V = np.zeros((nNodes, len(nodes)))
    V[nodes, np.arange(len(nodes))] = 1.0
    if reference is not None:
        V -= reference[:, None]
    return contrastSE(H, V, **kw)


def anchorScale(theta, nodes, targetMean=1400.0, targetSd=400.0):
    """Affine map from theta (log-odds) to published points, pinned to a population.

    Returned as (scale, offset) with ``display = theta * scale + offset``. Anchoring on
    the currently-ranked population each run is what keeps published numbers stable:
    the joint fit determines theta only up to the scale the likelihood implies, so
    without a fixed anchor the whole distribution drifts slightly between runs and
    every sailor's number moves even when nobody's standing changed.

    Rating *differences* (per-race credit, head-to-head gaps) take ``scale`` only.
    """
    sub = theta[nodes]
    sd = float(np.std(sub))
    if sd <= 0:
        return 1.0, targetMean
    scale = targetSd / sd
    return scale, targetMean - scale * float(np.mean(sub))


@dataclass
class FleetFit:
    ratingType: str
    data: WHRData
    theta: np.ndarray
    credit: np.ndarray        # per row, display points
    rating: np.ndarray        # per row, display points
    scale: float
    offset: float
    w: float
    sigma0: float


def fitFleetRatingType(rootDir="", ratingType="sr", w=0.055, sigma0=3.0,
                       timeResolution="regatta", targetSeasons=("s26", "f26"),
                       maxiter=5000, verbose=True,
                       targetMean=1450.0, targetSd=430.0, postDf=None):
    """Fit one fleet rating graph end to end: theta, per-race credit, display scale."""
    data = buildWHRData(rootDir=rootDir, ratingTypes=(ratingType,),
                        timeResolution=timeResolution, postDf=postDf)
    if verbose:
        print(f"[{ratingType}] nodes {data.nNodes:,} fields {data.nFields:,} "
              f"rows {len(data.rowNode):,}", flush=True)
    theta, _ = fitWHR(data, w=w, sigma0=sigma0, maxiter=maxiter, verbose=verbose)
    credit = rowCredit(theta, data, w, sigma0)

    # Anchor on the sailors who are actually ranked, i.e. the target seasons.
    cur = np.isin(data.rows["season"].to_numpy(), list(targetSeasons))
    anchorNodes = np.unique(data.rowNode[cur]) if cur.any() else np.arange(data.nNodes)
    scale, offset = anchorScale(theta, anchorNodes, targetMean, targetSd)

    return FleetFit(ratingType, data, theta, credit * scale,
                    theta[data.rowNode] * scale + offset, scale, offset, w, sigma0)


FLEET_RATING_TYPES = ("sr", "cr", "wsr", "wcr")


def fitAllFleet(rootDir="", w=0.055, sigma0=3.0, timeResolution="regatta",
                targetSeasons=("s26", "f26"), ratingTypes=FLEET_RATING_TYPES,
                maxiter=2500, verbose=True):
    """Fit every fleet rating graph. Each is independent, so they are fitted separately.

    Team racing (tsr/tcr/wtsr/wtcr) is a different likelihood - 3-boat teams with
    win/loss outcomes rather than a finishing order - and is left on openskill for now.
    """
    return {rt: fitFleetRatingType(rootDir=rootDir, ratingType=rt, w=w, sigma0=sigma0,
                                   timeResolution=timeResolution,
                                   targetSeasons=targetSeasons, maxiter=maxiter,
                                   verbose=verbose)
            for rt in ratingTypes}


def fitWithSE(rootDir="", ratingType="sr", w=0.6, sigma0=3.0,
              targetSeasons=("s26", "f26"), maxiter=1500, verbose=True,
              targetMean=1400.0, targetSd=400.0, postDf=None,
              regattaNoiseFraction=0.17):
    """Season-resolution fit plus posterior SEs for the currently-ranked sailors.

    SEs are computed here rather than at regatta resolution because the quantity is
    about a sailor's *current level*, and the season-resolution Hessian factorizes in
    about a minute where the regatta-resolution one risks a very large LU factor.
    """
    import scipy.sparse.linalg as spl

    data = buildWHRData(rootDir=rootDir, ratingTypes=(ratingType,),
                        timeResolution="season", postDf=postDf)
    theta, _ = fitWHR(data, w=w, sigma0=sigma0, maxiter=maxiter, verbose=verbose)

    rows = data.rows.copy()
    rows["node"] = data.rowNode
    cur = rows[rows["season"].isin(list(targetSeasons))]
    if cur.empty:
        return pd.DataFrame(columns=["sailorID", "rating", "se", "lcb"])
    # One row per SAILOR, not per node. Nodes are (sailor, season), so a sailor active
    # in both target seasons has two of them; take their most recent season as the
    # current rating, or the table gets duplicate entries and loadWHRRatings keeps
    # whichever happens to be read last.
    cur = cur.assign(_t=cur["season"].map(seasonIndex))
    info = (cur.groupby("node")
               .agg(sailorID=("sailorID", "first"), region=("region", "first"),
                    season=("season", "first"), races=("score", "size"),
                    regattas=("regatta", "nunique"),
                    _t=("_t", "first"))
               .reset_index()            # keep `node` as a column, not the index
               .sort_values("_t")
               .groupby("sailorID", as_index=False)
               .last())
    nodes = info["node"].to_numpy()

    ref = np.zeros(data.nNodes)
    wts = info["races"].to_numpy(float)
    ref[nodes] = wts / wts.sum()

    H = buildHessian(theta, data, w, sigma0)
    lu = spl.splu(H)
    ses = np.empty(len(nodes))
    for s in range(0, len(nodes), 400):
        sl = slice(s, min(s + 400, len(nodes)))
        V = np.zeros((data.nNodes, sl.stop - sl.start))
        V[nodes[sl], np.arange(sl.stop - sl.start)] = 1.0
        V -= ref[:, None]
        X = lu.solve(V)
        ses[sl] = np.sqrt(np.maximum((V * X).sum(axis=0), 0.0))

    # Anchor on ALL current-season sailors, not the ranked subset. Anchoring on the
    # ranked subset is circular - the eligibility threshold is in display points, so a
    # larger scale inflates every SE, which shrinks the ranked set, which shrinks the
    # theta spread, which inflates the scale again. That feedback does not settle: it
    # cut the women's skipper list from 203 to 73 on its own. Anchoring on a fixed
    # population makes the scale independent of the threshold, and the threshold is
    # expressed relative to the scale instead (see maxRatingSEFraction).
    scale, offset = anchorScale(theta, nodes, targetMean, targetSd)

    info = info.reset_index(drop=True)
    info = info.drop(columns=[c for c in ("_t", "node") if c in info.columns])
    info["ratingType"] = ratingType
    info["rating"] = theta[nodes] * scale + offset
    # Add the per-regatta form/conditions noise the likelihood does not model. Races
    # inside one regatta are not independent samples of skill, so a one-regatta record
    # carries irreducible uncertainty no number of races can remove.
    seModel = ses * scale
    perRegatta = regattaNoiseFraction * targetSd
    nreg = info["regattas"].to_numpy(float).clip(min=1.0)
    info["se"] = np.sqrt(seModel ** 2 + (perRegatta ** 2) / nreg)
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
    return info


def runFleetPipeline(rootDir="", outDir=None, ratingTypes=FLEET_RATING_TYPES,
                     wCurve=0.055, wLevel=0.6, sigma0=3.0,
                     targetSeasons=("s26", "f26"), verbose=True,
                     anchors=None, targetMean=1400.0, targetSd=400.0, postDf=None,
                     includeTR=True, trSeasons=("s26",), wTR=0.03, sigma0TR=1.0,
                     regattaNoiseFraction=0.17):
    """Produce everything the site needs, for every fleet rating type.

    Two fits per rating type, because they answer different questions:

    * regatta resolution (``wCurve``) -> the published rating curve and the per-race
      credit that replaces the sequential per-race delta
    * season resolution (``wLevel``) -> the posterior SE used for intervals and for
      ranking eligibility, which needs a Hessian that factorizes

    Writes ``whr_races.parquet`` (per-race credit) and ``whr_sailors.parquet``
    (rating, SE, lower confidence bound) and returns both frames.
    """
    outDir = rootDir if outDir is None else outDir
    raceFrames, sailorFrames = [], []
    for rt in ratingTypes:
        if verbose:
            print(f"\n=== {rt} ===", flush=True)
        tmC, tsC = (anchors or {}).get(rt, (targetMean, targetSd))
        fit = fitFleetRatingType(rootDir=rootDir, ratingType=rt, w=wCurve,
                                 sigma0=sigma0, timeResolution="regatta",
                                 targetSeasons=targetSeasons, verbose=verbose,
                                 targetMean=tmC, targetSd=tsC, postDf=postDf)
        rf = raceLadder(fit)
        rf["ratingType"] = rt
        raceFrames.append(rf)

        tm, ts = (anchors or {}).get(rt, (targetMean, targetSd))
        sailorFrames.append(fitWithSE(rootDir=rootDir, ratingType=rt, w=wLevel,
                                      sigma0=sigma0, targetSeasons=targetSeasons,
                                      verbose=verbose, targetMean=tm, targetSd=ts,
                                      postDf=postDf,
                                      regattaNoiseFraction=regattaNoiseFraction))

    if includeTR:
        # Team racing is a different likelihood - 3v3 win/loss, so Bradley-Terry on the
        # combined strength of each side rather than Plackett-Luce over a finishing
        # order. On a like-for-like whole-regatta holdout it beats openskill on every
        # team-race rating type (e.g. tsr accuracy 0.696 vs 0.625).
        import whrTR
        trSailors, trRaces = whrTR.runTRPipeline(
            rootDir=rootDir, w=wTR, sigma0=sigma0TR, targetSeasons=trSeasons,
            targetMean=targetMean, targetSd=targetSd, verbose=verbose)
        if len(trSailors):
            sailorFrames.append(trSailors)
        if len(trRaces):
            raceFrames.append(trRaces)

    races = pd.concat(raceFrames, ignore_index=True)
    sailors = pd.concat(sailorFrames, ignore_index=True)

    # regAvg: the average rating of everyone entered in the regatta, which the site
    # shows next to a sailor's own rating. The openskill pass computed this from its
    # own ratings while sweeping; recompute it from the joint fit so it stays correct
    # without that pass. Averaged across both positions in the regatta, matching
    # main.getRegAvgFR, and a regatta is either fleet or team racing so grouping by
    # regatta alone does not mix the two.
    # Fleet rows carry the regatta's fitted rating as `nodeRating` (raceLadder walks
    # oldRating/newRating within the regatta); team-race rows carry it as `rating`.
    # Coalesce the two, or regAvg is NaN for every fleet row.
    perRegattaRating = races.get("rating")
    if perRegattaRating is None:
        perRegattaRating = races["nodeRating"]
    elif "nodeRating" in races.columns:
        perRegattaRating = perRegattaRating.fillna(races["nodeRating"])
    races["regAvg"] = perRegattaRating.groupby(races["regatta"]).transform("mean")

    races.to_parquet(outDir + "whr_races.parquet")
    sailors.to_parquet(outDir + "whr_sailors.parquet")
    if verbose:
        print(f"\nwrote {len(races):,} race rows and {len(sailors):,} sailor rows")
    return races, sailors


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "../"
    runFleetPipeline(rootDir=root)


def raceLadder(fit: FleetFit):
    """Turn the fitted curve plus per-race credit into a per-race rating ladder.

    The site publishes a per-race rating change, which a joint fit does not natively
    produce - it fits a curve, not a sequence of updates. This walks each sailor
    through each regatta so that

        newRating - oldRating == credit

    for every race, and the walk lands exactly on the regatta's fitted rating. The
    existing FleetScores columns therefore keep their meaning and the front end needs
    no change; `credit` is stored alongside at full precision, because oldRating and
    newRating are INT and rounding would flatten small credits.
    """
    d = fit.data.rows[["raceID", "sailorID", "season", "regatta", "field",
                       "raceNumber", "position", "score"]].copy()
    d["credit"] = fit.credit
    d["nodeRating"] = fit.rating          # constant within a (sailor, regatta) node
    # Raw theta (log-odds) as well: pairwise probabilities are sigmoid(theta_i-theta_j),
    # and with openskill off the oldMu column would otherwise sit at the prior and the
    # evaluation harness would silently measure nothing.
    d["theta"] = fit.theta[fit.data.rowNode]
    d["raceNumber"] = pd.to_numeric(d["raceNumber"], errors="coerce")
    d = d.sort_values(["sailorID", "regatta", "raceNumber", "raceID"], kind="stable")

    g = d.groupby(["sailorID", "regatta"], sort=False)["credit"]
    total = g.transform("sum")
    creditBefore = g.cumsum() - d["credit"]
    d["oldRating"] = d["nodeRating"] - total + creditBefore
    d["newRating"] = d["oldRating"] + d["credit"]

    # Predicted finishing place from the fitted strengths, so `predicted` stays
    # consistent with the ratings actually being published.
    d["predicted"] = (d.groupby("field")["nodeRating"]
                       .rank(ascending=False, method="first").astype(int))
    return d


def warmStart(newData: WHRData, oldData: WHRData, oldTheta):
    """Carry a previous fit's theta onto a rebuilt (larger) node set.

    Node ids are positional and shift whenever races are added, so the mapping goes
    through the stable "<sailorID>\\x00<bucket>" labels. Nodes that did not exist in
    the old fit start at that sailor's most recent known value, falling back to 0.

    This is what makes an incremental refresh cheap: adding one regatta barely moves
    the optimum, so L-BFGS started here converges in a handful of iterations instead
    of the ~1400 a cold start needs.
    """
    prev = dict(zip(oldData.nodeLabel, oldTheta))
    theta = np.zeros(newData.nNodes)
    # fall back to the sailor's latest previous value for brand-new nodes
    latest = {}
    for lbl, t in zip(oldData.nodeLabel, oldTheta):
        latest[lbl.split("\x00", 1)[0]] = t
    hit = 0
    for i, lbl in enumerate(newData.nodeLabel):
        if lbl in prev:
            theta[i] = prev[lbl]; hit += 1
        else:
            theta[i] = latest.get(lbl.split("\x00", 1)[0], 0.0)
    return theta, hit
