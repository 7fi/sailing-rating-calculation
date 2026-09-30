"""Cross-region calibration harness for the sailing ratings.

This is the gate for every change to the rating model. It answers one question:
**are sailors from different conferences placed correctly relative to each other?**

Two protocols, and they are NOT interchangeable:

* ``prequential`` (rolling-origin) - every race is scored from its own pre-race
  ``oldMu``/``oldSigma``, so every race is already an out-of-sample prediction and no
  holdout is needed. This is the correct protocol for comparing forward-only variants
  of the online filter.
* ``regatta holdout`` - hold out whole *regattas* (never individual races or rows, or
  results leak within a regatta). This is the only valid protocol for an iterated
  whole-history estimator, which has no "pre-race" state. A *time* split is wrong
  there: iteration deliberately propagates later information backward, which a time
  split would score as leakage.

The headline metric is a fitted per-region offset ``d_R`` in display points. For every
same-race pair of sailors from different regions we fit

    P(i beats j) = Phi((mu_i + d_R(i) - mu_j - d_R(j)) / sqrt(2*beta^2 + s_i^2 + s_j^2))

with ``sum(d_R) = 0`` for identifiability. ``d_R`` is how many rating points a region
is collectively mis-placed by.

``d_R`` IS A DIAGNOSTIC AND MUST NEVER BE APPLIED TO THE RATINGS. Adding it back would
be exactly the per-conference fudge factor this project rules out. Its only job is to
be a number that region-blind modelling changes drive toward zero.
"""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import ndtr, log_ndtr, expit, log_expit

from config import Config
from regions import teamRegions

# PCCSC and NWICSA are treated as one region for cross-regional purposes, matching
# calculationsFR.updateRaces.
REGION_MERGE = {"NWICSA": "PCCSC"}


# --------------------------------------------------------------------------- loading

def loadPredictionFrame(rootDir="", frFile="postcalcFRraces.parquet",
                        rawFile="racesfr.parquet", seasons=None,
                        ratingTypes=("sr",)):
    """Load post-calc race rows joined to their contemporaneous team and rated field.

    Requires the oldMu/oldSigma/newMu/newSigma columns: a win probability cannot be
    recovered from the display ordinal alone, since
    ``ordinal = alpha*(mu - z*sigma) + target`` is one equation in two unknowns.
    """
    cols = ["raceID", "season", "regatta", "raceNumber", "division", "sailorID",
            "position", "score", "predicted", "ratingType", "scoring", "penalty",
            "oldMu", "oldSigma", "oldRating", "boat", "venue"]
    df = pd.read_parquet(rootDir + frFile, columns=cols)

    missing = {"oldMu", "oldSigma"} - set(df.columns)
    if missing:
        raise ValueError(
            f"{frFile} lacks {sorted(missing)}. Re-run the pipeline after the "
            "updateRaces change that persists raw mu/sigma."
        )

    if ratingTypes:
        df = df[df["ratingType"].isin(ratingTypes)]
    if seasons:
        df = df[df["season"].isin(seasons)]

    # adjusted_raceID is the actual rated field (it merges A/B/C for Combined
    # scoring), so pairs must be formed within it, not within raceID.
    raw = pd.read_parquet(rawFile if rootDir == "" else rootDir + rawFile,
                          columns=["raceID", "key", "Sailor", "Team", "Position",
                                   "adjusted_raceID"])
    raw["sailorID"] = raw["key"].where(
        raw["key"].notna() & (raw["key"] != "Unknown"),
        raw["Sailor"].astype(str) + "-" + raw["Team"].astype(str))
    raw = raw.drop_duplicates(subset=["raceID", "sailorID", "Position"])

    df = df.merge(raw[["raceID", "sailorID", "Position", "Team", "adjusted_raceID"]],
                  left_on=["raceID", "sailorID", "position"],
                  right_on=["raceID", "sailorID", "Position"], how="left")
    df = df.drop(columns=["Position"])

    # The rated field is (adjusted_raceID, position): skippers and crews are rated in
    # separate openskill calls.
    df["field"] = df["adjusted_raceID"].astype(str) + "|" + df["position"].astype(str)
    # postcalc splits the regatta slug out of raceID, dropping the season prefix that
    # racesfr's Regatta column carries. Rebuild the full key so the two can be joined
    # or held out together.
    df["regattaFull"] = df["season"].astype(str) + "/" + df["regatta"].astype(str)
    return df


def attachRegions(df, teamCol="Team"):
    """Map each row's contemporaneous team to its conference."""
    df = df.copy()
    df["region"] = df[teamCol].map(teamRegions).replace(REGION_MERGE)
    return df


# ----------------------------------------------------------------------------- pairs

def buildPairs(df, crossOnly=True, maxPairs=None, seed=0):
    """All within-field sailor pairs, as flat arrays. Ties are dropped.

    ``y = 1`` when sailor i actually beat sailor j (lower finishing place wins).
    """
    d = df.dropna(subset=["region", "oldMu", "oldSigma", "score"])
    d = d.sort_values("field", kind="stable").reset_index(drop=True)

    sizes = d.groupby("field", sort=False).size().to_numpy()
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])

    # Cache triu index patterns per field size; fields are small (median ~14).
    patterns = {}
    ii, jj = [], []
    for n, s in zip(sizes, starts):
        if n < 2:
            continue
        if n not in patterns:
            patterns[n] = np.triu_indices(n, 1)
        a, b = patterns[n]
        ii.append(a + s)
        jj.append(b + s)
    if not ii:
        return pd.DataFrame(columns=["muI", "muJ", "sigI", "sigJ",
                                     "regionI", "regionJ", "y"])
    ii = np.concatenate(ii)
    jj = np.concatenate(jj)

    score = d["score"].to_numpy()
    keep = score[ii] != score[jj]                       # drop ties
    region = d["region"].to_numpy()
    if crossOnly:
        keep &= region[ii] != region[jj]
    ii, jj = ii[keep], jj[keep]

    if maxPairs is not None and len(ii) > maxPairs:
        pick = np.random.default_rng(seed).choice(len(ii), maxPairs, replace=False)
        ii, jj = ii[pick], jj[pick]

    mu = d["oldMu"].to_numpy(float)
    sg = d["oldSigma"].to_numpy(float)
    return pd.DataFrame({
        "muI": mu[ii], "muJ": mu[jj],
        "sigI": sg[ii], "sigJ": sg[jj],
        "regionI": region[ii], "regionJ": region[jj],
        "y": (score[ii] < score[jj]).astype(np.float64),
    })


def pairWinProb(muI, sigI, muJ, sigJ, beta, link="probit"):
    """P(i beats j).

    ``probit`` matches openskill PlackettLuce.predict_win for the 2-team case.
    ``logit`` is exact for a Plackett-Luce strength vector: with strengths exp(theta),
    P(i beats j) = exp(t_i)/(exp(t_i)+exp(t_j)) = sigmoid(t_i - t_j). Pass the joint
    fit's theta as mu with sigma 0 and beta 1.
    """
    if link == "logit":
        return expit((muI - muJ) / beta)
    s = np.sqrt(2.0 * beta ** 2 + sigI ** 2 + sigJ ** 2)
    return ndtr((muI - muJ) / s)


# ------------------------------------------------------------------ region offset fit

@dataclass
class RegionOffsetFit:
    regions: list
    offsets: np.ndarray          # d_R, display points, sums to zero
    stderr: np.ndarray           # SE of d_R, display points
    lrStat: float                # 2*(ll_fit - ll_null)
    df: int                      # R-1
    nPairs: int

    def table(self):
        z = np.divide(self.offsets, self.stderr,
                      out=np.zeros_like(self.offsets), where=self.stderr > 0)
        return (pd.DataFrame({"region": self.regions, "offset_pts": self.offsets,
                              "se_pts": self.stderr, "z": z})
                .set_index("region").sort_values("offset_pts").round(3))

    @property
    def maxAbsZ(self):
        se = np.where(self.stderr > 0, self.stderr, np.inf)
        return float(np.max(np.abs(self.offsets) / se))


def fitRegionOffsets(pairs, beta, alpha, regions=None, link="probit"):
    """Fit one free skill offset per region on cross-region pairs.

    Only cross-region pairs carry information: for a within-region pair the offset
    difference is identically zero.
    """
    p = pairs[pairs["regionI"] != pairs["regionJ"]]
    if regions is None:
        regions = sorted(set(p["regionI"]) | set(p["regionJ"]))
    R = len(regions)
    if R < 2 or len(p) == 0:
        return RegionOffsetFit(regions, np.zeros(R), np.zeros(R), 0.0, 0, len(p))

    idx = {r: k for k, r in enumerate(regions)}
    ri = p["regionI"].map(idx).to_numpy()
    rj = p["regionJ"].map(idx).to_numpy()
    y = p["y"].to_numpy()

    if link == "logit":
        s = np.full(len(p), float(beta))
    else:
        s = np.sqrt(2.0 * beta ** 2 + p["sigI"].to_numpy(float) ** 2
                    + p["sigJ"].to_numpy(float) ** 2)
    offset = (p["muI"].to_numpy(float) - p["muJ"].to_numpy(float)) / s

    # d = M @ theta with M = [[I],[-1...-1]] forces sum(d) = 0.
    M = np.vstack([np.eye(R - 1), -np.ones((1, R - 1))])
    X = (M[ri] - M[rj]) / s[:, None]                     # N x (R-1)

    if link == "logit":
        def negll(theta):
            z = offset + X @ theta
            return -np.sum(y * log_expit(z) + (1.0 - y) * log_expit(-z))

        def grad(theta):
            z = offset + X @ theta
            return -X.T @ (y - expit(z))
    else:
        def negll(theta):
            z = offset + X @ theta
            return -np.sum(y * log_ndtr(z) + (1.0 - y) * log_ndtr(-z))

        def grad(theta):
            z = offset + X @ theta
            pr = np.clip(ndtr(z), 1e-12, 1 - 1e-12)
            phi = np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)
            w = phi / (pr * (1 - pr))
            return -X.T @ ((y - pr) * w)

    res = minimize(negll, np.zeros(R - 1), jac=grad, method="L-BFGS-B")
    theta = res.x

    # Fisher information for a probit model -> SEs.
    z = offset + X @ theta
    if link == "logit":
        pr = np.clip(expit(z), 1e-12, 1 - 1e-12)
        W = pr * (1 - pr)
    else:
        pr = np.clip(ndtr(z), 1e-12, 1 - 1e-12)
        phi = np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)
        W = phi ** 2 / (pr * (1 - pr))
    I = X.T @ (X * W[:, None])
    try:
        covTheta = np.linalg.inv(I)
    except np.linalg.LinAlgError:
        covTheta = np.linalg.pinv(I)
    covD = M @ covTheta @ M.T

    d = M @ theta
    se = np.sqrt(np.clip(np.diag(covD), 0, None))
    lr = 2.0 * (negll(np.zeros(R - 1)) - negll(theta))
    # mu units -> display points
    return RegionOffsetFit(regions, d * alpha, se * alpha, float(lr), R - 1, len(p))


# --------------------------------------------------------------------------- metrics

def logLoss(pairs, beta, link="probit"):
    pr = np.clip(pairWinProb(pairs["muI"], pairs["sigI"], pairs["muJ"],
                             pairs["sigJ"], beta, link=link), 1e-12, 1 - 1e-12)
    y = pairs["y"].to_numpy()
    return float(-np.mean(y * np.log(pr) + (1 - y) * np.log(1 - pr)))


def pairAccuracy(pairs):
    """Fraction of pairs the rating orders correctly. Immune to probability calibration."""
    ahead = (pairs["muI"].to_numpy() > pairs["muJ"].to_numpy()).astype(float)
    return float(np.mean(ahead == pairs["y"].to_numpy()))


def regionCalibration(pairs, beta, link="probit"):
    """Per region: actual win rate vs predicted, over the pairs it appears in.

    Positive ``calib`` means the region wins more often than predicted, i.e. it is
    under-rated.
    """
    pr = pairWinProb(pairs["muI"], pairs["sigI"], pairs["muJ"], pairs["sigJ"], beta,
                     link=link)
    y = pairs["y"].to_numpy()
    rows = pd.concat([
        pd.DataFrame({"region": pairs["regionI"], "y": y, "p": pr,
                      "ahead": pairs["muI"].to_numpy() > pairs["muJ"].to_numpy()}),
        pd.DataFrame({"region": pairs["regionJ"], "y": 1 - y, "p": 1 - pr,
                      "ahead": pairs["muJ"].to_numpy() > pairs["muI"].to_numpy()}),
    ])
    g = rows.groupby("region")
    out = pd.DataFrame({
        "pairs": g.size(),
        "actual_winrate": g["y"].mean(),
        "pred_winrate": g["p"].mean(),
        "model_ahead": g["ahead"].mean(),
    })
    out["calib"] = out["actual_winrate"] - out["pred_winrate"]
    out["order_bias"] = out["actual_winrate"] - out["model_ahead"]
    return out.sort_values("calib").round(4)


def placeBias(df):
    """Legacy metric: mean(actual place - predicted place) on cross-region races.

    Kept for continuity with earlier reporting, but prefer the pair-based metrics:
    within one race sum(predicted) ~ sum(score) by construction, so this is only
    interpretable as a contrast between groups that co-occur in races.
    """
    d = df.dropna(subset=["region"]).copy()
    nreg = d.groupby("field")["region"].transform("nunique")
    c = d[nreg > 1].copy()
    if c.empty:
        return pd.DataFrame()
    c["err"] = c["score"] - c["predicted"]
    host = c.groupby("field")["region"].agg(lambda s: s.value_counts().idxmax())
    c["host"] = c["field"].map(host)
    away = c[c["region"] != c["host"]]
    return pd.concat([
        c.groupby("region")["err"].mean().rename("all_cross"),
        away.groupby("region")["err"].mean().rename("away"),
        c.groupby("region")["err"].size().rename("rows"),
    ], axis=1).round(3)


def report(df, label="", config=None, maxPairs=None):
    """Compute the full gate-metric set for one model variant."""
    config = config or Config()
    beta, alpha = config.model.beta, config.alpha

    allPairs = buildPairs(df, crossOnly=False, maxPairs=maxPairs)
    crossPairs = allPairs[allPairs["regionI"] != allPairs["regionJ"]]
    withinPairs = allPairs[allPairs["regionI"] == allPairs["regionJ"]]

    fit = fitRegionOffsets(crossPairs, beta, alpha)
    out = {
        "label": label,
        "n_rows": int(len(df)),
        "n_pairs": int(len(allPairs)),
        "n_cross_pairs": int(len(crossPairs)),
        "logloss_all": logLoss(allPairs, beta),
        "logloss_cross": logLoss(crossPairs, beta),
        "acc_all": pairAccuracy(allPairs),
        "acc_cross": pairAccuracy(crossPairs),
        "acc_within": pairAccuracy(withinPairs),
        "max_abs_offset_pts": float(np.max(np.abs(fit.offsets))),
        "max_abs_z": fit.maxAbsZ,
        "lr_stat": fit.lrStat,
        "_fit": fit,
        "_calibration": regionCalibration(crossPairs, beta),
        "_placeBias": placeBias(df),
    }
    return out


def compare(baseline, candidate):
    keys = [k for k in baseline if not k.startswith("_") and k != "label"]
    rows = []
    for k in keys:
        b, c = baseline[k], candidate[k]
        rows.append({"metric": k, "baseline": b, "candidate": c,
                     "delta": (c - b) if isinstance(b, (int, float)) else None})
    return pd.DataFrame(rows).set_index("metric")


def printReport(rep):
    print(f"\n{'=' * 72}\n{rep['label']}\n{'=' * 72}")
    print(f"rows {rep['n_rows']:,}   pairs {rep['n_pairs']:,}   "
          f"cross-region pairs {rep['n_cross_pairs']:,}")
    print(f"\npairwise accuracy   all {rep['acc_all']:.4f}   "
          f"within-region {rep['acc_within']:.4f}   cross-region {rep['acc_cross']:.4f}")
    print(f"pairwise log-loss   all {rep['logloss_all']:.4f}   "
          f"cross-region {rep['logloss_cross']:.4f}")
    print(f"\nfitted region offsets (display points, sum to zero)")
    print(rep["_fit"].table().to_string())
    print(f"\n  max |offset| = {rep['max_abs_offset_pts']:.2f} pts"
          f"   max |z| = {rep['max_abs_z']:.1f}"
          f"   LR({rep['_fit'].df}) = {rep['lr_stat']:.1f}")
    print(f"\nper-region calibration on cross-region pairs")
    print(rep["_calibration"].to_string())
    print(f"\nlegacy place bias")
    print(rep["_placeBias"].to_string())


if __name__ == "__main__":
    import sys
    rootDir = sys.argv[1] if len(sys.argv) > 1 else "../"
    seasons = ["f24", "s25", "f25", "s26"]
    df = attachRegions(loadPredictionFrame(rootDir=rootDir, seasons=seasons))
    printReport(report(df, label=f"sr | {'/'.join(seasons)} | {rootDir}"))
