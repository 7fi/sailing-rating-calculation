from dataclasses import dataclass, field
from openskill.models import PlackettLuce
from typing import ClassVar

@dataclass
class Config:
    targetElo : int = 1000
    model: PlackettLuce = field(default_factory=lambda: PlackettLuce(beta=25.0/120.0))
    alpha : float = 200 / (25.0 / 3.0)
    targetSeasons : ClassVar[list[str]] = ['s26', 'f26']
    targetTRSeasons : ClassVar[list[str]] = ['s26']
    gradCutoff : int = 2026
    merges: ClassVar[dict[str, str]] = {
        'carter-anderson-2027': 'carter-anderson',
        'elliott-bates-2021': 'elliott-bates',
        'ian-hopkins-guerra-2026': 'ian-hopkins-guerra',
        'connor-nelson-2024': 'connor-nelson',
        'Gavin Hudson-Northeastern': 'gavin-hudson',
        'Jeremy Bullock-Northeastern': 'jeremy-bullock',
        'Emma Cole-Northeastern': 'emma-cole',
        'emma-cole-2026': 'emma-cole',
        'kaelyn-holmes-2029': 'kaelyn-holmes',
        'Nathalie Caudron-Northeastern': 'nathalie-caudron', 
        'olivia-figley-2026': 'olivia-figley',
        'Winter vetrone-Northeastern': 'winter-vetrone',
        'pierce-olsen-2029' : "pierce-olsen",
        # Safe to re-enable now that handleMerges tolerates a missing merge target.
        # 'fynn-olsen-2029' : 'fynn-olsen',
        'marcus-abate-2020' : 'marcus-abate'
    }
    numTops : ClassVar[dict[str, int]] = {'fr': {'open': 3, 'womens': 2}, 'tr': {'open': 3, 'womens': 3}}

    # Number of out-of-region opponents a sailor must have faced to be officially
    # ranked. Read by Sailor.isRankEligible.
    requiredOutLinks : int = 75   # fallback only; the SE gate governs when a fit exists

    # Iterating the openskill sweep. Measured NOT to fix the cross-region bias (it is a
    # magnitude-of-level-transfer problem, not a direction-of-time one), so these stay
    # at the no-op defaults. Kept because iteration does improve ordering slightly.
    epochs : int = 1
    epochShrink : float = 1.0    # rho: shrink mu toward the prior mean between epochs
    epochDamping : float = 1.0   # eta: damp the epoch-to-epoch mu step
    sigmaFloor : float | None = None

    # Joint Whole-History Rating fit (refactor/whr.py). This is what actually removes
    # the cross-region bias: PCCSC's fitted region offset goes from -84 display points
    # (z = -17) under openskill to within noise of zero.
    useWHR : bool = True        # False keeps the openskill ratings authoritative
    # When False the openskill rating math is skipped entirely: the race sweep still
    # runs to build the score rows (partners, penalties, ratio, cross-region links),
    # but no rate()/predict_rank() call is made and every rating column is filled in
    # from the joint fit afterwards. The openskill code stays in place and comes back
    # by setting this True.
    runOpenskill : bool = False
    runWHRFit : bool = True      # run the fit in main(); False reuses whr_*.parquet
    whrWCurve : float = 0.055    # Wiener drift per day, regatta-resolution curve
    whrWLevel : float = 0.6      # Wiener drift per season, level fit used for SEs
    # Prior sd on a sailor's first rating. Chosen on overall held-out log-loss (tied
    # best at 2.0) with the cross-region offset spread as the tie-break.
    #
    # Do NOT tighten this to suppress thin records. Shrinkage toward the global mean
    # bites hardest on sailors with the least data, and race volume correlates with
    # region - PCCSC sailors average 13 races and 2 regattas a season against NEISA's
    # 9 and 1 - so tightening sigma0 quietly shifts the leaderboard toward the West
    # Coast. Measured, NEISA+MAISA share of the top 100 falls from 78 at sigma0=3.0 to
    # 72 at 0.75, and the region-offset spread rises from 0.18 to 0.38.
    #
    # Thin records are handled by whrRegattaNoiseFraction + the SE gate instead, which
    # key off how much a record actually pins a sailor down (weak 7-boat fleet,
    # undefeated -> SE 347) rather than off how much they raced.
    whrSigma0 : float = 3.0
    whrWTR : float = 0.03        # team racing drift, tuned separately on held-out regattas
    whrSigma0TR : float = 1.0
    # Display scale. Anchored on ALL current-season sailors, which keeps the scale
    # independent of the eligibility threshold; anchoring on the ranked subset instead
    # is circular and does not settle. Published numbers land close to today's
    # (ranked open-skipper mean ~1480 vs ~1570 now), and ranks are unaffected by scale.
    whrTargetMean : float = 1400.0
    whrTargetSd : float = 400.0
    whrAnchor : ClassVar[dict[str, tuple[float, float]]] = {}   # optional per-type override
    # Eligibility: "we know where this sailor sits nationally to within N points".
    # Replaces requiredOutLinks, which measured connectivity backwards - PCCSC had the
    # lowest pass rate (10.5%) and the *lowest* median SE (best-determined) of any
    # conference. At 100 points, 79% of sailors are rankable vs 24% under outLinks.
    # Expressed as a fraction of whrTargetSd so it is scale-free: 0.25 * 400 = 100
    # points. maxRatingSE is derived, so changing the display scale cannot silently
    # change who qualifies.
    maxRatingSEFraction : float = 0.25

    # A sailor's races within one regatta share a weekend, a venue, a boat and a
    # partner, so they are not independent samples of skill. Measured regatta-to-
    # regatta scatter of a sailor's own rating is ~68 points on a 400-point spread,
    # i.e. 0.17 sd; that much form/conditions noise is irreducible per regatta and the
    # likelihood does not model it. It is added to each SE as
    #     se_total^2 = se_model^2 + (frac*targetSd)^2 / nRegattas
    # which barely touches a sailor with many regattas and substantially widens a
    # sailor with one.
    whrRegattaNoiseFraction : float = 0.0
    # Two-level empirical-Bayes shrinkage: sailor -> team -> global, instead of every
    # sailor toward the global mean independently. See whr._hierarchicalShrink. Nesting
    # inside the school stops a concentrated roster beating a deep one purely because
    # each of the deep roster's thin records was dragged to average in isolation.
    # Measured: Michigan 26 -> 33, Boston University 33 -> 31, top-30 regional mix
    # unchanged. Volume-neutral at team level, which every per-sailor rule was not.
    hierarchicalShrinkage : bool = True

    # Minimum distinct regattas before a sailor appears on an official leaderboard.
    # Left OFF: it is a volume rule, and volume correlates with region. At 2 it cut
    # NEISA+MAISA from 72 to 57 of the eligible top 100 while PCCSC ended up with the
    # LARGEST eligible pool despite NEISA having more sailors - the same failure mode
    # as the old outLinks gate. The SE gate covers the same ground without the bias.
    minRegattasRanked : int = 0

    # Individual leaderboards rank by the interval lower bound, so nobody tops the
    # chart off a handful of races. Team ratings average six sailors and use the point
    # estimate instead - see the note in Teams.getOrderedSailors.
    # Which statistic the leaderboards sort and publish on.
    #
    #   'lcb'    rating - 1.96*SE      conservative; what the first WHR release used
    #   'rating' the raw point estimate
    #   'shrunk' empirical-Bayes posterior mean
    #
    # Individual rankings are back on 'lcb' and the raw rating: that combination is the
    # one that read as most accurate in review, and the later "improvements" (shrinkage,
    # tighter sigma0, a regatta minimum) each optimised a proxy metric while making the
    # published list worse. Team ratings keep 'shrunk', which did measurably and
    # visibly improve them.
    # 'rating', not 'lcb': ranking on the lower bound while publishing the point
    # estimate puts a sailor showing 1900 below one showing 1850, which is the
    # objection RatingSystem.md raises. With the SE gate active the two order almost
    # identically (spearman 0.9964, mean 20 ranks, vs 45 ranks ungated) - the gate
    # does the work, not the subtraction.
    rankingStatistic : str = 'rating'
    publishedStatistic : str = 'rating'
    teamRatingStatistic : str = 'shrunk'
    # Team racing uses the UNSHRUNK rating, and must. A 3v3 match only ever observes
    # the combined strength of three sailors, so an individual theta has SE ~= 1.00 sd
    # while the mean of three is ~0.36 sd - the aggregate is identified even though the
    # split is not. Empirical-Bayes shrinkage is computed per sailor, and for TR
    # var(rating) < mean(se^2), so tau2 clamps to its floor and the shrink factor is
    # ~4e-12: every sailor collapses onto the population mean and every team published
    # exactly 1400. Shrinking individuals first destroys precisely the information the
    # top-N sum would have recovered.
    teamRatingStatisticTR : str = 'rating'

    teamRatingUseLowerBound : bool = False

    @property
    def maxRatingSE(self) -> float:
        return self.maxRatingSEFraction * self.whrTargetSd

    # frfile = 'racesfrtest.parquet'
    frfile = 'racesfr.parquet'
    trfile = 'racesTR.parquet'
    trSailorInfoFile = 'trSailorinfoAll.json'
    
    sailorInfoFile = 'sailor_data2.parquet'
    
    doScrape : bool = False
    calcAll : bool = True
    doUpload : bool = True