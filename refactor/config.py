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
        'Winter vetrone-Northeastern': 'winter-vetrone'
    }
    numTops : ClassVar[dict[str, int]] = {'fr': {'open': 3, 'womens': 2}, 'tr': {'open': 3, 'womens': 3}}

    # Number of out-of-region opponents a sailor must have faced to be officially
    # ranked. Read by Sailor.isRankEligible.
    requiredOutLinks : int = 150

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
    whrSigma0 : float = 3.0      # prior sd on a sailor's first rating
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

    # Individual leaderboards rank by the interval lower bound, so nobody tops the
    # chart off a handful of races. Team ratings average six sailors and use the point
    # estimate instead - see the note in Teams.getOrderedSailors.
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