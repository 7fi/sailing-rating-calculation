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
        # 'fynn-olsen-2029' : 'fynn-olsen',
        'marcus-abate-2020' : 'marcus-abate'
    }
    numTops : ClassVar[dict[str, int]] = {'fr': {'open': 3, 'womens': 2}, 'tr': {'open': 3, 'womens': 3}}
    # Cross-region comparison threshold for rank eligibility, and the only place it is
    # set. It read 120 here but went unused while Sailors.isRankEligible hardcoded 150 -
    # against a sailor.outLinks that was double-counted, so the bar actually applied was
    # 75. Kept at 75 to leave the published leaderboard unchanged now that outLinks is
    # counted once.
    requiredOutLinks : int = 75
    
    # frfile = 'racesfrtest.parquet'
    frfile = 'racesfr.parquet'
    trfile = 'racesTR.parquet'
    trSailorInfoFile = 'trSailorinfoAll.json'
    
    sailorInfoFile = 'sailor_data2.parquet'
    
    doScrape : bool = False
    calcAll : bool = True
    doUpload : bool = False