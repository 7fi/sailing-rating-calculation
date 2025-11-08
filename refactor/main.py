from openskill.models import PlackettLuce
from dataclasses import dataclass

class Sailor:
    def __init__(self, name, key, year, links, teams, seasons, races, rivals, sr, cr, wsr, wcr, tsr, tcr, wtsr, wtcr, gender=""):
        self.name = name
        self.key = key
        self.gender = gender
        self.year = year
        self.links = links
        self.teams = teams

        self.cross = 0
        self.outLinks = 0

        self.skipperRank = 0
        self.crewRank = 0
        self.womenSkipperRank = 0
        self.womenCrewRank = 0

        self.skipperRankTR = 0
        self.crewRankTR = 0
        self.womenSkipperRankTR = 0
        self.womenCrewRankTR = 0

        self.seasons = seasons
        self.races = races
        self.rivals = rivals

        # fleet racing
        self.sr = sr
        self.cr = cr
        self.wsr = wsr
        self.wcr = wcr
        # Team racing
        self.tsr = tsr
        self.tcr = tcr
        self.wtsr = wtsr
        self.wtcr = wtcr

        self.avgSkipperRatio = 0
        self.avgCrewRatio = 0
        
        self.config = Config()
        
    def getRating(self, position, type, womens):
        if position.lower() == 'skipper':
            if type.lower() == 'fleet':
                if womens:
                    return self.wsr
                return self.sr
            elif type.lower() == 'team':
                if womens:
                    return self.wtsr
                return self.tsr
        elif position.lower == 'crew':
            if type.lower() == 'fleet':
                if womens:
                    return self.wcr
                return self.cr
            elif type.lower() == 'team':
                if womens:
                    return self.wtcr
                return self.tcr
            
      
    def hasTargetSeasons(self, targetSeasons, pos):
        return not set([s[0] for s in self.seasons[pos]]).isdisjoint(targetSeasons)

    def __repr__(self):
        return f"{self.name}: {self.teams}, {str(self.sr.ordinal(target=self.config.targetElo, alpha=self.config.alpha))} {str(self.cr.ordinal(target=self.config.targetElo, alpha=self.config.alpha))} {self.seasons} {len(self.races)}"

@dataclass
class Config:
    targetElo : int = 1000
    model : PlackettLuce = PlackettLuce(beta=25.0/120.0)
    alpha : float = 200/model.sigma
    targetSeasons : list[str] = ['f25']
    targetTRSeasons : list[str] = ['s25']
    gradCutoff : int = 2025
    merges: dict[str, str] = {
        'carter-anderson-2027': 'carter-anderson',
        'elliott-bates-2021': 'elliott-bates',
        'ian-hopkins-guerra-2026': 'ian-hopkins-guerra',
        'connor-nelson-2024': 'connor-nelson',
        'Gavin Hudson-Northeastern': 'gavin-hudson',
        'Jeremy Bullock-Northeastern': 'jeremy-bullock',
        'Emma Cole-Northeastern': 'emma-cole',
        'Nathalie Caudron-Northeastern': 'nathalie-caudron'
    }


if __name__ == "__main__":
  print("hi")