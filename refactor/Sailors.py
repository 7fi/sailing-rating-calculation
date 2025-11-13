from config import Config
from openskill.models import PlackettLuceRating, PlackettLuce
from dataclasses import dataclass
from typing import ClassVar
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np
from datetime import date

@dataclass
class Sailor:
    name : str
    key : str
    gender : str
    year : str
    links : str
    teams : str

    seasons : dict[str, list[dict]]
    races : list[dict]
    rivals : dict
    
    # fleet racing
    sr : PlackettLuceRating
    cr : PlackettLuceRating
    wsr : PlackettLuceRating
    wcr : PlackettLuceRating
    # Team racing
    tsr : PlackettLuceRating
    tcr : PlackettLuceRating
    wtsr : PlackettLuceRating 
    wtcr : PlackettLuceRating 
    
    cross : int = 0
    outLinks : int = 0

    skipperRank : int = 0
    crewRank : int = 0
    womenSkipperRank : int = 0
    womenCrewRank : int = 0

    skipperRankTR : int = 0
    crewRankTR : int = 0
    womenSkipperRankTR : int = 0
    womenCrewRankTR : int = 0

    avgSkipperRatio : int = 0
    avgCrewRatio : int = 0
        
    def getRating(self, position : str, raceType : str, womens: bool):
        pos = position.lower()
        typ = raceType.lower()
        if pos not in ('skipper', 'crew') or typ not in ('fleet', 'team'):
            raise ValueError(f"invalid position/type: {position}/{raceType}")
        
        prefix = 'w' if womens else ''
        prefix += 't' if typ == 'team' else ''
        part = 's' if pos == 'skipper' else 'c'
        return getattr(self, f"{prefix}{part}r")
      
    def hasTargetSeasons(self, targetSeasons, pos):
        return not set([s[0] for s in self.seasons[pos]]).isdisjoint(targetSeasons)
    
    def isRankEligible(self, targetSeasons, pos, gradCutoff, needsOutlinks=True):
        return not (self.hasTargetSeasons(targetSeasons, pos) # has target seasons
                        # and has 70 outlinks   
                        and sum([race['outLinks'] 
                                for race in self.races if 'outLinks' in race.keys()]) > 70 if needsOutlinks else True
                        # and graduates after the cutoff
                        and (2000 + int(self.year.split()[0]) > gradCutoff 
                                if isinstance(self.year, str) and len(self.year.split()) > 1 else int(self.year) > gradCutoff))
    def resetRanks(self):
        self.skipperRank = 0
        self.crewRank = 0
        self.womenSkipperRank = 0
        self.womenCrewRank = 0
        self.skipperRankTR = 0
        self.crewRankTR = 0
        self.womenSkipperRankTR = 0
        self.womenCrewRankTR = 0
        
    def __repr__(self):
        return f"{self.name}: {self.teams}, {str(self.sr.ordinal(target=self.config.targetElo, alpha=self.config.alpha))} {str(self.cr.ordinal(target=self.config.targetElo, alpha=self.config.alpha))} {self.seasons} {len(self.races)}"
    


def make_sailor(args):
    key, link, name, first_name, last_name, gender, year, teamLink, team, id, external_id = args
    model = PlackettLuce(beta=25.0 / 120.0)
    rating = model.rating
    return key, Sailor(
        name, key, gender, year,
        [link], [team],
        seasons={'skipper': [], 'crew': []},
        races=[], rivals={},
        sr=rating(), cr=rating(), wsr=rating(), wcr=rating(),
        tsr=rating(), tcr=rating(), wtsr=rating(), wtcr=rating()
    )

def setupPeople(df_sailor_ratings, df_sailor_info, config: Config):

    # read from file first
    if not config.calcAll:
        raise NotImplementedError
        people = {row.key: Sailor(row.Sailor, row.key, row.GradYear, row.Links, row.Teams, row.Seasons, row.Races, row.Rivals,
                                  config.model.create_rating(
                                      [row.srMU, row.srSigma], row.key),
                                  config.model.create_rating(
                                      [row.crMU, row.crSigma], row.key),
                                  config.model.create_rating(
                                      [row.wsrMU, row.wsrSigma], row.key),
                                  config.model.create_rating(
                                      [row.wcrMU, row.wcrSigma], row.key),
                                  config.model.create_rating(
                                      [row.tsrMU, row.tsrSigma], row.key),
                                  config.model.create_rating(
                                      [row.tcrMU, row.tcrSigma], row.key),
                                  config.model.create_rating(
                                      [row.wtsrMU, row.wtsrSigma], row.key),
                                  config.model.create_rating(
                                      [row.wtcrMU, row.wtcrSigma], row.key),
                                  gender=row.gender) for row in df_sailor_ratings.itertuples()}

        for key in df_sailor_info['key']:
            if key not in people.keys():
                row = df_sailor_info.loc[df_sailor_info['key'] == key].iloc[0]
                people[key] = Sailor(row.name, row.key, row.year, [row.link], [row.team], {'skipper': [], 'crew': []}, [], {},
                                     config.model.rating(),
                                     config.model.rating(),
                                     config.model.rating(),
                                     config.model.rating(),
                                     config.model.rating(),
                                     config.model.rating(),
                                     config.model.rating(),
                                     config.model.rating(),
                                     gender=row.gender)
    else: # Set up people dictionary
        rows = list(df_sailor_info.itertuples(index=False, name=None))
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(make_sailor, rows))
        return dict(results)

def handleMerges(df_races, people, config : Config):
    # merge sailor objects
    for oldkey, newkey in config.merges.items():
        if oldkey in people.keys():
            new = people[newkey]
            old = people[oldkey]
            new.links = new.links + old.links
            if old.teams != new.teams:
                new.teams = new.teams + old.teams
            del people[oldkey]
            
            df_races['Link'] = df_races['Link'].replace(oldkey, newkey)
            df_races['key'] = df_races['key'].replace(oldkey, newkey)
    return people, df_races
        
def validPerson(p, type, config: Config):
    # print((2000 + int(p.year.split()[0]) if isinstance(p.year, str) and len(p.year.split()) > 1 else int(p.year)))
    return (p.cross > 20
            and p.outLinks > 70
            # if sum([race['cross'] for race in p.races if 'cross' in race.keys()]) > 20
            # and sum([race['outLinks'] for race in p.races if 'outLinks' in race.keys()]) > 70
            and not p.hasTargetSeasons(config.targetSeasons, type)
            # and (2000 + int(p.year.split()[0]) > gradCutoff if isinstance(p.year, str) and len(p.year.split()) > 1 else int(p.year) > gradCutoff)
            # and sum([p['raceCount'][seas] for seas in targetSeasons if seas in p['raceCount'].keys()]) > 5
            )


def outputSailorsToFile(people, config: Config ):
    allRows = []
    for sailor, p in people.items():
        avgSkipperRatio = float(np.array(
            [r['ratio'] for r in p.races if r['pos'] == 'Skipper' and 'ratio' in r.keys()]).mean())
        avgCrewRatio = float(np.array(
            [r['ratio'] for r in p.races if r['pos'] == 'Crew' and 'ratio' in r.keys()]).mean())
        p.avgSkipperRatio = avgSkipperRatio
        p.avgCrewRatio = avgCrewRatio

        allRows.append([p.name, len(p.races), sailor,
                        p.skipperRank, p.crewRank, p.womenSkipperRank,
                        p.womenCrewRank, p.skipperRankTR, p.womenSkipperRankTR,
                        p.teams,
                        p.gender,
                        p.sr.ordinal(target=config.targetElo,
                                     alpha=config.alpha),
                        p.cr.ordinal(target=config.targetElo,
                                     alpha=config.alpha),
                        p.wsr.ordinal(target=config.targetElo,
                                      alpha=config.alpha),
                        p.wcr.ordinal(target=config.targetElo,
                                      alpha=config.alpha),
                        p.tsr.ordinal(target=config.targetElo,
                                      alpha=config.alpha),
                        p.tcr.ordinal(target=config.targetElo,
                                      alpha=config.alpha),
                        p.wtsr.ordinal(target=config.targetElo,
                                       alpha=config.alpha),
                        p.wtcr.ordinal(target=config.targetElo,
                                       alpha=config.alpha),
                        p.sr.mu, p.sr.sigma,
                        p.cr.mu, p.cr.sigma,
                        p.wsr.mu, p.wsr.sigma,
                        p.wcr.mu, p.wcr.sigma,
                        p.tsr.mu, p.tsr.sigma,
                        p.tcr.mu, p.tcr.sigma,
                        p.wtsr.mu, p.wtsr.sigma,
                        p.wtcr.mu, p.wtcr.sigma,
                        sum([race['outLinks']
                            for race in p.races if 'outLinks' in race.keys()]),
                        p.year, p.links,
                        p.seasons,
                        sum([race['cross']
                            for race in p.races if 'cross' in race.keys()]),
                        p.races, p.rivals, avgSkipperRatio, avgCrewRatio])

    df_sailors = pd.DataFrame(allRows, columns=['Sailor', 'numRaces', 'key', 'SkipperRank', 'CrewRank', 'WomenSkipperRank', 'WomenCrewRank', 'TRSkipperRank', 'TRWomenSkipperRank', 'Teams', 'gender',
                                                'srOrd',
                                                'crOrd',
                                                'wsrOrd',
                                                'wcrOrd',
                                                'tsrOrd',
                                                'tcrOrd',
                                                'wtsrOrd',
                                                'wtcrOrd',
                                                'srMU', 'srSigma',
                                                'crMU', 'crSigma',
                                                'wsrMU', 'wsrSigma',
                                                'wcrMU', 'wcrSigma',
                                                'tsrMU', 'tsrSigma',
                                                'tcrMU', 'tcrSigma',
                                                'wtsrMU', 'wtsrSigma',
                                                'wtcrMU', 'wtcrSigma',
                                                'outLinks', 'GradYear', 'Links',
                                                'Seasons', 'Cross', 'Races',  'Rivals', 'skipperAvgRatio', 'crewAvgRatio'])

    df_sailors.to_json(f'sailors-{date.today().strftime("%Y%m%d")}.json', index=False)
    df_sailors.to_json(f'sailors-latest.json', index=False)
    df_sailors = df_sailors.sort_values(
        by='numRaces', ascending=False).reset_index(drop=True)
    
    
def calculateSailorRanks(people : dict[str,Sailor], config : Config):
    eligible_skippers = [p for p in people.values()
                         if p.isRankEligible(config.targetSeasons, 'skipper', config.gradCutoff)]
    
    eligible_crews = [p for p in people.values()
                      if p.isRankEligible(config.targetSeasons, 'crew', config.gradCutoff)]

    # TODO: Count tr and fr seasons seperately
    eligible_skippers_tr = [p for p in people.values()
                            if p.isRankEligible(config.targetTRSeasons, 'skipper', config.gradCutoff, False)]
    eligible_crews_tr = [p for p in people.values()
                         if p.isRankEligible(config.targetTRSeasons, 'crew', config.gradCutoff, False)]

    for p in people.values():
        p.resetRanks()

    for i, s in enumerate(sorted([p for p in eligible_skippers if p.sr.mu != config.model.mu], key=lambda p: p.sr.ordinal(), reverse=True)):
        s.skipperRank = i + 1
    for i, s in enumerate(sorted([p for p in eligible_crews if p.cr.mu != config.model.mu], key=lambda p: p.cr.ordinal(), reverse=True)):
        s.crewRank = i + 1

    for i, s in enumerate(sorted([p for p in eligible_skippers if p.wsr.mu != config.model.mu], key=lambda p: p.wsr.ordinal(), reverse=True)):
        s.womenSkipperRank = i + 1
    for i, s in enumerate(sorted([p for p in eligible_crews if p.wcr.mu != config.model.mu], key=lambda p: p.wcr.ordinal(), reverse=True)):
        s.womenCrewRank = i + 1

    for i, s in enumerate(sorted([p for p in eligible_skippers_tr if p.tsr.mu != config.model.mu], key=lambda p: p.tsr.ordinal(), reverse=True)):
        s.skipperRankTR = i + 1
    for i, s in enumerate(sorted([p for p in eligible_crews_tr if p.tcr.mu != config.model.mu], key=lambda p: p.tcr.ordinal(), reverse=True)):
        s.crewRankTR = i + 1

    for i, s in enumerate(sorted([p for p in eligible_skippers_tr if p.wtsr.mu != config.model.mu], key=lambda p: p.wtsr.ordinal(), reverse=True)):
        s.womenSkipperRankTR = i + 1
    for i, s in enumerate(sorted([p for p in eligible_crews_tr if p.wtcr.mu != config.model.mu], key=lambda p: p.wtcr.ordinal(), reverse=True)):
        s.womenCrewRankTR = i + 1
    
    return people