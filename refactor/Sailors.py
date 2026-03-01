from config import Config
from openskill.models import PlackettLuceRating, PlackettLuce
from dataclasses import dataclass, field
from typing import ClassVar
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np
from datetime import date
from functools import partial

@dataclass
class Sailor:
    name : str
    key : str
    gender : str
    year : str
    links : str
    teams : list[str]

    seasons : dict[str, list[tuple[str, str]]]
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
    
    ratingTypesReset : list[str] = field(default_factory=list)
        
    def getRating(self, position : str, raceType : str, womens: bool, ordinal : bool = False, config : Config = None):
        pos = position.lower()
        typ = raceType.lower()
        if pos not in ('skipper', 'crew') or typ not in ('fleet', 'team'):
            raise ValueError(f"invalid position/type: {position}/{raceType}")
        
        prefix = 'w' if womens else ''
        prefix += 't' if typ == 'team' else ''
        part = 's' if pos == 'skipper' else 'c'
        ratingObj = getattr(self, f"{prefix}{part}r")
        if ordinal and Config is not None:
            return ratingObj.ordinal(target=config.targetElo,
                                       alpha=config.alpha)
        return ratingObj
    
    def getSeasonRaceCount(self, season, pos):
        return len([r for r in self.races if r['raceID'].split("/")[0] == season and r['pos'].lower() == pos.lower()])
    
    def isOnTeamInSeasons(self, team, seasons):
        return team in [t for [s, t] in self.seasons['skipper'] if s in seasons] or team in [t for [s, t] in self.seasons['crew'] if s in seasons]
      
    def hasTargetSeasons(self, targetSeasons, pos):
        seasonsSet = set([s[0] for s in self.seasons[pos]])
        return not seasonsSet.isdisjoint(targetSeasons)
    
    def getOutLinks(self):
        return sum([race['outLinks'] for race in self.races if 'outLinks' in race.keys()])
    
    def getCrossLinks(self):
        return len([race for race in self.races if 'cross' in race.keys() and race['cross']])
    
    def isRankEligible(self, targetSeasons, pos, gradCutoff, needsOutlinks=True):
        if self.year is None or self.year == "?? *":
            # print(f"{self.key} has none year")
            return False
        
        try:
            if isinstance(self.year, str) and len(self.year.split()) > 1:
                betterYear = 2000 + int(self.year.split()[0])  
            elif isinstance(self.year, str) and self.year.isnumeric():
                betterYear = int(self.year)
            elif isinstance(self.year, int) and self.year > 2000:
                betterYear = self.year
            else: 
                return False
        except ValueError as e:
            print(self.year)
            print(e)
            print(f"error happened to {self.key}")
            return False
        
        return (self.hasTargetSeasons(targetSeasons, pos) # has target seasons
                        and self.getOutLinks() > 70 if needsOutlinks else True # and has 70 outlinks   
                        and betterYear > gradCutoff) # and graduates after the cutoff
        
    def resetRanks(self):
        self.skipperRank = 0
        self.crewRank = 0
        self.womenSkipperRank = 0
        self.womenCrewRank = 0
        self.skipperRankTR = 0
        self.crewRankTR = 0
        self.womenSkipperRankTR = 0
        self.womenCrewRankTR = 0
        
    def resetRatingToBeforeDate(self, resetDate, ratingType):
        if ratingType not in self.ratingTypesReset:
            self.ratingTypesReset.append(ratingType)
        
        resetDate = resetDate.timestamp()
        
        for pos in ['s', 'c']:
            newRT = 'w' if 'w' in ratingType else '' + 't' if 't' in ratingType else '' + pos + 'r'
            racesBeforeReset = [r for r in self.races if r['date'] < resetDate and r['ratingType'] == newRT]
            
            if len(racesBeforeReset) > 0:
                lastRaceBeforeReset = racesBeforeReset[-1]
            else: 
                continue
            
            # reset rating to rating after that race
            setattr(self, newRT, PlackettLuceRating(lastRaceBeforeReset['newMu'], lastRaceBeforeReset['newSigma']))
            
            # cut out future races that will be recalculated
            self.races = [r for r in self.races if r['ratingType'] != newRT or r['date'] < resetDate]
        
    def calculateRaceRivals(self, season, score, otherKeys, otherNames, otherTeams, scoreVals, pos, rivals=None):
        if rivals is None:
            rivals = self.rivals
          
        for otherKey, otherName, otherTeam, otherScore in zip(otherKeys, otherNames, otherTeams, scoreVals):
            if otherKey == self.key:
                continue
            
            wonThisRace = (1 if otherScore > score else 0)
            
            rival = rivals.setdefault(pos, # try and grab counts for this position, with fallback
                {}).setdefault(
                otherKey, # try and grab the info about the other sailor, with fallback
                {
                    'name': otherName,
                    'races': {},
                    'team': otherTeam,
                    'wins': {}
                }
            )

            rival['races'][season] = rival['races'].get(season, 0) + 1
            rival['wins'][season] = rival['wins'].get(season, 0) + wonThisRace
    
    def calculateAllRivals(self, dfr):
        # position, sailor, stat, season
        rivals = {}
        grouped = dfr.groupby(['adjusted_raceID', 'Position'])
        
        for race in self.races:
            season = race['raceID'].split("/")[0]
            if race['type'] == 'fleet':
                score = race['score']
                
                key = (race['raceID'], race['pos'])

                if key in grouped.groups:
                    raceRows = grouped.get_group(key)

                    self.calculateRaceRivals(
                        season,
                        score,
                        raceRows['key'].tolist(),
                        raceRows['Sailor'].tolist(),
                        raceRows['Team'].tolist(),
                        raceRows['Score'].tolist(),
                        race['pos'],
                        rivals
                    )
            
        return rivals
        
    def __repr__(self):
        config = Config()
        return f"{self.name}: {self.teams}, {str(self.sr.ordinal(target=config.targetElo, alpha=config.alpha))} {str(self.tsr.ordinal(target=config.targetElo, alpha=config.alpha))} {self.seasons} {len(self.races)}"


def make_sailor(config, args):
    key, link, name, first_name, last_name, gender, year, teamLink, team, id, external_id = args

    ratings = [PlackettLuceRating(config.model.mu, config.model.sigma) for _ in range(8)]

    return key, Sailor(
        name, key, gender, year,
        [link], [team],
        seasons={'skipper': [], 'crew': []},
        races=[], rivals={},
        sr=ratings[0], cr=ratings[1],
        wsr=ratings[2], wcr=ratings[3],
        tsr=ratings[4], tcr=ratings[5],
        wtsr=ratings[6], wtcr=ratings[7],
    )

def createSailor(sd):
    newSeasons = {'skipper': [], 'crew': []}
    for pos in ['skipper', 'crew']:
        for entry in sd['Seasons'][pos]:
            newSeasons[pos].append((entry[0], entry[1]))
            
    return Sailor(sd['Sailor'], sd['key'], sd['gender'], sd['GradYear'], sd['Links'], sd['Teams'], newSeasons, sd['Races'], sd['Rivals'],     
            PlackettLuceRating(sd['srMU'], sd['srSigma']),
            PlackettLuceRating(sd['crMU'], sd['crSigma']),
            PlackettLuceRating(sd['wsrMU'], sd['wsrSigma']),
            PlackettLuceRating(sd['wcrMU'], sd['wcrSigma']),
            PlackettLuceRating(sd['tsrMU'], sd['tsrSigma']),
            PlackettLuceRating(sd['tcrMU'], sd['tcrSigma']),
            PlackettLuceRating(sd['wtsrMU'], sd['wtsrSigma']),
            PlackettLuceRating(sd['wtcrMU'], sd['wtcrSigma']), sd['Cross'], sd['outLinks'], sd['SkipperRank'], sd['CrewRank'], sd['WomenSkipperRank'], sd['WomenCrewRank'], sd['TRSkipperRank'], sd['TRCrewRank'],sd['TRWomenSkipperRank'], sd['TRWomenCrewRank'], sd['skipperAvgRatio'], sd['crewAvgRatio'])

def setupPeople(df_sailor_ratings, df_sailor_info, config: Config):
    if config.calcAll:
        rows = list(df_sailor_info.itertuples(index=False, name=None))
        results = map(partial(make_sailor, config), rows)
        return dict(results)
    
    else:
        people = {}
        for i, row in df_sailor_ratings.iterrows():
            people[row.key] = createSailor(row.to_dict())

        pplkeys = people.keys()
        rows = [row for row in df_sailor_info.itertuples(index=False, name=None) if row[0] not in pplkeys]
        results = map(partial(make_sailor, config), rows)
        newPeople = dict(results)
        
        people.update(newPeople)
        return people

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

        allRows.append([p.name, len(p.races), sailor,
                        p.skipperRank, p.crewRank, p.womenSkipperRank,
                        p.womenCrewRank, p.skipperRankTR, p.womenSkipperRankTR, p.crewRankTR, p.womenCrewRankTR,
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
                        p.races, p.rivals, p.avgSkipperRatio, p.avgCrewRatio])

    df_sailors = pd.DataFrame(allRows, columns=['Sailor', 'numRaces', 'key', 'SkipperRank', 'CrewRank', 'WomenSkipperRank', 'WomenCrewRank', 'TRSkipperRank', 'TRWomenSkipperRank', 'TRCrewRank', 'TRWomenCrewRank', 'Teams', 'gender',
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

    # df_sailors.to_json(f'sailors-{date.today().strftime("%Y%m%d")}.json', index=False)
    df_sailors.to_json(f'sailors-latest-testing.json', index=False)
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

def updateSailorRatios(people: dict[str, Sailor]):
    for key, p in people.items():
        avgSkipperRatio = float(np.array(
            [r['ratio'] for r in p.races if r['pos'].lower() == 'skipper' and 'ratio' in r.keys()]).mean())
        avgCrewRatio = float(np.array(
            [r['ratio'] for r in p.races if r['pos'].lower() == 'crew' and 'ratio' in r.keys()]).mean())
        p.avgSkipperRatio = avgSkipperRatio
        p.avgCrewRatio = avgCrewRatio

def getCounts(races):
    # season_counts = defaultdict(int)
    season_counts = {}

    for race in races:
        season = race["raceID"].split("/")[0]
        if season not in season_counts.keys():
            season_counts[season] = {}
        if race['pos'] not in season_counts[season].keys():
            season_counts[season][race['pos']] = 0
        season_counts[season][race['pos']] += 1

    return dict(season_counts)

def uploadSailors(people, connection, config : Config, batch_size=300):
    
    # eligible = [p for p in people.values() if (targetSeasons[-1] in p.seasons['skipper']
    #                                            or targetSeasons[-1] in p.seasons['crew'])
    #             and len(p.races) > 0
    #             and type(p.races[-1]['date']) != type("hi")
    #             and (today - p.races[-1]['date']).days < 14]
    eligible = list(people.values())
    print(len(eligible))

    sailor_rows = []
    sailor_teams_rows = []

    for i, p in enumerate(eligible):
        if p.key is None:
            print("No key for", p.name)
            continue

        avg_sk = 0 if p.avgSkipperRatio is None or np.isnan(p.avgSkipperRatio) else p.avgSkipperRatio
        avg_cr = 0 if p.avgCrewRatio is None or np.isnan(p.avgCrewRatio) else p.avgCrewRatio

        sailor_rows.append((
            p.key.replace("/", "-"),
            p.name,
            p.gender,
            int(p.sr.ordinal(target=config.targetElo, alpha=200 / config.model.sigma)),
            int(p.cr.ordinal(target=config.targetElo, alpha=200 / config.model.sigma)),
            int(p.wsr.ordinal(target=config.targetElo, alpha=200 / config.model.sigma)),
            int(p.wcr.ordinal(target=config.targetElo, alpha=200 / config.model.sigma)),
            int(p.tsr.ordinal(target=config.targetElo, alpha=200 / config.model.sigma)),
            int(p.tcr.ordinal(target=config.targetElo, alpha=200 / config.model.sigma)),
            int(p.wtsr.ordinal(target=config.targetElo, alpha=200 / config.model.sigma)),
            int(p.wtcr.ordinal(target=config.targetElo, alpha=200 / config.model.sigma)),
            int(p.skipperRank),
            int(p.crewRank),
            int(p.womenSkipperRank),
            int(p.womenCrewRank),
            int(p.skipperRankTR),
            int(p.crewRankTR),
            int(p.womenSkipperRankTR),
            int(p.womenCrewRankTR),
            avg_sk,
            avg_cr,
            p.year
        ))
        
        raceCounts = (lambda rc_norm, ps: {
                        'skipper': {season: rc_norm.get(season, {}).get('Skipper', 0) for season in [s[0] for s in list(ps['skipper'])]},
                        'crew': {season: rc_norm.get(season, {}).get('Crew', 0) for season in [s[0] for s in list(ps['crew'])]}
                    })( (lambda rc: {s: {pos.title(): cnt for pos, cnt in posd.items()} for s, posd in rc.items()})(getCounts(p.races)), p.seasons ) 
                    
        for position in ['skipper', 'crew']:
            if p.key is None:
                continue
            try:
                for season, team in set(p.seasons[position]):
                    sailor_teams_rows.append((
                        p.key.replace("/", "-"),
                        team,
                        season,
                        position,
                        raceCounts[position][season]
                    ))
            except Exception as e:
                print(position, p.seasons, p.seasons[position])
                raise e

        # Commit in batches
        if (i + 1) % batch_size == 0:
            print(f"Uploading sailors {i - batch_size + 1} to {i}...", len(sailor_teams_rows))
            with connection.cursor() as cursor:
                cursor.executemany("""
                    INSERT INTO Sailors (
                        sailorID, name, gender, sr, cr, wsr, wcr, tsr, tcr, wtsr, wtcr,
                        sRank, cRank, wsRank, wcRank, tsRank, tcRank, wtsRank, wtcRank,
                        avgSkipperRatio, avgCrewRatio, year
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        sr = VALUES(sr),
                        cr = VALUES(cr),
                        wsr = VALUES(wsr),
                        wcr = VALUES(wcr),
                        tsr = VALUES(tsr),
                        tcr = VALUES(tcr),
                        wtsr = VALUES(wtsr),
                        wtcr = VALUES(wtcr),
                        sRank = VALUES(sRank),
                        cRank = VALUES(cRank),
                        wsRank = VALUES(wsRank),
                        wcRank = VALUES(wcRank),
                        tsRank = VALUES(tsRank),
                        tcRank = VALUES(tcRank),
                        wtsRank = VALUES(wtsRank),
                        wtcRank = VALUES(wtcRank),
                        avgSkipperRatio = VALUES(avgSkipperRatio),
                        avgCrewRatio = VALUES(avgCrewRatio)
                """, sailor_rows)
            sailor_rows.clear()
            
            if sailor_teams_rows:
                try:
                    with connection.cursor() as cursor:
                        cursor.executemany("""
                            INSERT INTO SailorTeams(sailorID, teamID, season, position, raceCount)
                            VALUES(%s,%s,%s,%s,%s)
                            ON DUPLICATE KEY UPDATE
                                raceCount = VALUES(raceCount)
                        """, sailor_teams_rows)
                    sailor_teams_rows.clear()
                except Exception as e:
                    print(sailor_teams_rows)
                    raise e

            connection.commit()

    # Final flush
    with connection.cursor() as cursor:
        if sailor_rows:
            cursor.executemany("""
                    INSERT INTO Sailors (
                        sailorID, name, gender, sr, cr, wsr, wcr, tsr, tcr, wtsr, wtcr,
                        sRank, cRank, wsRank, wcRank, tsRank, tcRank, wtsRank, wtcRank,
                        avgSkipperRatio, avgCrewRatio, year
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        sr = VALUES(sr),
                        cr = VALUES(cr),
                        wsr = VALUES(wsr),
                        wcr = VALUES(wcr),
                        tsr = VALUES(tsr),
                        tcr = VALUES(tcr),
                        wtsr = VALUES(wtsr),
                        wtcr = VALUES(wtcr),
                        sRank = VALUES(sRank),
                        cRank = VALUES(cRank),
                        wsRank = VALUES(wsRank),
                        wcRank = VALUES(wcRank),
                        tsRank = VALUES(tsRank),
                        tcRank = VALUES(tcRank),
                        wtsRank = VALUES(wtsRank),
                        wtcRank = VALUES(wtcRank),
                        avgSkipperRatio = VALUES(avgSkipperRatio),
                        avgCrewRatio = VALUES(avgCrewRatio)
                """, sailor_rows)
            connection.commit()

        if sailor_teams_rows:
            cursor.executemany("""
                    INSERT INTO SailorTeams(sailorID, teamID, season, position, raceCount)
                    VALUES(%s,%s,%s,%s,%s)
                    ON DUPLICATE KEY UPDATE
                        raceCount = VALUES(raceCount)
                """, sailor_teams_rows)

            connection.commit()

    print("✅ All sailors uploaded successfully!")