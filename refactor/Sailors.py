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
    teams : list[str]

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
        if self.year is None:
            print(f"{self.key} has none year")
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
        
    def resetRatingToBeforeRace(self, raceID):
        # find the race before this race
        
        # set 
        
        return 
        
    def __repr__(self):
        config = Config()
        return f"{self.name}: {self.teams}, {str(self.sr.ordinal(target=config.targetElo, alpha=config.alpha))} {str(self.tsr.ordinal(target=config.targetElo, alpha=config.alpha))} {self.seasons} {len(self.races)}"


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
                        p.races, p.rivals, p.avgSkipperRatio, p.avgCrewRatio])

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

    df_sailors.to_parquet(f'sailors-{date.today().strftime("%Y%m%d")}.parquet', index=False)
    df_sailors.to_parquet(f'sailors-latest.parquet', index=False)
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
            [r['ratio'] for r in p.races if r['pos'] == 'Skipper' and 'ratio' in r.keys()]).mean())
        avgCrewRatio = float(np.array(
            [r['ratio'] for r in p.races if r['pos'] == 'Crew' and 'ratio' in r.keys()]).mean())
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
    rival_rows = []
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

        for position in ['Skipper', 'Crew']:
            if position not in p.rivals:
                continue
            for key, values in p.rivals[position].items():
                for season in values['races']:
                    rival_rows.append((
                        p.key.replace("/", "-"),
                        key,
                        values['name'],
                        values['team'],
                        position,
                        season,
                        values['races'][season],
                        values['wins'][season]
                    ))
                    
        for position in ['skipper', 'crew']:
            if p.key is None:
                continue
            for season, team in set(p.seasons[position]):
                sailor_teams_rows.append((
                    p.key.replace("/", "-"),
                    team,
                    season,
                    position,
                    raceCounts[position][season]
                ))

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

            if rival_rows:
                with connection.cursor() as cursor:
                    cursor.executemany("""
                        INSERT INTO SailorRivals (
                            sailorID, rivalID, rivalName, rivalTeam, position, season, raceCount, winCount
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE
                            raceCount = VALUES(raceCount),
                            winCount = VALUES(winCount)
                    """, rival_rows)
                rival_rows.clear()
            
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
        if rival_rows:
            cursor.executemany("""
                        INSERT INTO SailorRivals (
                            sailorID, rivalID, rivalName, rivalTeam, position, season, raceCount, winCount
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE
                            raceCount = VALUES(raceCount),
                            winCount = VALUES(winCount)
                    """, rival_rows)
        if sailor_teams_rows:
            cursor.executemany("""
                    INSERT INTO SailorTeams(sailorID, teamID, season, position, raceCount)
                    VALUES(%s,%s,%s,%s,%s)
                    ON DUPLICATE KEY UPDATE
                        raceCount = VALUES(raceCount)
                """, sailor_teams_rows)

            connection.commit()

    print("✅ All sailors uploaded successfully!")