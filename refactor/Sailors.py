from config import Config
from openskill.models import PlackettLuceRating, PlackettLuce
from dataclasses import dataclass, field
from typing import ClassVar
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np
from datetime import date, datetime
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
    
    raceCount: dict[str, dict[str, int]] = 0
    
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

    # Joint WHR fit results, keyed by rating type:
    #   {'sr': {'rating': 1623.4, 'se': 48.2, 'lcb': 1528.9}, ...}
    # Populated from whr_sailors.parquet by loadWHRRatings; empty when useWHR is off.
    whr : dict = field(default_factory=dict)
        
    def getRating(self, position : str, raceType : str, womens: bool, ordinal : bool = False, config : Config = None):
        pos = position.lower()
        typ = raceType.lower()
        if pos not in ('skipper', 'crew') or typ not in ('fleet', 'team'):
            raise ValueError(f"invalid position/type: {position}/{raceType}")
        
        prefix = 'w' if womens else ''
        prefix += 't' if typ == 'team' else ''
        part = 's' if pos == 'skipper' else 'c'
        ratingObj = getattr(self, f"{prefix}{part}r")
        if ordinal and config is not None:
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
        return self.outLinks
        return sum([race['outLinks'] for race in self.races if 'outLinks' in race.keys()])
    
    def getCrossLinks(self):
        return self.cross
        return len([race for race in self.races if 'cross' in race.keys() and race['cross']])
    
    def whrRating(self, ratingType, key="rating"):
        """Fitted WHR rating / se / lcb for a rating type, or None if not fitted."""
        entry = self.whr.get(ratingType)
        return None if entry is None else entry.get(key)

    def isRankEligible(self, targetSeasons, pos, gradCutoff, outLinks=None , needsOutlinks=True, requiredOutLinks=150,
                       ratingType=None, maxRatingSE=None, minRegattas=0):
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
        
        if not (self.hasTargetSeasons(targetSeasons, pos) and betterYear > gradCutoff):
            return False

        # Preferred gate: the posterior SE of "where does this sailor sit nationally",
        # in rating points. This replaces the outLinks count, which measured
        # connectivity backwards - PCCSC had the lowest pass rate and simultaneously
        # the best-determined ratings of any conference.
        if maxRatingSE is not None and ratingType is not None:
            se = self.whrRating(ratingType, "se")
            if se is not None:
                # Distinct regattas, not races: one regatta is one sample of form no
                # matter how many races it contained.
                if minRegattas and (self.whrRating(ratingType, "regattas") or 0) < minRegattas:
                    return False
                return se < maxRatingSE
            # no WHR fit for this sailor/type: fall through to the legacy gate

        if outLinks is None:
            outLinks = self.outLinks

        return (outLinks > requiredOutLinks) if needsOutlinks else True
        
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
        
        if type(resetDate) != float:
            resetDate = resetDate.timestamp()
        
        for pos in ['s', 'c']:
            # Build the rating attribute name, e.g. 'wfr' -> 'wsr'/'wcr', 'tr' -> 'tsr'/'tcr'.
            # Written as an explicit concatenation because the conditional-expression
            # form parsed as 'w' if 'w' in rt else ('t' if 't' in rt else pos+'r'),
            # which returned bare 'w' or 't' for 3 of the 4 rating types.
            newRT = ('w' if 'w' in ratingType else '') + ('t' if 't' in ratingType else '') + pos + 'r'
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
            
    # Everything after the eight ratings is passed by KEYWORD on purpose. Passing these
    # positionally silently landed sd['Cross'] in `raceCount` and shifted every
    # subsequent field by one (cross <- outLinks, outLinks <- SkipperRank, ...).
    return Sailor(sd['Sailor'], sd['key'], sd['gender'], sd['GradYear'], sd['Links'], sd['Teams'], newSeasons,
                  [], #sd['Races']
                  {}, #sd['Rivals']
            PlackettLuceRating(sd['srMU'], sd['srSigma']),
            PlackettLuceRating(sd['crMU'], sd['crSigma']),
            PlackettLuceRating(sd['wsrMU'], sd['wsrSigma']),
            PlackettLuceRating(sd['wcrMU'], sd['wcrSigma']),
            PlackettLuceRating(sd['tsrMU'], sd['tsrSigma']),
            PlackettLuceRating(sd['tcrMU'], sd['tcrSigma']),
            PlackettLuceRating(sd['wtsrMU'], sd['wtsrSigma']),
            PlackettLuceRating(sd['wtcrMU'], sd['wtcrSigma']),
            cross=sd['Cross'], outLinks=sd['outLinks'],
            skipperRank=sd['SkipperRank'], crewRank=sd['CrewRank'],
            womenSkipperRank=sd['WomenSkipperRank'], womenCrewRank=sd['WomenCrewRank'],
            skipperRankTR=sd['TRSkipperRank'], crewRankTR=sd['TRCrewRank'],
            womenSkipperRankTR=sd['TRWomenSkipperRank'], womenCrewRankTR=sd['TRWomenCrewRank'],
            avgSkipperRatio=sd['skipperAvgRatio'], avgCrewRatio=sd['crewAvgRatio'])

def setupPeople(df_sailor_ratings, df_sailor_info, config: Config):
    if config.calcAll:
        rows = list(df_sailor_info.itertuples(index=False, name=None))
        results = map(partial(make_sailor, config), rows)
        return dict(results)
    
    else:
        people = {}
        
        for _, row in df_sailor_ratings.iterrows():
            people[row.key] = createSailor(row.to_dict())

        pplkeys = people.keys()
        rows = [row for row in df_sailor_info.itertuples(index=False, name=None) if row[0] not in pplkeys]
        results = map(partial(make_sailor, config), rows)
        people.update(dict(results))
        
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

def outputSailorsToFile(people, rootDir, config: Config, raceCounts: dict = None):
    """Serialize sailors to sailors-latest.json (the state the incremental path reads back).

    numRaces/outLinks/Cross come from the Sailor attributes and the supplied race
    counts, not from p.races: the calculation pass never populates p.races, so
    deriving them from it wrote 0 for every sailor on every run.
    """
    raceCounts = raceCounts or {}
    allRows = []
    for sailor, p in people.items():

        allRows.append([p.name, raceCounts.get(sailor, 0), sailor,
                        p.skipperRank, p.crewRank, p.womenSkipperRank,
                        p.womenCrewRank, p.skipperRankTR, p.womenSkipperRankTR, p.crewRankTR, p.womenCrewRankTR,
                        p.teams,
                        p.gender,
                        # The *Ord columns carry whatever is being published, so they
                        # follow the joint fit when it is in use rather than always
                        # reporting openskill ordinals.
                        publishedRating(p, 'sr', config),
                        publishedRating(p, 'cr', config),
                        publishedRating(p, 'wsr', config),
                        publishedRating(p, 'wcr', config),
                        publishedRating(p, 'tsr', config),
                        publishedRating(p, 'tcr', config),
                        publishedRating(p, 'wtsr', config),
                        publishedRating(p, 'wtcr', config),
                        p.sr.mu, p.sr.sigma,
                        p.cr.mu, p.cr.sigma,
                        p.wsr.mu, p.wsr.sigma,
                        p.wcr.mu, p.wcr.sigma,
                        p.tsr.mu, p.tsr.sigma,
                        p.tcr.mu, p.tcr.sigma,
                        p.wtsr.mu, p.wtsr.sigma,
                        p.wtcr.mu, p.wtcr.sigma,
                        p.outLinks,
                        p.year, p.links,
                        p.seasons,
                        p.cross,
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
    df_sailors.to_json(rootDir + f'sailors-latest.json', index=False)
    df_sailors = df_sailors.sort_values(
        by='numRaces', ascending=False).reset_index(drop=True)
    
    
# (rating attribute, Sailor rank field, position, uses team-race seasons)
RANK_SPECS = [
    ("sr",   "skipperRank",       "skipper", False),
    ("cr",   "crewRank",          "crew",    False),
    ("wsr",  "womenSkipperRank",  "skipper", False),
    ("wcr",  "womenCrewRank",     "crew",    False),
    ("tsr",  "skipperRankTR",     "skipper", True),
    ("tcr",  "crewRankTR",        "crew",    True),
    ("wtsr", "womenSkipperRankTR","skipper", True),
    ("wtcr", "womenCrewRankTR",   "crew",    True),
]


def calculateSailorRanks(people : dict[str,Sailor], config : Config):
    """Assign per-rating-type national ranks.

    Eligibility and ordering are both resolved per rating type rather than per
    position, because a sailor's open rating can be well determined while their
    women's-fleet rating is not. With useWHR on, eligibility is the posterior SE gate
    and ordering is by the interval lower bound.
    """
    for p in people.values():
        p.resetRanks()

    for ratingType, rankField, pos, isTR in RANK_SPECS:
        seasons = config.targetTRSeasons if isTR else config.targetSeasons
        # Team racing has never had a connectivity gate; keep that until the joint fit
        # covers team races.
        useSE = config.useWHR and not isTR

        eligible = [p for p in people.values()
                    if p.isRankEligible(
                        seasons, pos, config.gradCutoff,
                        needsOutlinks=not isTR,
                        requiredOutLinks=config.requiredOutLinks,
                        ratingType=ratingType if useSE else None,
                        maxRatingSE=config.maxRatingSE if useSE else None,
                        minRegattas=config.minRegattasRanked if useSE else 0)
                    and hasRating(p, ratingType, config)]

        ordered = sorted(eligible, key=lambda p: rankingKey(p, ratingType, config),
                         reverse=True)
        for i, p in enumerate(ordered):
            setattr(p, rankField, i + 1)
        print(f"  {ratingType}: {len(ordered):,} ranked")

    return people


def updateSailorRatios(people: dict[str, Sailor], df_frAfter: pd.DataFrame):
    """Set each sailor's average finish ratio per position.

    Derived from the accumulated post-calc race frame rather than Sailor.races:
    Sailor.races is never populated by the calculation pass (updateRaces appends to a
    single shared list), so the previous implementation silently set every ratio to
    0.0. Reading the frame also avoids duplicating ~1.5M race dicts into the Sailor
    objects, which is what makes sailors-latest.json enormous.
    """
    if df_frAfter.empty:
        return

    ratios = df_frAfter.loc[df_frAfter['ratio'].notna()]
    means = ratios.groupby(['sailorID', ratios['position'].str.lower()])['ratio'].mean()

    for key, p in people.items():
        p.avgSkipperRatio = float(means.get((key, 'skipper'), 0.0))
        p.avgCrewRatio = float(means.get((key, 'crew'), 0.0))

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
    
    sailorSQL = """
                    INSERT INTO Sailors (
                        sailorID, name, gender, sr, cr, wsr, wcr, tsr, tcr, wtsr, wtcr,
                        srSE, crSE, wsrSE, wcrSE,
                        sRank, cRank, wsRank, wcRank, tsRank, tcRank, wtsRank, wtcRank,
                        avgSkipperRatio, avgCrewRatio, crossLinks, outLinks, year
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        sr = VALUES(sr),
                        cr = VALUES(cr),
                        wsr = VALUES(wsr),
                        wcr = VALUES(wcr),
                        tsr = VALUES(tsr),
                        tcr = VALUES(tcr),
                        wtsr = VALUES(wtsr),
                        wtcr = VALUES(wtcr),
                        srSE = VALUES(srSE),
                        crSE = VALUES(crSE),
                        wsrSE = VALUES(wsrSE),
                        wcrSE = VALUES(wcrSE),
                        sRank = VALUES(sRank),
                        cRank = VALUES(cRank),
                        wsRank = VALUES(wsRank),
                        wcRank = VALUES(wcRank),
                        tsRank = VALUES(tsRank),
                        tcRank = VALUES(tcRank),
                        wtsRank = VALUES(wtsRank),
                        wtcRank = VALUES(wtcRank),
                        avgSkipperRatio = VALUES(avgSkipperRatio),
                        avgCrewRatio = VALUES(avgCrewRatio),
                        crossLinks = VALUES(crossLinks),
                        outLinks = VALUES(outLinks)
                """
    sailorTeamsSQL = """
                            INSERT INTO SailorTeams(sailorID, teamID, season, position, raceCount)
                            VALUES(%s,%s,%s,%s,%s)
                            ON DUPLICATE KEY UPDATE
                                raceCount = VALUES(raceCount)
                        """

    for i, p in enumerate(eligible):
        if p.key is None or p.name is None or p.name == "":
            print("No key for", p.name)
            continue

        avg_sk = 0 if p.avgSkipperRatio is None or np.isnan(p.avgSkipperRatio) else p.avgSkipperRatio
        avg_cr = 0 if p.avgCrewRatio is None or np.isnan(p.avgCrewRatio) else p.avgCrewRatio

        sailor_rows.append((
            p.key.replace("/", "-"),
            p.name,
            p.gender,
            # publishedRating returns the joint-fit value when useWHR is on and falls
            # back to the openskill ordinal otherwise, so these columns keep their
            # meaning and scale either way.
            int(publishedRating(p, 'sr', config)),
            int(publishedRating(p, 'cr', config)),
            int(publishedRating(p, 'wsr', config)),
            int(publishedRating(p, 'wcr', config)),
            int(publishedRating(p, 'tsr', config)),
            int(publishedRating(p, 'tcr', config)),
            int(publishedRating(p, 'wtsr', config)),
            int(publishedRating(p, 'wtcr', config)),
            publishedSE(p, 'sr', config),
            publishedSE(p, 'cr', config),
            publishedSE(p, 'wsr', config),
            publishedSE(p, 'wcr', config),
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
            int(p.cross),
            int(p.outLinks),
            p.year
        ))
        
        # raceCounts = (lambda rc_norm, ps: {
        #                 'skipper': {season: rc_norm.get(season, {}).get('Skipper', 0) for season in [s[0] for s in list(ps['skipper'])]},
        #                 'crew': {season: rc_norm.get(season, {}).get('Crew', 0) for season in [s[0] for s in list(ps['crew'])]}
        #             })( (lambda rc: {s: {pos.title(): cnt for pos, cnt in posd.items()} for s, posd in rc.items()})(getCounts(p.races)), p.seasons ) 
                    
        # for position in ['skipper', 'crew']:
        #     if p.key is None:
        #         continue
        #     try:
        #         for season, team in set(p.seasons[position]):
        #             sailor_teams_rows.append((
        #                 p.key.replace("/", "-"),
        #                 team,
        #                 season,
        #                 position,
        #                 raceCounts[position][season]
        #             ))
        #     except Exception as e:
        #         print(position, p.seasons, p.seasons[position])
        #         raise e

        # Commit in batches
        if (i + 1) % batch_size == 0:
            print(f"Uploading sailors {i - batch_size + 1} to {i}...", len(sailor_teams_rows))
            with connection.cursor() as cursor:
                try :
                    cursor.executemany(sailorSQL, sailor_rows)
                except Exception as e:
                    print(f"sailorSQL", sailor_rows)
                    raise e
            sailor_rows.clear()
            
            if sailor_teams_rows:
                try:
                    with connection.cursor() as cursor:
                        cursor.executemany(sailorTeamsSQL, sailor_teams_rows)
                    sailor_teams_rows.clear()
                except Exception as e:
                    print(sailor_teams_rows)
                    raise e

            connection.commit()

    # Final flush
    with connection.cursor() as cursor:
        if sailor_rows:
            cursor.executemany(sailorSQL, sailor_rows)
            connection.commit()

        if sailor_teams_rows:
            cursor.executemany(sailorTeamsSQL, sailor_teams_rows)
            connection.commit()

    print("✅ All sailors uploaded successfully!")

def _asInt(v, default=0):
    """int() that tolerates a missing/NaN value (e.g. a column only some frames have)."""
    try:
        return default if v is None or v != v else int(v)
    except (TypeError, ValueError):
        return default


def loadWHRRatings(people: dict[str, Sailor], rootDir, config: Config,
                   whrFile="whr_sailors.parquet"):
    """Attach joint-WHR ratings, SEs and lower confidence bounds onto the Sailors.

    Reads the output of whr.runFleetPipeline. Silently no-ops if the file is absent so
    the openskill pipeline still runs on a clean checkout.
    """
    import os
    path = rootDir + whrFile
    if not os.path.exists(path):
        print(f"No {whrFile}; skipping WHR ratings. Run refactor/whr.py first.")
        return people

    df = pd.read_parquet(path)
    attached = 0
    for row in df.itertuples(index=False):
        p = people.get(row.sailorID)
        if p is None:
            continue
        p.whr[row.ratingType] = {"rating": float(row.rating),
                                 "se": float(row.se),
                                 "lcb": float(row.lcb),
                                 "shrunk": float(getattr(row, "shrunk", row.rating)),
                                 "popmean": float(getattr(row, "popmean", row.rating)),
                                 "regattas": _asInt(getattr(row, "regattas", 0))}
        attached += 1
    print(f"Attached {attached:,} WHR ratings across "
          f"{df['ratingType'].nunique()} rating types.")
    return people


def rankingKey(sailor: Sailor, ratingType: str, config: Config):
    """Value to sort rankings by.

    Sorts on the empirical-Bayes shrunk rating for every rating type. Ranking on
    rating - 1.96*SE ordered the board partly by how precisely a sailor was known,
    which favoured whoever raced most; ranking on the raw point estimate let a sailor
    who won four races in a seven-boat fleet reach the top of the country. Shrinking
    by reliability regresses thin records toward average by the right amount instead.
    """
    if config.useWHR:
        v = sailor.whrRating(ratingType, getattr(config, "rankingStatistic", "lcb"))
        if v is not None:
            return v
    return getattr(sailor, ratingType).ordinal(target=config.targetElo,
                                               alpha=config.alpha)


def publishedRating(sailor: Sailor, ratingType: str, config: Config):
    """The rating value written to the DB and shown on the site.

    With useWHR on this is the joint-fit rating, which is anchored per rating type onto
    the ranked population so it lands on the same scale the site already displays and
    the existing INT columns can be reused unchanged. Team-race types have no joint fit
    yet, so they always fall through to openskill.
    """
    if config.useWHR:
        r = sailor.whrRating(ratingType, getattr(config, "publishedStatistic", "rating"))
        if r is not None:
            return r
    return getattr(sailor, ratingType).ordinal(target=config.targetElo,
                                               alpha=config.alpha)


def hasRating(sailor: Sailor, ratingType: str, config: Config):
    """Whether this sailor has a usable rating of this type.

    Asks the joint fit when it is in use. The previous test was
    `getattr(p, ratingType).mu != config.model.mu` - "has openskill moved this sailor
    off the prior mean" - which silently required the openskill pass to have run even
    when its ratings were no longer being published.
    """
    if config.useWHR:
        if sailor.whrRating(ratingType) is not None:
            return True
        if sailor.whr:
            return False        # fitted, just not for this rating type
    return getattr(sailor, ratingType).mu != config.model.mu


def publishedSE(sailor: Sailor, ratingType: str, config: Config):
    """Posterior SE of the rating in display points, or None when not fitted."""
    return sailor.whrRating(ratingType, "se") if config.useWHR else None


SAILOR_WHR_DDL = """
-- Minimal migration for the joint WHR fit.
--
-- The rating columns (sr, cr, wsr, wcr, tsr, tcr, wtsr, wtcr) are REUSED as-is: the
-- joint fit is anchored per rating type onto the ranked population, so it lands on
-- the same scale the front end already renders and stays an INT. Ranks, ratios,
-- crossLinks and outLinks are unchanged. outLinks is retained as a descriptive stat
-- even though it no longer gates eligibility.
--
-- Only the four fleet standard errors are new. They are nullable, so the front end
-- can ignore them until it is ready to render intervals.

ALTER TABLE Sailors
  ADD COLUMN srSE  FLOAT NULL AFTER wtcr,
  ADD COLUMN crSE  FLOAT NULL AFTER srSE,
  ADD COLUMN wsrSE FLOAT NULL AFTER crSE,
  ADD COLUMN wcrSE FLOAT NULL AFTER wsrSE;

-- Per-race credit: the joint-fit replacement for the sequential per-race delta.
-- oldRating/newRating are REUSED as-is - the ladder in whr.raceLadder is built so that
-- newRating - oldRating == credit, so any existing "gained this race" display keeps
-- working. credit is stored separately at full precision because oldRating and
-- newRating are INT and rounding would flatten small values.
ALTER TABLE FleetScores
  ADD COLUMN credit FLOAT NULL AFTER newRating;
"""


def applyWHRToRaces(df_frAfter: pd.DataFrame, rootDir, config: Config,
                    whrFile="whr_races.parquet"):
    """Overlay the joint fit's per-race ladder onto the fleet score rows.

    Replaces oldRating/newRating/predicted with the joint-fit values and adds
    `credit`. The ladder is built so newRating - oldRating == credit, so the existing
    "rating gained this race" display keeps working with no front-end change.

    Rows with no joint fit (team races, or fleet rows the fit skipped) keep their
    openskill values and get a NULL credit.
    """
    import os
    path = rootDir + whrFile
    if not os.path.exists(path):
        print(f"No {whrFile}; leaving openskill per-race ratings in place.")
        df_frAfter["credit"] = np.nan
        return df_frAfter

    cols = ["raceID", "sailorID", "position", "ratingType", "oldRating", "newRating",
            "predicted", "credit", "regAvg"]
    # theta is only present once the pipeline has been re-run since it was added.
    try:
        w = pd.read_parquet(path, columns=cols + ["theta"])
    except Exception:
        w = pd.read_parquet(path, columns=cols)
    key = ["raceID", "sailorID", "position", "ratingType"]
    merged = df_frAfter.merge(w, on=key, how="left", suffixes=("", "_whr"))

    hit = merged["credit"].notna()
    for col in ("oldRating", "newRating", "predicted", "regAvg"):
        merged[col] = merged[col + "_whr"].where(hit, merged[col])
    # Put the joint fit's raw theta where the harness expects a skill value, so the
    # pairwise metrics keep working once the openskill pass is off (use link="logit",
    # beta=1). oldSigma goes to 0: the joint fit's uncertainty is per sailor, not per
    # race, and lives on whr_sailors.parquet.
    if "theta" in merged.columns:
        merged["oldMu"] = merged["theta"].where(hit, merged.get("oldMu"))
        merged["oldSigma"] = np.where(hit, 0.0, merged.get("oldSigma", 0.0))
        merged = merged.drop(columns=["theta"])
    merged = merged.drop(columns=[c for c in merged.columns if c.endswith("_whr")])
    print(f"Applied WHR per-race ratings to {int(hit.sum()):,} of {len(merged):,} fleet rows.")
    return merged
