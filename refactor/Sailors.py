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

    # {ratingType: [(date, mu, sigma), ...]} in chronological order. Compact on purpose:
    # storing the full race dicts here meant ~1.5M dicts held in 26k objects.
    ratingHistory : dict[str, list[tuple[float, float, float]]] = field(default_factory=dict)
        
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
    
    def isOnTeamInSeasons(self, team, seasons):
        return team in [t for [s, t] in self.seasons['skipper'] if s in seasons] or team in [t for [s, t] in self.seasons['crew'] if s in seasons]
      
    def hasTargetSeasons(self, targetSeasons, pos):
        seasonsSet = set([s[0] for s in self.seasons[pos]])
        return not seasonsSet.isdisjoint(targetSeasons)
    
    def getOutLinks(self):
        return self.outLinks

    def getCrossLinks(self):
        return self.cross
    
    def isRankEligible(self, targetSeasons, pos, gradCutoff, requiredOutLinks, outLinks=None, needsOutlinks=True):
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
        
        if outLinks == None:
            outLinks = self.outLinks
        
        return (self.hasTargetSeasons(targetSeasons, pos) # has target seasons
                        and (outLinks > requiredOutLinks if needsOutlinks else True) # and has x outlinks
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
        
    def totalRaces(self):
        return sum(len(v) for v in self.ratingHistory.values())

    def recordRating(self, ratingType, date, mu, sigma):
        """Append this race's post-update rating state to the sailor's history.

        Only (date, mu, sigma) is kept - enough to restore the rating on a resume,
        without holding a copy of every race row.
        """
        self.ratingHistory.setdefault(ratingType, []).append((date, mu, sigma))

    @staticmethod
    def ratingAttr(ratingType, pos):
        """Attribute name for a regatta rating type ('fr'/'wfr'/'tr'/'wtr') and position.

        NOTE the parentheses. Written without them this read as
        `'w' if 'w' in rt else ('t' if 't' in rt else pos + 'r')`, which returned a bare
        'w' or 't' for 3 of the 4 rating types and silently broke the reset.
        """
        return (('w' if 'w' in ratingType else '')
                + ('t' if 't' in ratingType else '')
                + pos + 'r')

    def resetRatingToBeforeDate(self, resetDate, ratingType, config : Config):
        if ratingType not in self.ratingTypesReset:
            self.ratingTypesReset.append(ratingType)

        if type(resetDate) != float:
            resetDate = resetDate.timestamp()

        for pos in ['s', 'c']:
            newRT = self.ratingAttr(ratingType, pos)
            history = self.ratingHistory.get(newRT, [])
            before = [h for h in history if h[0] < resetDate]

            if before:
                # roll back to the rating as of the last race before the reset point
                _, mu, sigma = before[-1]
            else:
                # every race of this type is after the reset point, so the rating has to
                # go back to the starting rating. Returning early here (the old
                # behaviour) left the fully-updated rating in place.
                mu, sigma = config.model.mu, config.model.sigma

            setattr(self, newRT, PlackettLuceRating(mu, sigma))

            # drop the history that is about to be recalculated
            self.ratingHistory[newRT] = before
        
    def __repr__(self):
        config = Config()
        return f"{self.name}: {self.teams}, {str(self.sr.ordinal(target=config.targetElo, alpha=config.alpha))} {str(self.tsr.ordinal(target=config.targetElo, alpha=config.alpha))} {self.seasons} {self.totalRaces()}"


def make_sailor(config, args):
    key, link, name, first_name, last_name, gender, year, teamLink, team, id, external_id = args

    ratings = [PlackettLuceRating(config.model.mu, config.model.sigma) for _ in range(8)]

    return key, Sailor(
        name, key, gender, year,
        [link], [team],
        seasons={'skipper': [], 'crew': []},
        rivals={},
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
            
    # Keyword arguments throughout. Passing these positionally put `Cross` into
    # `raceCount` and shifted all 12 following fields by one slot, so `SkipperRank`
    # landed in `outLinks` and the eligibility gate read a rank as a comparison count.
    return Sailor(
        name=sd['Sailor'], key=sd['key'], gender=sd['gender'], year=sd['GradYear'],
        links=sd['Links'], teams=sd['Teams'], seasons=newSeasons, rivals={},
        sr=PlackettLuceRating(sd['srMU'], sd['srSigma']),
        cr=PlackettLuceRating(sd['crMU'], sd['crSigma']),
        wsr=PlackettLuceRating(sd['wsrMU'], sd['wsrSigma']),
        wcr=PlackettLuceRating(sd['wcrMU'], sd['wcrSigma']),
        tsr=PlackettLuceRating(sd['tsrMU'], sd['tsrSigma']),
        tcr=PlackettLuceRating(sd['tcrMU'], sd['tcrSigma']),
        wtsr=PlackettLuceRating(sd['wtsrMU'], sd['wtsrSigma']),
        wtcr=PlackettLuceRating(sd['wtcrMU'], sd['wtcrSigma']),
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

def loadRatingHistory(people : dict[str, Sailor], df_oldRaces, config : Config):
    """Rebuild each sailor's (date, mu, sigma) history from saved post-calc race rows.

    Required for a resume: resetRatingToBeforeDate rolls a rating back to the last race
    before the reset point, and without this there is no history to roll back to.
    """
    needed = ['sailorID', 'ratingType', 'date', 'newMu', 'newSigma']
    missing = [c for c in needed if c not in df_oldRaces.columns]
    if missing:
        print(f"Post-calc rows lack {missing}; cannot rebuild rating history. "
              f"Re-run once with calcAll=True to write them.")
        return people

    # newMu/newSigma included in the dropna: concatenating a migrated fleet parquet with
    # an unmigrated team one would otherwise leave NaN ratings in the history.
    df = df_oldRaces[needed].dropna(subset=needed)
    df = df.sort_values('date', kind='stable')

    matched = 0
    for (key, ratingType), grp in df.groupby(['sailorID', 'ratingType'], sort=False):
        p = people.get(key)
        if p is None:
            continue
        p.ratingHistory[ratingType] = list(
            zip(grp['date'].tolist(), grp['newMu'].tolist(), grp['newSigma'].tolist()))
        matched += len(grp)

    print(f"Rebuilt rating history for {matched} race rows")
    return people

def handleMerges(df_races, people, config : Config):
    # merge sailor objects
    for oldkey, newkey in config.merges.items():
        if oldkey not in people:
            continue

        old = people.pop(oldkey)
        new = people.get(newkey)

        if new is None:
            # The merge target does not exist as its own sailor (e.g. the canonical key
            # was never scraped). Re-key the existing record rather than raising.
            old.key = newkey
            people[newkey] = old
        else:
            new.links = new.links + old.links
            if old.teams != new.teams:
                new.teams = new.teams + old.teams

        df_races['Link'] = df_races['Link'].replace(oldkey, newkey)
        df_races['key'] = df_races['key'].replace(oldkey, newkey)
    return people, df_races
        
def outputSailorsToFile(people, rootDir, config: Config ):
    allRows = []
    for sailor, p in people.items():

        allRows.append([p.name, p.totalRaces(), sailor,
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
                        p.outLinks,
                        p.year, p.links,
                        p.seasons,
                        p.cross,
                        p.rivals, p.avgSkipperRatio, p.avgCrewRatio])

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
                                                'Seasons', 'Cross', 'Rivals', 'skipperAvgRatio', 'crewAvgRatio'])

    # df_sailors.to_json(f'sailors-{date.today().strftime("%Y%m%d")}.json', index=False)
    df_sailors.to_json(rootDir + f'sailors-latest.json', index=False)
    df_sailors = df_sailors.sort_values(
        by='numRaces', ascending=False).reset_index(drop=True)
    
    
def calculateSailorRanks(people : dict[str,Sailor], config : Config):
    eligible_skippers = [p for p in people.values()
                         if p.isRankEligible(config.targetSeasons, 'skipper', config.gradCutoff,
                                             config.requiredOutLinks)]

    eligible_crews = [p for p in people.values()
                      if p.isRankEligible(config.targetSeasons, 'crew', config.gradCutoff,
                                          config.requiredOutLinks)]

    # TODO: Count tr and fr seasons seperately
    eligible_skippers_tr = [p for p in people.values()
                            if p.isRankEligible(config.targetTRSeasons, 'skipper', config.gradCutoff, needsOutlinks=False)]
    eligible_crews_tr = [p for p in people.values()
                         if p.isRankEligible(config.targetTRSeasons, 'crew', config.gradCutoff, needsOutlinks=False)]
    
    print(len(eligible_skippers_tr))

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

def updateSailorRatios(people: dict[str, Sailor], df_races):
    """Set each sailor's mean finishing ratio per position from the post-calc race rows.

    Previously read `p.races`, which was never populated, so every ratio was 0.0.
    `df_races` is the combined fleet+team frame with lowercase `position` and a `ratio`.
    """
    means = (df_races.loc[df_races['ratio'].notna()]
             .groupby(['sailorID', 'position'])['ratio'].mean())

    for key, p in people.items():
        p.avgSkipperRatio = float(means.get((key, 'skipper'), 0.0))
        p.avgCrewRatio = float(means.get((key, 'crew'), 0.0))

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
                        sRank, cRank, wsRank, wcRank, tsRank, tcRank, wtsRank, wtcRank,
                        avgSkipperRatio, avgCrewRatio, crossLinks, outLinks, year
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
            int(p.sr.ordinal(target=config.targetElo, alpha=config.alpha)),
            int(p.cr.ordinal(target=config.targetElo, alpha=config.alpha)),
            int(p.wsr.ordinal(target=config.targetElo, alpha=config.alpha)),
            int(p.wcr.ordinal(target=config.targetElo, alpha=config.alpha)),
            int(p.tsr.ordinal(target=config.targetElo, alpha=config.alpha)),
            int(p.tcr.ordinal(target=config.targetElo, alpha=config.alpha)),
            int(p.wtsr.ordinal(target=config.targetElo, alpha=config.alpha)),
            int(p.wtcr.ordinal(target=config.targetElo, alpha=config.alpha)),
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