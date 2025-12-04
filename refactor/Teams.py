from config import Config
from regions import teamRegions
import datetime
import pandas as pd
import numpy as np
import mysql
from Sailors import Sailor


def validPerson(p : Sailor, pos, config: Config):
    # if(p.hasTargetSeasons(config.targetSeasons, pos)):
        # print(p.name, pos, p.hasTargetSeasons(config.targetSeasons, pos))
    return (p.cross > 20
            and p.outLinks > 70
            and not p.hasTargetSeasons(config.targetSeasons, pos) # don't have to check this bc only using currentSailors
            # and (2000 + int(p.year.split()[0]) > gradCutoff if isinstance(p.year, str) and len(p.year.split()) > 1 else int(p.year) > gradCutoff)
            # and sum([p['raceCount'][seas] for seas in config.targetSeasons if seas in p['raceCount'].keys()]) > 5
            )

def getOrderedSailors(people : list[Sailor], ratingType, pos, config : Config):
    numTops = config.numTops['open' if 'w' not in ratingType else 'womens']
    isTR = 't' in ratingType
    orderedSailors = sorted([p for p in people
                            if p.isRankEligible(config.targetSeasons, pos, config.gradCutoff, not isTR)
                             and getattr(p, ratingType).mu != config.model.mu
                             ],
                            key=lambda x: getattr(x, ratingType).ordinal(
                                target=config.targetElo, alpha=config.alpha),
                            reverse=True)
    # print(ratingType, [t.name for t in orderedSailors[:numTops]])
    
    sailorSum = sum([getattr(p, ratingType).ordinal(target=config.targetElo, alpha=config.alpha)
                            for p in orderedSailors[:numTops]])
    topSailors = [{'name': p.name, 'key': p.key,
                    ratingType: getattr(p, ratingType).ordinal(target=config.targetElo, alpha=config.alpha)} for p in orderedSailors[:numTops]]
    return topSailors, sailorSum

def calculateTopSailors(filtered_people, isTeamRace, isWomens, config: Config):
    prefix = 't' if isTeamRace else ''
    if isWomens:
        prefix = 'w' + prefix
    topSkippers, topSkippersSum = getOrderedSailors(filtered_people, prefix + 'sr', 'skipper', config)
    topCrews, topCrewsSum = getOrderedSailors(filtered_people, prefix + 'cr', 'crew', config)

    numTops = config.numTops['open']
    topRating = (topSkippersSum + topCrewsSum) / (numTops * 2)
    return topRating, topSkippers, topCrews

def getRankType(sailor, season, topSailors, rankTypes, config: Config):
    rankType = ''
    if season in config.targetSeasons:
        for sailorList, rt in zip(topSailors, rankTypes):
            for rankingSailor in sailorList:
                if sailor.key == rankingSailor['key']:
                    if rankType == '':
                        rankType = rt
                    else:
                        rankType = rankType + '.' + rt
    return rankType
    
def uploadSailorTeams(filtered_people : list[Sailor], team, topSkippers: list[list[dict]], topCrews: list[list[dict]], connection, config: Config):
    rankTypesSkipper = ['sr', 'wsr', 'tsr', 'wtsr']
    rankTypesCrew = ['cr', 'wcr', 'tcr', 'wtcr']
    
    batch_size = 200
    rows_to_insert = []
    
    for sailor in filtered_people:
        for position, topSailors, rankTypes in zip(['skipper', 'crew'], [topSkippers, topCrews], [rankTypesSkipper, rankTypesCrew]):
            for season, seasonTeam in sailor.seasons[position]:
                if seasonTeam == team: # Only insert if sailor was actually on this team in this season
                    rankType = getRankType(sailor, season, topSailors, rankTypes, config)
                    raceCount = sailor.getSeasonRaceCount(season, position)
                                
                    rows_to_insert.append((sailor.key,
                            team,
                            season,
                            position,
                            raceCount,
                            rankType))

    # Insert in batches
    for start in range(0, len(rows_to_insert), batch_size):
        batch = rows_to_insert[start:start + batch_size]
        try:
            with connection.cursor() as cursor:
                cursor.executemany("""
                    INSERT INTO SailorTeams
                        (sailorID, teamID, season, position, raceCount, rankType)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        raceCount = VALUES(raceCount),
                        rankType = VALUES(rankType)
                """, batch)
            connection.commit()

        except mysql.connector.errors.IntegrityError as e:
            print("Batch insert failed:", e)
            raise e
    
def calculateAvgRatio(filtered_people: list[Sailor]):
    skipperRatios = [p.avgSkipperRatio for p in filtered_people if p.avgSkipperRatio != 0 and not np.isnan(p.avgSkipperRatio)]
    avgSkipperRatio = 0
    if(len(skipperRatios) > 0):
        avgSkipperRatio = sum(skipperRatios) / len(skipperRatios)
    
    crewRatios = [p.avgCrewRatio for p in filtered_people if p.avgCrewRatio != 0 and not np.isnan(p.avgCrewRatio)]
    avgCrewRatio = 0
    if(len(crewRatios) > 0):
        avgCrewRatio = sum(crewRatios) / len(crewRatios)
    
    avgRatio = (avgSkipperRatio + avgCrewRatio) / 2
    return avgRatio

def calculateAvgRating(people : list[Sailor], config:Config):
    ratings = []
    for p in people:
        sr = p.sr.ordinal(target=config.targetElo, alpha=config.alpha)
        cr = p.cr.ordinal(target=config.targetElo, alpha=config.alpha)
        wsr = p.wsr.ordinal(target=config.targetElo, alpha=config.alpha)
        wcr = p.wcr.ordinal(target=config.targetElo, alpha=config.alpha)
        ratings.append(max([sr if sr != config.targetElo else 0, 
                         cr if cr != config.targetElo else 0, 
                         wsr if wsr != config.targetElo else 0, 
                         wcr if wcr != config.targetElo else 0]))
        
        tsr = p.sr.ordinal(target=config.targetElo, alpha=config.alpha)
        tcr = p.cr.ordinal(target=config.targetElo, alpha=config.alpha)
        wtsr = p.wsr.ordinal(target=config.targetElo, alpha=config.alpha)
        wtcr = p.wcr.ordinal(target=config.targetElo, alpha=config.alpha)
        ratings.append(max([tsr if tsr != config.targetElo else 0, 
                         tcr if tcr != config.targetElo else 0, 
                         wtsr if wtsr != config.targetElo else 0, 
                         wtcr if wtcr != config.targetElo else 0]))

    return sum(ratings) / len(ratings) if len(ratings) > 0 else 0
    
def uploadTeams(people: dict[str, Sailor], connection, config: Config):
    for team, region in teamRegions.items():
        sailors : list[Sailor] = [p for key, p in people.items() if team in p.teams]
        currentSailors : list[Sailor] = [p for p in sailors if p.isOnTeamInSeasons(team, config.targetSeasons)]
        
        topRating, topSkippers, topCrews = calculateTopSailors(currentSailors, False, False, config)
        topWomenRating, topWomenSkippers, topWomenCrews = calculateTopSailors(currentSailors, False, True, config)
        topRatingTR, topSkippersTR, topCrewsTR = calculateTopSailors(currentSailors, True, False, config)
        topWomenRatingTR, topWomenSkippersTR, topWomenCrewsTR = calculateTopSailors(currentSailors, True, True, config)
        
        avg = calculateAvgRating(currentSailors, config)
        avgRatio = calculateAvgRatio(currentSailors)
        
        # if team == 'UC Santa Cruz':
        #     for s in currentSailors:
        #         print(s.key, s.getOutLinks())
        
        with connection.cursor() as cursor:
            cursor.execute("""
                INSERT INTO Teams
                    (teamID, teamName, topFleetRating, topWomenRating, topTeamRating,
                    topWomenTeamRating, avgRating, avgRatio, region)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    topFleetRating = VALUES(topFleetRating),
                    topWomenRating = VALUES(topWomenRating),
                    topTeamRating = VALUES(topTeamRating),
                    topWomenTeamRating = VALUES(topWomenTeamRating),
                    avgRating = VALUES(avgRating),
                    avgRatio = VALUES(avgRatio)
            """, (team, team, topRating, topWomenRating, topRatingTR,
                topWomenRatingTR, avg, avgRatio, region))
        connection.commit()

        
        uploadSailorTeams(sailors, team, [topSkippers, topWomenSkippers, topSkippersTR, topWomenSkippersTR], [topCrews,topWomenCrews, topCrewsTR, topWomenCrewsTR], connection, config)
        
        # print("Updated ", team)