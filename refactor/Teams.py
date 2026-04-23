from config import Config
from regions import teamRegions
import datetime
import pandas as pd
import numpy as np
import mysql
from Sailors import Sailor

def getOrderedSailors(people : list[Sailor], ratingType, pos, outlinks_dict, config : Config):
    # print(ratingType)
    numTops = config.numTops['tr' if 't' in ratingType else 'fr']['open' if 'w' not in ratingType else 'womens']
    isTR = 't' in ratingType
    outlinks_keys = outlinks_dict.keys()
    orderedSailors = sorted([p for p in people
                            if p.isRankEligible(config.targetSeasons, pos, config.gradCutoff, outLinks= outlinks_dict[p.key] if p.key in outlinks_keys else None, needsOutlinks= not isTR)
                             and getattr(p, ratingType).mu != config.model.mu
                             ],
                            key=lambda x: getattr(x, ratingType).ordinal(
                                target=config.targetElo, alpha=config.alpha),
                            reverse=True)
    print(ratingType)
    print(orderedSailors)

    sailorSum = sum([getattr(p, ratingType).ordinal(target=config.targetElo, alpha=config.alpha)
                            for p in orderedSailors[:numTops]])
    topSailors = [{'name': p.name, 'key': p.key,
                    ratingType: getattr(p, ratingType).ordinal(target=config.targetElo, alpha=config.alpha)} for p in orderedSailors[:numTops]]
    print(topSailors)
    return topSailors, sailorSum

def calculateTopSailors(filtered_people, outlinks_dict, isTeamRace, isWomens, config: Config):
    prefix = 't' if isTeamRace else ''
    if isWomens:
        prefix = 'w' + prefix
    topSkippers, topSkippersSum = getOrderedSailors(filtered_people, prefix + 'sr', 'skipper', outlinks_dict, config)
    topCrews, topCrewsSum = getOrderedSailors(filtered_people, prefix + 'cr', 'crew', outlinks_dict, config)

    numTops = config.numTops['tr' if isTeamRace else 'fr']['open']
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
                        rankType += '.' + rt
    return rankType
    
def uploadSailorTeams(filtered_people : list[Sailor], team, topSkippers: list[list[dict]], topCrews: list[list[dict]], racecounts_dict, winp_dict, connection, config: Config):
    rankTypesSkipper = ['sr', 'wsr', 'tsr', 'wtsr']
    rankTypesCrew = ['cr', 'wcr', 'tcr', 'wtcr']
    
    batch_size = 200
    rows_to_insert = []
    
    for sailor in filtered_people:
        for position, topSailors, rankTypes in zip(['skipper', 'crew'], [topSkippers, topCrews], [rankTypesSkipper, rankTypesCrew]):
            for season, seasonTeam in sailor.seasons[position]:
                if seasonTeam == team: # Only insert if sailor was actually on this team in this season
                    rankType = getRankType(sailor, season, topSailors, rankTypes, config)
                    raceCount = racecounts_dict.get(sailor.key, {}).get(position.lower(), {}).get(season, 0)
                    winPercent = winp_dict.get((sailor.key, position, season), 0)
                    
                    rows_to_insert.append((sailor.key,
                            team,
                            season,
                            position,
                            raceCount,
                            winPercent,
                            rankType))

    # Insert in batches
    for start in range(0, len(rows_to_insert), batch_size):
        batch = rows_to_insert[start:start + batch_size]
        try:
            with connection.cursor() as cursor:
                cursor.executemany("""
                    INSERT INTO SailorTeams
                        (sailorID, teamID, season, position, raceCount, winPercent, rankType)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        raceCount = VALUES(raceCount),
                        winPercent = VALUES(winPercent),
                        rankType = VALUES(rankType)
                """, batch)
            connection.commit()

        except mysql.connector.errors.IntegrityError as e:
            print("Batch insert failed:", e)
            raise e
    
def calculateAvgRatio(filtered_people: list[Sailor], winp_dict):
    winps = [winp_dict.get((s.key, pos, season), 0) for s in filtered_people for pos in ['skipper', 'crew'] for season, st in s.seasons[pos] ]
    # winps = []
    # for pos in ['skipper', 'crew']:
    #     for sailor in filtered_people:
    #         for season in sailor.seasons[pos]:
    #             print(sailor.key, season)
    #             winps.append(winp_dict.get((sailor.key, pos, season), 0))
    
    if len(winps) > 0:
        avgRatio = np.mean(winps)
    else:
        avgRatio = 0
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
    
def uploadTeams(people: dict[str, Sailor], outlinks_dict, racecounts_dict, winp_dict, connection, config: Config):
    for team, region in teamRegions.items():
        if team != 'Northeastern':
            continue
        sailors : list[Sailor] = [p for key, p in people.items() if team in p.teams]
        currentSailors : list[Sailor] = [p for p in sailors if p.isOnTeamInSeasons(team, config.targetSeasons)]
        
        topRating, topSkippers, topCrews = calculateTopSailors(currentSailors, outlinks_dict, False, False, config)
        topWomenRating, topWomenSkippers, topWomenCrews = calculateTopSailors(currentSailors, outlinks_dict, False, True, config)
        topRatingTR, topSkippersTR, topCrewsTR = calculateTopSailors(currentSailors, outlinks_dict, True, False, config)
        topWomenRatingTR, topWomenSkippersTR, topWomenCrewsTR = calculateTopSailors(currentSailors, outlinks_dict, True, True, config)
        
        avg = calculateAvgRating(currentSailors, config)
        avgRatio = calculateAvgRatio(currentSailors, winp_dict)

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

        
        uploadSailorTeams(sailors, team, [topSkippers, topWomenSkippers, topSkippersTR, topWomenSkippersTR], [topCrews,topWomenCrews, topCrewsTR, topWomenCrewsTR], racecounts_dict,  winp_dict, connection, config)
        
        # print("Updated ", team)