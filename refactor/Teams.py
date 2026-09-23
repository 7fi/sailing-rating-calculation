from config import Config
from regions import teamRegions
import datetime
import pandas as pd
import numpy as np
import mysql
from Sailors import Sailor, rankingKey, hasRating, publishedRating

def getOrderedSailors(people : list[Sailor], ratingType, pos, outlinks_dict, config : Config):
    numTops = config.numTops['tr' if 't' in ratingType else 'fr']['open' if 'w' not in ratingType else 'womens']
    isTR = 't' in ratingType
    outlinks_keys = outlinks_dict.keys()
    eligible_people = [p for p in people
                        if p.isRankEligible(config.targetSeasons, pos, config.gradCutoff, outLinks= outlinks_dict[p.key]
                        if p.key in outlinks_keys else None, needsOutlinks= not isTR,
                        requiredOutLinks=config.requiredOutLinks,
                        ratingType=ratingType if (config.useWHR and not isTR) else None,
                        maxRatingSE=config.maxRatingSE if (config.useWHR and not isTR) else None)
                        and hasRating(p, ratingType, config)]
    # Team ratings use the POINT estimate, not the interval lower bound the individual
    # leaderboard uses. Subtracting 1.96*SE from every sailor ranks teams partly by how
    # precisely their sailors are known, which systematically favours teams whose
    # sailors race more: USC's top three had 65/56/29 races and Northeastern's 21/10/10,
    # so USC won on lower bound (1799 vs 1790) while losing on point estimate
    # (1863 vs 1895). The joint fit's point estimate is already shrunk toward the prior
    # for low-data sailors, so the extra penalty is double counting.
    if config.teamRatingUseLowerBound:
        def value(p):
            v = p.whrRating(ratingType, 'lcb') if config.useWHR else None
            return v if v is not None else publishedRating(p, ratingType, config)
    else:
        value = lambda p: publishedRating(p, ratingType, config)

    orderedSailors = sorted(eligible_people, key=value, reverse=True)
    top = orderedSailors[:numTops]

    # Sum and count are returned together so the caller can divide by the number of
    # sailors actually found. Dividing by numTops regardless scored a missing sailor
    # as a rating of ZERO rather than as unknown, which crushed thin rosters - 41 of
    # 161 teams have fewer than three eligible open skippers.
    sailorSum = sum(value(p) for p in top)
    topSailors = [{'name': p.name, 'key': p.key, ratingType: value(p)} for p in top]
    return topSailors, sailorSum, len(top)

def calculateTopSailors(filtered_people, outlinks_dict, isTeamRace, isWomens, config: Config):
    prefix = 't' if isTeamRace else ''
    if isWomens:
        prefix = 'w' + prefix
    topSkippers, topSkippersSum, nSkippers = getOrderedSailors(filtered_people, prefix + 'sr', 'skipper', outlinks_dict, config)
    topCrews, topCrewsSum, nCrews = getOrderedSailors(filtered_people, prefix + 'cr', 'crew', outlinks_dict, config)

    found = nSkippers + nCrews
    topRating = (topSkippersSum + topCrewsSum) / found if found else 0
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
                if seasonTeam != team: # Only insert if sailor was actually on this team in this season
                    continue
                
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

FLEET_TYPES = ('sr', 'cr', 'wsr', 'wcr')
TEAM_TYPES = ('tsr', 'tcr', 'wtsr', 'wtcr')


def calculateAvgRating(people : list[Sailor], config:Config):
    """Mean over sailors of their best fleet rating and their best team-race rating.

    Goes through publishedRating, so it follows whichever model is in use. It used to
    read the openskill ordinals directly - and the team-race block actually read the
    FLEET attributes, so team ratings never contributed and fleet ones were counted
    twice.
    """
    ratings = []
    for p in people:
        for group in (FLEET_TYPES, TEAM_TYPES):
            vals = [publishedRating(p, rt, config) for rt in group
                    if hasRating(p, rt, config)]
            ratings.append(max(vals) if vals else 0)

    return sum(ratings) / len(ratings) if len(ratings) > 0 else 0
    
def uploadTeams(people: dict[str, Sailor], outlinks_dict, racecounts_dict, winp_dict, team_link_map, connection, config: Config):
    for team, region in teamRegions.items():
        # if team != 'Northeastern':
        #     continue
        sailors : list[Sailor] = [p for key, p in people.items() if team in p.teams]
        currentSailors : list[Sailor] = [p for p in sailors if p.isOnTeamInSeasons(team, config.targetSeasons)]
        
        topRating, topSkippers, topCrews = calculateTopSailors(currentSailors, outlinks_dict, False, False, config)
        topWomenRating, topWomenSkippers, topWomenCrews = calculateTopSailors(currentSailors, outlinks_dict, False, True, config)
        topRatingTR, topSkippersTR, topCrewsTR = calculateTopSailors(currentSailors, outlinks_dict, True, False, config)
        topWomenRatingTR, topWomenSkippersTR, topWomenCrewsTR = calculateTopSailors(currentSailors, outlinks_dict, True, True, config)
        
        avg = calculateAvgRating(currentSailors, config)
        avgRatio = calculateAvgRatio(currentSailors, winp_dict)
        link = team_link_map.get(team, '')

        with connection.cursor() as cursor:
            cursor.execute("""
                INSERT INTO Teams
                    (teamID, teamName, link, topFleetRating, topWomenRating, topTeamRating,
                    topWomenTeamRating, avgRating, avgRatio, region)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    topFleetRating = VALUES(topFleetRating),
                    topWomenRating = VALUES(topWomenRating),
                    topTeamRating = VALUES(topTeamRating),
                    topWomenTeamRating = VALUES(topWomenTeamRating),
                    avgRating = VALUES(avgRating),
                    avgRatio = VALUES(avgRatio)
            """, (team, team, link, topRating, topWomenRating, topRatingTR,
                topWomenRatingTR, avg, avgRatio, region))
        connection.commit()

        
        uploadSailorTeams(sailors, team, [topSkippers, topWomenSkippers, topSkippersTR, topWomenSkippersTR], [topCrews,topWomenCrews, topCrewsTR, topWomenCrewsTR], racecounts_dict,  winp_dict, connection, config)
        
        # print("Updated ", team)