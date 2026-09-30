from config import Config
from regions import teamRegions
import datetime
import pandas as pd
import numpy as np
import mysql
from Sailors import Sailor, rankingKey, hasRating, publishedRating

def getOrderedSailors(people : list[Sailor], ratingType, pos, outlinks_dict, config : Config):
    """Each school's top N sailors for one rating type, and their rating sum.

    Two deliberate differences from the individual leaderboard:

    * **No uncertainty gate.** Measured against actual cross-region regatta results,
      applying the SE gate here made things worse (PCCSC bias +0.112 vs +0.084),
      because it strips a team's weaker sailors and leaves only their best. The gate
      exists to decide who is confidently *ranked*, not how strong a team is.
    * **A missing top-N slot counts as the population mean**, not as zero and not as
      absent. Dividing by numTops regardless scored a thin roster as if its third
      sailor had a rating of 0; dividing by however many were found rewarded thin
      rosters instead, letting a team with two strong sailors and no depth outrank a
      team with three.

    Ordering and summing both use publishedRating, i.e. the empirical-Bayes shrunk
    estimate, so a sailor with one regatta contributes close to average rather than
    at face value.
    """
    numTops = config.numTops['tr' if 't' in ratingType else 'fr']['open' if 'w' not in ratingType else 'womens']
    isTR = 't' in ratingType
    eligible_people = [p for p in people
                       if p.isRankEligible(config.targetSeasons, pos, config.gradCutoff,
                                           needsOutlinks=False)
                       and hasRating(p, ratingType, config)]

    # Team ratings use their own statistic (default 'shrunk'), independent of what the
    # individual leaderboard publishes.
    stat = getattr(config, "teamRatingStatistic", "shrunk")
    def teamValue(p):
        v = p.whrRating(ratingType, stat) if config.useWHR else None
        return v if v is not None else publishedRating(p, ratingType, config)

    orderedSailors = sorted(eligible_people, key=teamValue, reverse=True)
    top = orderedSailors[:numTops]

    popMean = None
    for p in eligible_people:
        popMean = p.whrRating(ratingType, "popmean")   # same for rating/shrunk
        if popMean is not None:
            break
    if popMean is None:
        popMean = config.whrTargetMean

    sailorSum = sum(teamValue(p) for p in top)
    sailorSum += (numTops - len(top)) * popMean
    topSailors = [{'name': p.name, 'key': p.key,
                   ratingType: publishedRating(p, ratingType, config)} for p in top]
    return topSailors, sailorSum, numTops


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