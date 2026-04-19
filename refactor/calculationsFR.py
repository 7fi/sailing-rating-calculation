import numpy as np
from regions import teamRegions
from config import Config
from Sailors import Sailor
from openskill.models import PlackettLuceRating
import time

def updateRatings(racers : list[Sailor], ratings : list[PlackettLuceRating], pos, womens):
    for racer, new_rating in zip(racers, ratings):
        if pos == 'Skipper':
            if womens:
                racer.wsr = new_rating[0]
            else:
                racer.sr = new_rating[0]
        else:
            if womens:
                racer.wcr = new_rating[0]
            else:
                racer.cr = new_rating[0]
                
def updateSeasons(sailor, season, team, pos):
    if season not in [s[0] for s in sailor.seasons[pos.lower()]]:
        sailor.seasons[pos.lower()].append((season, team))

def updateCrossLinks(sailor, isCross, regions, race, config : Config):
    outLinks = 0
    
    if sailor.teams[-1] not in teamRegions.keys():
        print("Sailor's team not found in global team region list", sailor.teams[-1], race)
        return outLinks
    if None in regions:
        # print("None found in list of regions!!!")
        return outLinks
    
    # Only calculate number of cross regional sailors if it is the current season
    doCr = race.split("/")[0] in config.targetSeasons and isCross == 1
    sailorReg = ('PCCSC' if teamRegions[sailor.teams[-1]] == 'NWICSA' else teamRegions[sailor.teams[-1]])
    
    if isCross: # and doCr
        # Calculate the number of sailors that are not in the sailor's region
        outLinks = sum(1 for reg in regions if reg != sailorReg)
        # Note: We don't need to filter out the sailor themselves from this list, because they will have the same region as themseleves so it will not be counted.
        sailor.cross += 1
        sailor.outLinks += outLinks
    
    return outLinks

def updateRaces(newRaces, venue, actualID, penalties, racers : list[Sailor], scoreVals, predictions, partnerKeys, partnerNames, startingRating, ratings, teams, teamBoatNames, boatType, race, scoring, season, date, womens, regattaAvg, pos, config : Config):
    if pos.lower() not in ['skipper', 'crew']:
        print("Pos is weird value in updateRaces ", pos)

    # Make list of regions and combine PCCSC and NWICSA (those shouldnt count as cross regional for rating purposes)
    regions = [teamRegions[p.teams[-1]] if p.teams[-1]
               in teamRegions.keys() else None for p in racers]
    regions = ['PCCSC' if reg == 'NWICSA' else reg for reg in regions]

    # Check if race has any out of conference sailors
    isCross = True if len(set(regions)) > 1 else False

    # Loop through each sailor and the associated values
    for sailor, score, penalty, pred, partnerKey, partnerName, oldRating, new_rating, team, teamBoatName in zip(racers, scoreVals, penalties, predictions, partnerKeys, partnerNames, startingRating, ratings, teams, teamBoatNames):

        outLinks = updateCrossLinks(sailor, isCross, regions, race, config)

        updateSeasons(sailor, season, team, pos)

        ratingType = ('w' if womens else '') + ('s' if pos.lower() == 'skipper' else 'c') + 'r'
        
        sailor.outLinks += outLinks
        
        newRaces.append({
            'raceID': actualID,
            'season': actualID.split("/")[0],
            'regatta': actualID.split("/")[1],
            'raceNumber': actualID.split("/")[2][:-1],
            'division': actualID.split("/")[2][-1],
            'sailorID': sailor.key,
            'partnerID': partnerKey,
            'partnerName': partnerName,
            'score': int(score),
            'predicted': pred[0],
            'ratio': 1 - ((int(score) - 1) / (len(racers) - 1)),
            'penalty': penalty,
            'position': pos,
            'date': date,
            'scoring': scoring,
            'venue': venue,
            'boat': boatType,
            'boatName': teamBoatName,
            'ratingType': ratingType,
            'oldRating': oldRating,
            'newRating': new_rating[0].ordinal(target=config.targetElo, alpha=200 / config.model.sigma),
            'regAvg': regattaAvg,
            'outLinks': outLinks,
            'calculatedAt': time.time()
        })

def getRacers(people : list[Sailor], names, keys, teams, regatta, resetDate, date, ratingType):
    racers = []
    try:
        racers = [people[key] if key != 'Unknown'
                  and key is not None
                  else people[name + "-" + team] for key, name, team in zip(keys, names, teams)]
    except Exception as e:
        print(regatta)
        raise e

    # Handle resetting if necessary: 
    if resetDate is None:
        return racers

    for racer in racers:
        if ratingType in racer.ratingTypesReset:
            continue
        
        racer.resetRatingToBeforeDate(resetDate, ratingType)

    return racers

def calculateFR(newRaces : list, people : dict[str, Sailor], resetDate, date, regatta, race, row, pos, scoring, season, regattaAvg, womens, ratingType, config : Config):
    """Calculates new ratings and updates the rating, races, and rivals for a given fleet race. 
    """
    if pos.lower() not in ['skipper', 'crew']:
        print("Pos is weird value in main calcfr ", pos)
    scores = row[row['Position'] == pos]
    keys = scores['key']  # the sailor keys
    names = scores['Sailor']
    
    partnerKeys = scores['PartnerLink']
    partnerKeys = [pk if pk not in config.merges.keys() else config.merges[pk] for pk in partnerKeys]
    partnerNames = scores['Partner']
    
    teams = scores['Team']  # the sailors team
    teamBoatNames = scores['TeamBoatName']  # the sailors team
    scoreVals = list(scores['Score'])  # the score values
    penalties = list(scores['penalty'])
    
    excusedPenalties = ["DNS", "BKD", "RDG", "BYE"]

    # check for invalid race conditions
    if len(keys) < 2:  # less than two sailors
        return
    if np.isnan(scoreVals[0]):  # B division did not complete the set
        return
    
    boatType = scores['Boat'].iat[0]
    venue = scores['Venue'].iat[0]
    actualID = scores['raceID'].iat[0]

    racers : list[Sailor] = getRacers(people, names, keys, teams, regatta, resetDate, date, ratingType)

    ratings = [[r.getRating(pos, 'fleet', womens)] for r in racers]

    startingRating = [r[0].ordinal(target=config.targetElo, alpha=config.alpha) for r in ratings]

    # Determine active racers (those without penalties) for rating calculation
    active_mask = [p not in excusedPenalties for p in penalties]
    active_ratings = [r for r, m in zip(ratings, active_mask) if m]
    active_scores = [s for s, m in zip(scoreVals, active_mask) if m]

    if len(active_ratings) < 2:
        return

    active_ratings = config.model.rate(active_ratings, active_scores)

    # Reconstruct full ratings list with updates only for active racers
    new_ratings = []
    active_iter = iter(active_ratings)
    for m in active_mask:
        if m:
            new_ratings.append(next(active_iter))
        else:
            new_ratings.append(ratings[len(new_ratings)])  # Keep old rating for penalized

    ratings = new_ratings

    predictions = config.model.predict_rank(ratings)

    updateRatings(racers, ratings, pos, womens)
    
    updateRaces(newRaces, venue, actualID, penalties, racers, scoreVals, predictions, partnerKeys, partnerNames, startingRating, ratings, teams, teamBoatNames, boatType, race, scoring, season, date, womens, regattaAvg, pos, config)