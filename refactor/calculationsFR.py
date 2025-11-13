import numpy as np
from regions import teamRegions
from config import Config
from Sailors import Sailor
from openskill.models import PlackettLuceRating

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

def updateRivals(sailor : Sailor, season, score, racers, scoreVals, pos):
    for other, otherScore in zip(racers, scoreVals):
        if other.key != sailor.key:
            if pos not in sailor.rivals:
                sailor.rivals[pos] = {}

            if other.key not in sailor.rivals[pos]:
                sailor.rivals[pos][other.key] = {'name': other.name, 'races': {}, 'team': other.teams[-1], 'wins': {}}

            if season not in sailor.rivals[pos][other.key]['races'].keys():
                sailor.rivals[pos][other.key]['races'][season] = 0
            if season not in sailor.rivals[pos][other.key]['wins'].keys():
                sailor.rivals[pos][other.key]['wins'][season] = 0

            sailor.rivals[pos][other.key]['races'][season] += 1
            if otherScore > score:
                sailor.rivals[pos][other.key]['wins'][season] += 1
                
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

def updateRaces(scores, racers, scoreVals, predictions, partnerKeys, partnerNames, startingRating, ratings, teams, race, scoring, season, date, womens, regattaAvg, pos, config : Config):
    if pos.lower() not in ['skipper', 'crew']:
        print("Pos is weird value in updateRaces ", pos)
    venue = scores['Venue'].iat[0]
    actualID = scores['raceID'].iat[0]

    # Make list of regions and combine PCCSC and NWICSA (those shouldnt count as cross regional for rating purposes)
    regions = [teamRegions[p.teams[-1]] if p.teams[-1]
               in teamRegions.keys() else None for p in racers]
    regions = ['PCCSC' if reg == 'NWICSA' else reg for reg in regions]

    # Check if race has any out of conference sailors
    isCross = True if len(set(regions)) > 1 else False

    # Loop through each sailor and the associated values
    for sailor, score, pred, partnerKey, partnerName, oldRating, new_rating, team in zip(racers, scoreVals, predictions, partnerKeys, partnerNames, startingRating, ratings, teams):

        outLinks = updateCrossLinks(sailor, isCross, regions, race, config)

        updateSeasons(sailor, season, team, pos)

        updateRivals(sailor, season, score, racers, scoreVals, pos)

        ratingpos = ('w' if womens else '') + ('s' if pos.lower() == 'skipper' else 'c') + 'r'
        
        # add race to each sailor's score
        sailor.races.append({
            # Need to rewrite to include DNF and such (correctly evaluating score but its hard to tell )
            'score': int(score),
            'pos': pos,
            'predicted': pred[0],
            'ratio': 1 - ((int(score) - 1) / (len(racers) - 1)),
            'regAvg': regattaAvg,
            'cross': isCross,
            'outLinks': outLinks,
            'ratingpos': ratingpos,
            'oldRating': oldRating,
            'newRating': new_rating[0].ordinal(target=config.targetElo, alpha=200 / config.model.sigma),
            'womens': womens,
            'date': date,
            'partner': {'name': partnerName, 'key': partnerKey},
            'venue': venue,
            'raceID': actualID,
            'pos': 'fleet',
            'scoring': scoring
        })
        
def getPartners(scores, config : Config):
    partnerKeys = scores['PartnerLink']
    partnerKeys = [pk if pk not in config.merges.keys() else config.merges[pk] for pk in partnerKeys]
    partnerNames = scores['Partner']
    
    return partnerKeys, partnerNames

def getRacers(people, scores, keys, teams, regatta):
    names = scores['Sailor']  # the sailor names
    
    racers = []
    try:
        # Grab people objects
        racers = [people[key] if key != 'Unknown'
                  and key is not None
                  else people[name + "-" + team] for key, name, team in zip(keys, names, teams)]
    except Exception as e:
        print(regatta)
        raise e

    return racers

def calculateFR(people : dict[str, Sailor], date, regatta, race, row, pos, scoring, season, regattaAvg, womens, config : Config):
    """Calculates new ratings and updates the rating, races, and rivals for a given fleet race. 
    """
    if pos.lower() not in ['skipper', 'crew']:
        print("Pos is weird value in main calcfr ", pos)
    scores = row[row['Position'] == pos]
    keys = scores['key']  # the sailor keys
    teams = scores['Team']  # the sailors team
    scoreVals = list(scores['Score'])  # the score values

    # check for invalid race conditions
    if len(keys) < 2:  # less than two sailors
        return
    if np.isnan(scoreVals[0]):  # B division did not complete the set
        return

    racers : list[Sailor] = getRacers(people, scores, keys, teams,regatta)

    partnerKeys , partnerNames = getPartners(scores, config)

    ratings = [[r.getRating(pos, 'fleet', womens)] for r in racers]

    startingRating = [r[0].ordinal(target=config.targetElo, alpha=config.alpha) for r in ratings]

    ratings = config.model.rate(ratings, scoreVals)

    predictions = config.model.predict_rank(ratings)

    updateRatings(racers, ratings, pos, womens)
    
    updateRaces(scores, racers, scoreVals, predictions, partnerKeys, partnerNames, startingRating, ratings, teams, race, scoring, season, date, womens, regattaAvg, pos, config)