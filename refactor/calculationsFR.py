import numpy as np
from regions import teamRegions
from config import Config
from Sailors import Sailor
from openskill.models import PlackettLuceRating
import time

def regionOf(team):
    """Conference for a team name, or None if unknown.

    PCCSC and NWICSA are treated as one region: the whole west coast should not count
    as cross-regional against itself for rating purposes.
    """
    reg = teamRegions.get(team)
    return 'PCCSC' if reg == 'NWICSA' else reg

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

def updateCrossLinks(sailor, team, isCross, regions, race, config : Config):
    outLinks = 0

    # `team` is the team this sailor sailed for in THIS race. Using sailor.teams[-1]
    # (their most recent team) would misattribute every race a transfer sailed before
    # transferring.
    sailorReg = regionOf(team)

    if sailorReg is None:
        print("Sailor's team not found in global team region list", team, race)
        return outLinks
    if None in regions:
        # print("None found in list of regions!!!")
        return outLinks

    # Only calculate number of cross regional sailors if it is the current season
    doCr = race.split("/")[0] in config.targetSeasons and isCross == 1
    
    if isCross: # and doCr
        # Calculate the number of sailors that are not in the sailor's region
        outLinks = sum(1 for reg in regions if reg != sailorReg)
        # Note: We don't need to filter out the sailor themselves from this list, because they will have the same region as themseleves so it will not be counted.
        sailor.cross += 1
        sailor.outLinks += outLinks
    
    return outLinks

def updateRaces(newRaces, venue, actualIDs, penalties, racers : list[Sailor], scoreVals, predictions, partnerKeys, partnerNames, startingRating, startingMuSigma, ratings, teams, teamBoatNames, boatType, race, scoring, season, date, womens, regattaAvg, pos, config : Config):
    if pos.lower() not in ['skipper', 'crew']:
        print("Pos is weird value in updateRaces ", pos)

    # Regions of everyone in this race, from the team each sailed for AT THE TIME
    # (the row's Team), not their most recent team.
    regions = [regionOf(t) for t in teams]

    # Check if race has any out of conference sailors
    isCross = True if len(set(regions)) > 1 else False

    # Loop through each sailor and the associated values
    for sailor, actualID, score, penalty, pred, partnerKey, partnerName, oldRating, oldMuSigma, new_rating, team, teamBoatName in zip(racers, actualIDs, scoreVals, penalties, predictions, partnerKeys, partnerNames, startingRating, startingMuSigma, ratings, teams, teamBoatNames):

        outLinks = updateCrossLinks(sailor, team, isCross, regions, race, config)

        updateSeasons(sailor, season, team, pos)

        ratingType = ('w' if womens else '') + ('s' if pos.lower() == 'skipper' else 'c') + 'r'

        # NOTE: sailor.outLinks is incremented inside updateCrossLinks. Do not add it
        # again here - that double-counted it, making the DB value 2x the parquet sum
        # and the effective eligibility threshold 75 rather than 150.

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
            'newRating': new_rating[0].ordinal(target=config.targetElo, alpha=config.alpha),
            'oldMu': oldMuSigma[0],
            'oldSigma': oldMuSigma[1],
            'newMu': new_rating[0].mu,
            'newSigma': new_rating[0].sigma,
            'regAvg': regattaAvg,
            'outLinks': outLinks,
            'calculatedAt': time.time()
        })

        sailor.recordRating(ratingType, date, new_rating[0].mu, new_rating[0].sigma)

def resetRacers(racers : list[Sailor], resetDate, ratingType, config : Config):
    """Roll each racer's rating back to just before resetDate, once per rating type."""
    if resetDate is None:
        return

    for racer in racers:
        if ratingType in racer.ratingTypesReset:
            continue

        racer.resetRatingToBeforeDate(resetDate, ratingType, config)

def getRacers(people : list[Sailor], names, keys, teams, regatta, resetDate, date, ratingType, config : Config):
    racers = []
    try:
        racers = [people[key] if key != 'Unknown'
                  and key is not None
                  else people[name + "-" + team] for key, name, team in zip(keys, names, teams)]
    except Exception as e:
        print(regatta)
        raise e

    resetRacers(racers, resetDate, ratingType, config)

    return racers

def calculateFR(newRaces : list, people : dict[str, Sailor], resetDate, date, regatta, race, row, pos, scoring, season, regattaAvg, womens, ratingType, config : Config):
    """Calculates new ratings and updates the rating, races, and rivals for a given fleet race. 
    """
    if pos.lower() not in ['skipper', 'crew']:
        print("Pos is weird value in main calcfr ", pos)
    scores = row[row['Position'] == pos]
    keys = scores['key']  # the sailor keys
    names = scores['Sailor']
    
    
    teams = scores['Team']  # the sailors team
    teamBoatNames = scores['TeamBoatName']  # the sailors team
    scoreVals = list(scores['Score'])  # the score values
    
    penalties = list(scores['penalty'])
    excusedPenalties = ["DNS", "BKD", "RDG", "BYE"]
    
    partnerKeys = scores['PartnerLink']
    partnerKeys = [pk if pk not in config.merges.keys() else config.merges[pk] for pk in partnerKeys] # handle merges
    partnerNames = scores['Partner']
    if "Unknown" in partnerKeys:
        partnerKeys = [pk if pk != "Unknown" else f"{pn}-{team}" for pk, pn, team in zip(partnerKeys, partnerNames, teams)]
    
    # check for invalid race conditions
    if len(keys) < 2:  # less than two sailors
        return
    if np.isnan(scoreVals[0]):  # B division did not complete the set
        return
    
    boatType = scores['Boat'].iat[0]
    venue = scores['Venue'].iat[0]
    # Races are grouped by adjusted_raceID, which drops the division for Combined
    # scoring, so one group can span A/B/C. Keep each sailor's own raceID instead of
    # the first row's, or every sailor gets recorded in the first row's division.
    actualIDs = list(scores['raceID'])

    racers : list[Sailor] = getRacers(people, names, keys, teams, regatta, resetDate, date, ratingType, config)

    ratings = [[r.getRating(pos, 'fleet', womens)] for r in racers]

    startingRating = [r[0].ordinal(target=config.targetElo, alpha=config.alpha) for r in ratings]
    # ordinal = alpha*(mu - 3*sigma) + targetElo is one equation in two unknowns, so the
    # raw (mu, sigma) has to be recorded separately or the rating cannot be restored on a
    # resume. Captured here because `ratings` is rebound to the new ratings below.
    startingMuSigma = [(r[0].mu, r[0].sigma) for r in ratings]

    # Determine active racers (those without penalties) for rating calculation
    active_mask = [p not in excusedPenalties for p in penalties]
    active_ratings = [r for r, m in zip(ratings, active_mask) if m]
    # Finishing places, lower-is-better. openskill's `ranks` uses that convention;
    # its `scores` parameter is the OPPOSITE (higher-is-better), so always pass this
    # by the `ranks` keyword or the whole model silently inverts.
    active_places = [s for s, m in zip(scoreVals, active_mask) if m]

    if len(active_ratings) < 2:
        return

    # Predict BEFORE rating, or the "prediction" has already seen the result.
    # calculationsTR.calculateTR is the reference pattern.
    predictions = config.model.predict_rank(ratings)

    active_ratings = config.model.rate(active_ratings, ranks=active_places)

    # Reconstruct full ratings list with updates only for active racers
    new_ratings = []
    active_iter = iter(active_ratings)
    for m in active_mask:
        if m:
            new_ratings.append(next(active_iter))
        else:
            new_ratings.append(ratings[len(new_ratings)])  # Keep old rating for penalized

    ratings = new_ratings

    updateRatings(racers, ratings, pos, womens)
    
    updateRaces(newRaces, venue, actualIDs, penalties, racers, scoreVals, predictions, partnerKeys, partnerNames, startingRating, startingMuSigma, ratings, teams, teamBoatNames, boatType, race, scoring, season, date, womens, regattaAvg, pos, config)