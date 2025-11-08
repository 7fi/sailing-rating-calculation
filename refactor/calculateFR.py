import numpy as np
from regions import teamRegions

def updateRatings(racers, ratings, womens):
    for racer, new_rating in zip(racers, ratings):
        if type == 'Skipper':
            if womens:
                racer.wsr = new_rating[0]
            else:
                racer.sr = new_rating[0]
        else:
            if womens:
                racer.wcr = new_rating[0]
            else:
                racer.cr = new_rating[0]

def updateRaces(scores, racers, scoreVals, predictions, partnerKeys, partnerNames, startingRating, ratings, teams, race, scoring, season, date, womens, regattaAvg, config):
    # Common values for each sailor
    venue = scores['Venue'].iat[0]
    scoring = scores['Scoring'].iat[0]
    actualID = scores['raceID'].iat[0]

    # Make list of regions and combine PCCSC and NWICSA (those shouldnt count as cross regional for rating purposes)
    regions = [teamRegions[p.teams[-1]] if p.teams[-1]
               in teamRegions.keys() else None for p in racers]
    regions = ['PCCSC' if reg == 'NWICSA' else reg for reg in regions]

    # Check if race has any out of region sailors
    isCross = 1 if len(set(regions)) > 1 else 0

    # Only calculate number of cross regional sailors if it is the current season
    doCr = race.split("/")[0] in config.targetSeasons and isCross == 1

    # Loop through each sailor and the associated values
    for sailor, score, pred, partnerKey, partnerName, oldRating, new_rating, team in zip(racers, scoreVals, predictions, partnerKeys, partnerNames, startingRating, ratings, teams):
        outLinks = 0

        partnerKey = partnerKey if partnerKey not in config.merges.keys(
        ) else config.merges[partnerKey]

        if (isCross == 1):
            # Calculate the number of sailors that are not in the sailor's region
            outLinks = sum(1 for reg in regions  # adds 1 each time that a region in the regatta ...
                           if reg is not None  # Double check that the region is not none # first double check that the sailor's team is in the list of regions
                           and sailor.teams[-1] in teamRegions.keys()
                           # The sailor's region is not the same as the opponent)
                           and ('PCCSC' if reg == 'NWICSA' else reg) != ('PCCSC' if teamRegions[sailor.teams[-1]] == 'NWICSA' else teamRegions[sailor.teams[-1]]))
            # Note: We don't need to filter out the sailor themselves from this list, because they will have the same region as themseleves so it will not be counted.
            sailor.cross += 1
            sailor.outLinks += outLinks

        # update sailor's seasons
        if season not in [s[0] for s in sailor.seasons[type.lower()]]:
            sailor.seasons[type.lower()].append((season, team))
            if(sailor.key == 'carter-anderson'):
                print(sailor.seasons)

        # calculate rivals
        for other, otherScore in zip(racers, scoreVals):
            if other.key != sailor.key:
                if type not in sailor.rivals:
                    sailor.rivals[type] = {}

                if other.key not in sailor.rivals[type]:
                    sailor.rivals[type][other.key] = {
                        'name': other.name, 'races': {}, 'team': other.teams[-1], 'wins': {}}

                if season not in sailor.rivals[type][other.key]['races'].keys():
                    sailor.rivals[type][other.key]['races'][season] = 0
                if season not in sailor.rivals[type][other.key]['wins'].keys():
                    sailor.rivals[type][other.key]['wins'][season] = 0

                sailor.rivals[type][other.key]['races'][season] += 1
                if otherScore > score:
                    sailor.rivals[type][other.key]['wins'][season] += 1

        ratingType = 'wsr' if womens else 'sr' if type == 'Skipper' else 'wcr' if womens else 'cr'
        # add race to each sailor's score
        sailor.races.append({
            # Need to rewrite to include DNF and such (correctly evaluating score but its hard to tell )
            'score': int(score),
            'pos': type,
            'predicted': pred[0],
            # Calculate ratio here
            'ratio': 1 - ((int(score) - 1) / (len(racers) - 1)),
            'regAvg': regattaAvg,
            'cross': isCross,
            'outLinks': outLinks,
            'sr': sailor.sr.ordinal(target=config.targetElo, alpha=200 / config.model.sigma),
            'srmu': sailor.sr.mu,
            'srsig': sailor.sr.sigma,
            'ratingType': ratingType,
            'oldRating': oldRating,
            'newRating': new_rating[0].ordinal(target=config.targetElo, alpha=200 / config.model.sigma),
            'womens': womens,
            'date': date,
            'partner': {'name': partnerName, 'key': partnerKey},
            'venue': venue,
            'raceID': actualID,
            'type': 'fleet',
            'scoring': scoring
        })

def calculateFR(people, date, regatta, race, row, type, scoring, season, residuals, regattaAvg, womens, config):
    # Filter by current position
    scores = row[row['Position'] == type]
    # Grab lists for each datapoint
    keys = scores['key']  # the sailor keys
    names = scores['Sailor']  # the sailor names
    teams = scores['Team']  # the sailors team
    scoreVals = list(scores['Score'])  # the score values

    # check for invalid race conditions
    if len(keys) < 2:  # less than two sailors
        return
    if np.isnan(scoreVals[0]):  # B division did not complete the set
        return

    racers = []
    try:
        # Grab people objects
        racers = [people[key] if key != 'Unknown'
                  and key is not None
                  else people[name + "-" + team] for key, name, team in zip(keys, names, teams)]
    except Exception as e:
        print(regatta)
        raise e

    partnerKeys = scores['PartnerLink']
    partnerNames = scores['Partner']

    # Seperate out the openskill rating objects for use in the model
    ratings = [[r.getRating(type, 'fleet', womens)] for r in racers]

    # grab starting rating values for change calculation later
    startingRating = [r[0].ordinal(target=config.targetElo, alpha=config.alpha) for r in ratings]

    # Rate using the model
    ratings = config.model.rate(ratings, scoreVals)

    # Make predictions
    predictions = config.model.predict_rank(ratings)

    # calculate error and add to list (residuals)
    for pred, score in zip(predictions, scoreVals):
        residuals.append(score - pred[0])

    # Update racers' ratings
    updateRatings(racers, ratings, womens)
    
    updateRaces(scores, racers, scoreVals, predictions, partnerKeys, partnerNames, startingRating, ratings, teams, race, scoring, season, date, womens, regattaAvg, config)