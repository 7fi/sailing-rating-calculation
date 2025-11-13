from config import Config
from calculationsFR import updateSeasons
from Sailors import Sailor

def updateRatings(womens, teamARacers, teamBRacers, ratings, pos):
    for team, newRatings in zip([teamARacers, teamBRacers], ratings):
        for racer, new_rating in zip(team, newRatings):
            if pos == 'Skipper':
                if womens:
                    racer.wtsr = new_rating
                else:
                    racer.tsr = new_rating
            else:
                if womens:
                    racer.wtcr = new_rating
                else:
                    racer.tcr = new_rating

def updateRivals(racer, oppRacers, season, pos, toutcome):
    for opp in oppRacers:
        if pos not in racer.rivals:
            racer.rivals[pos] = {}

        if opp.key not in racer.rivals[pos]:
            racer.rivals[pos][opp.key] = {
                'name': opp.name, 'races': {}, 'team': opp.teams[-1], 'wins': {}}

        if season not in racer.rivals[pos][opp.key]['races'].keys():
            racer.rivals[pos][opp.key]['races'][season] = 0
        if season not in racer.rivals[pos][opp.key]['wins'].keys():
            racer.rivals[pos][opp.key]['wins'][season] = 0

        racer.rivals[pos][opp.key]['races'][season] += 1
        if toutcome == 'win':
            racer.rivals[pos][opp.key]['wins'][season] += 1
            
def getPartners(boats, pos):
    partnerKeys = []
    partnerNames = []
    if pos.lower() == 'skipper':
        partnerKeys = [boat['crewKey'] if boat['crewKey'] is not None else 'Unknown' for boat in boats]
        partnerNames = [boat['crewName'] if boat['crewName'] is not None else 'Unknown' for boat in boats]
    elif pos.lower() == 'crew':
        partnerKeys = [boat['skipperKey'] if boat['skipperKey']is not None else 'Unknown' for boat in boats]
        partnerNames = [boat['skipperName'] if boat['skipperName']is not None else 'Unknown' for boat in boats]
    return partnerKeys, partnerNames

def getTeamVals(row, people : dict[str, Sailor], rowVal : str, rowBoatVal : str, womens, pos, config:Config):
    teamName = row[rowVal].iat[0]
    teamKeys = [boat[pos.lower() + 'Key'] if boat[pos.lower() + 'Key']
                 is not None else 'Unknown' for boat in row[rowBoatVal].iat[0]]
    # teamANames = [boat[type.lower() + 'Name'] for boat in row[rowBoatVal].iat[0]]
    for oldkey, newkey in config.merges.items():
        if oldkey in teamKeys:
            teamKeys = [k if k != oldkey else newkey for k in teamKeys]
    teamRacers = [people[key]
                   for key in teamKeys if 'Unknown' not in key and key in people.keys()]
    
    teamRatings = [r.getRating(pos, 'team', womens) for r in teamRacers]
    
    return teamName, teamRacers, teamRatings

def updateRacesForTeam(tLetter, index, racers, oppRacers, boats, starting, teamName, pos, season, womens, row, date, predictions, venue, regattaAvg, ratings, config : Config):
    oLetter = 'A' if tLetter == 'B' else 'B'
    tscore = row['team' + tLetter + 'Score'].iat[0]
    toutcome = row['team'+ tLetter + 'Outcome'].iat[0]
    oppt = row['team'+ oLetter + 'Name'].iat[0]
    oppn = row['team'+ oLetter + 'Nick'].iat[0]
    
    partnerKeys, partnerNames = getPartners(boats, pos)

    new_ratings = [r.ordinal(target=config.targetElo, alpha=config.alpha) for r in ratings]

    for racer, partnerKey, partnerName, oldRating, new_rating in zip(racers, partnerKeys, partnerNames, starting, new_ratings):

        partnerKey = partnerKey if partnerKey not in config.merges.keys() else config.merges[partnerKey]

        updateRivals(racer, oppRacers, season, pos, toutcome)

        updateSeasons(racer, season, teamName, pos)

        ratingType = 'wtsr' if womens else 'tsr' if pos == 'Skipper' else 'wtcr' if womens else 'tcr'
        
        racer.races.append({
            'raceID': row['raceID'].iat[0], 'raceNum': int(row['raceNum'].iat[0]), 'round':  row['round'].iat[0],
            'pos': pos,
            'date': date,
            'womens': womens,
            'partner': {'key': partnerKey, 'name': partnerName},
            'opponentTeam': oppt,
            'opponentNick': oppn,
            'score': tscore,
            'outcome': toutcome,
            'ratingType': ratingType,
            'oldRating': oldRating,
            'newRating': new_rating,
            'predicted': 'win' if predictions[index][0] == 1 else 'lose',
            'regAvg': regattaAvg,
            'venue': venue,
            'type': 'team'
            })

def calculateTR(people : dict[str, Sailor], date : str, row, pos : str, season : str, regattaAvg : float, womens : bool, config : Config):
    venue = row['Venue'].iat[0]

    teamAName, teamARacers, teamARatings = getTeamVals(row, people, 'teamAName', 'teamABoats', womens, pos, config)
    teamBName, teamBRacers, teamBRatings = getTeamVals(row, people, 'teamBName', 'teamBBoats', womens, pos, config)

    startingARating = [r.ordinal(target=config.targetElo, alpha=config.alpha) for r in teamARatings]
    startingBRating = [r.ordinal(target=config.targetElo, alpha=config.alpha) for r in teamBRatings]

    if len(teamARatings) < 1 or len(teamBRatings) < 1:
        # print("not enough sailors in this race, skipping", row['raceID'].iat[0])
        return

    ranks = [1 if row['teamAOutcome'].iat[0] == 'win' else 2, 
             1 if row['teamBOutcome'].iat[0] == 'win' else 2]
    
    predictions = config.model.predict_rank([teamARatings, teamBRatings])

    teamARatings, teamBRatings = config.model.rate([teamARatings, teamBRatings],ranks=ranks)
    
    updateRacesForTeam('A', 0, teamARacers, teamBRacers, row['teamABoats'].iat[0], startingARating, teamAName, pos, season, womens, row, date, predictions, venue, regattaAvg, teamARatings, config)
    
    updateRacesForTeam('B', 1, teamBRacers, teamARacers, row['teamBBoats'].iat[0], startingBRating, teamBName, pos, season, womens, row, date, predictions, venue, regattaAvg, teamBRatings, config)