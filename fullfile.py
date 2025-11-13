import pandas as pd
import numpy as np
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from datetime import datetime, timedelta, date

import requests
from bs4 import BeautifulSoup
from collections import defaultdict
from refactor.AsyncScraper import runFleetScrape
from refactor.dataScraper import runSailorData
from refactor.TRScraper import scrapeTR
import json
import time

from refactor.regions import teamRegions
import mysql.connector

from openskill.models import PlackettLuce, BradleyTerryFull

class Sailor:
    def __init__(self, name, key, year, links, teams, seasons, races, rivals, sr, cr, wsr, wcr, tsr, tcr, wtsr, wtcr, gender=""):
        self.name = name
        self.key = key
        self.gender = gender
        self.year = year
        self.links = links
        self.teams = teams

        self.cross = 0
        self.outLinks = 0

        self.skipperRank = 0
        self.crewRank = 0
        self.womenSkipperRank = 0
        self.womenCrewRank = 0

        self.skipperRankTR = 0
        self.crewRankTR = 0
        self.womenSkipperRankTR = 0
        self.womenCrewRankTR = 0

        self.seasons = seasons
        self.races = races
        self.rivals = rivals

        # fleet racing
        self.sr = sr
        self.cr = cr
        self.wsr = wsr
        self.wcr = wcr
        # Team racing
        self.tsr = tsr
        self.tcr = tcr
        self.wtsr = wtsr
        self.wtcr = wtcr

        self.avgSkipperRatio = 0
        self.avgCrewRatio = 0

    def hasTargetSeasons(self, targetSeasons, pos):
        return not set([s[0] for s in self.seasons[pos]]).isdisjoint(targetSeasons)

    def __repr__(self):
        return f"{self.name}: {self.teams}, {str(self.sr.ordinal(target=targetElo, alpha=200/model.sigma))} {str(self.cr.ordinal(target=targetElo, alpha=200/model.sigma))} {self.seasons} {len(self.races)}"


def adjust_race_id(row):
    if row['Scoring'] == 'Combined':
        # Remove the last character (A/B) for combined scoring
        return row['raceID'][:-1]
    return row['raceID']


cred = credentials.Certificate("thecrowsnestapp-creds.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Create a connection
connection = mysql.connector.connect(
    host="localhost",
    port=3308,
    user="root",
    password="password",
    database="crowsnest"
)

cursor = connection.cursor()

model = PlackettLuce(beta=25.0/120.0)
targetElo = 1000

targetSeasons = ['f25']
targetTRSeasons = ['s25']
gradCutoff = 2025
# baseSigma = baseElo // 3
# offset = baseElo * 2

calc_all = True

merges = {'carter-anderson-2027': 'carter-anderson',
          'elliott-bates-2021': 'elliott-bates',
          'ian-hopkins-guerra-2026': 'ian-hopkins-guerra',
          'connor-nelson-2024': 'connor-nelson', 
          'Gavin Hudson-Northeastern': 'gavin-hudson', 
          'Nathalie Caudron-Northeastern': 'nathalie-caudron'}


def setupPeople(df_sailor_ratings, df_sailor_info, df_races_full):

    # read from file first
    if not calc_all:
        people = {row.key: Sailor(row.Sailor, row.key, row.GradYear, row.Links, row.Teams, row.Seasons, row.Races, row.Rivals,
                                  model.create_rating(
                                      [row.srMU, row.srSigma], row.key),
                                  model.create_rating(
                                      [row.crMU, row.crSigma], row.key),
                                  model.create_rating(
                                      [row.wsrMU, row.wsrSigma], row.key),
                                  model.create_rating(
                                      [row.wcrMU, row.wcrSigma], row.key),
                                  model.create_rating(
                                      [row.tsrMU, row.tsrSigma], row.key),
                                  model.create_rating(
                                      [row.tcrMU, row.tcrSigma], row.key),
                                  model.create_rating(
                                      [row.wtsrMU, row.wtsrSigma], row.key),
                                  model.create_rating(
                                      [row.wtcrMU, row.wtcrSigma], row.key),
                                  gender=row.gender) for row in df_sailor_ratings.itertuples()}

        for key in df_sailor_info['key']:
            if key not in people.keys():
                row = df_sailor_info.loc[df_sailor_info['key'] == key].iloc[0]
                people[key] = Sailor(row.name, row.key, row.year, [row.link], [row.team], {'skipper': [], 'crew': []}, [], {},
                                     model.rating(name=row.name),
                                     model.rating(name=row.name),
                                     model.rating(name=row.name),
                                     model.rating(name=row.name),
                                     model.rating(name=row.name),
                                     model.rating(name=row.name),
                                     model.rating(name=row.name),
                                     model.rating(name=row.name),
                                     gender=row.gender)
    else:
        # Set up people dictionary
        # people = {row.key: Sailor(row.name, row.key, row.year, [row.link], [row.team], {'skipper': [], 'crew': []}, gender=row.gender) for row in df_sailor_info.itertuples()}
        people = {row.key: Sailor(row.name, row.key, row.year, [row.link], [row.team], {'skipper': [], 'crew': []}, [], {},
                                  model.rating(name=row.name),
                                  model.rating(name=row.name),
                                  model.rating(name=row.name),
                                  model.rating(name=row.name),
                                  model.rating(name=row.name),
                                  model.rating(name=row.name),
                                  model.rating(name=row.name),
                                  model.rating(name=row.name),
                                  gender=row.gender) for row in df_sailor_info.itertuples()}

    # merge sailor objects
    for oldkey, newkey in merges.items():
        new = people[newkey]
        old = people[oldkey]
        new.links = new.links + old.links
        if old.teams != new.teams:
            new.teams = new.teams + old.teams
        del people[oldkey]
        
        df_races_full['key'] = df_races_full['key'].replace(oldkey, newkey)

    return people, df_races_full


def validPerson(p, type):
    # print((2000 + int(p.year.split()[0]) if isinstance(p.year, str) and len(p.year.split()) > 1 else int(p.year)))
    return (p.cross > 20
            and p.outLinks > 70
            # if sum([race['cross'] for race in p.races if 'cross' in race.keys()]) > 20
            # and sum([race['outLinks'] for race in p.races if 'outLinks' in race.keys()]) > 70
            and not p.hasTargetSeasons(targetSeasons, type)
            # and (2000 + int(p.year.split()[0]) > gradCutoff if isinstance(p.year, str) and len(p.year.split()) > 1 else int(p.year) > gradCutoff)
            # and sum([p['raceCount'][seas] for seas in targetSeasons if seas in p['raceCount'].keys()]) > 5
            )


def getTeamRatings(people, team, targetSeasons):
    filtered_people = [p for p in people.values() if team in p.teams]
    current = [p for p in filtered_people if not p.hasTargetSeasons(targetSeasons, 'skipper') or not p.hasTargetSeasons(targetSeasons, 'crew')]

    numTops = 3
    if len(current) > 0:
        topSkippers = sorted([p for p in filtered_people
                              if validPerson(p, 'skipper')],
                             key=lambda x: x.sr.ordinal(
                                 target=targetElo, alpha=200 / model.sigma),
                             reverse=True)
        topSkipperSum = sum([p.sr.ordinal(target=targetElo, alpha=200 / model.sigma)
                             for p in topSkippers[:numTops]])
        topSkippers = [{'name': p.name, 'key': p.key,
                        'mu': p.sr.mu, 'sigma': p.sr.sigma} for p in topSkippers]

        topCrews = sorted([p for p in filtered_people
                           if validPerson(p, 'crew')],
                          key=lambda x: x.cr.ordinal(
                              target=targetElo, alpha=200 / model.sigma),
                          reverse=True)
        topCrewsSum = sum([p.cr.ordinal(target=targetElo, alpha=200 / model.sigma)
                           for p in topCrews[:numTops]])
        topCrews = [{'name': p.name, 'key': p.key, 'mu': p.cr.mu,
                     'sigma': p.cr.sigma} for p in topCrews]

        topRating = (topSkipperSum + topCrewsSum) / (numTops * 2)
        # print(topSkippers, topCrews, topRating)

        # Women's
        numTops = 2
        topWomenSkippers = sorted([p for p in filtered_people if validPerson(p, 'skipper')],
                                  key=lambda x: x.wsr.ordinal(
                                      target=targetElo, alpha=200 / model.sigma),
                                  reverse=True)
        topWomenSkipperSum = sum([p.wsr.ordinal(target=targetElo, alpha=200 / model.sigma)
                                  for p in topWomenSkippers[:numTops]])
        topWomenSkippers = [{'name': p.name, 'key': p.key,
                             'mu': p.wsr.mu, 'sigma': p.wsr.sigma} for p in topWomenSkippers]

        topWomenCrews = sorted([p for p in filtered_people if validPerson(p, 'crew')],
                               key=lambda x: x.wcr.ordinal(
                                   target=targetElo, alpha=200 / model.sigma),
                               reverse=True)
        topWomenCrewsSum = sum([p.wcr.ordinal(target=targetElo, alpha=200 / model.sigma)
                                for p in topWomenCrews[:numTops]])
        topWomenCrews = [{'name': p.name, 'key': p.key,
                          'mu': p.wcr.mu, 'sigma': p.wcr.sigma} for p in topWomenCrews]

        topWomenRating = (topWomenSkipperSum +
                          topWomenCrewsSum) / (numTops * 2)

        # Team racing:

        numTops = 3

        topSkippersTR = sorted([p for p in filtered_people
                                if p.tsr.mu != model.mu
                                if not p.hasTargetSeasons(targetSeasons, 'skipper')],
                               key=lambda x: x.tsr.ordinal(
                                   target=targetElo, alpha=200 / model.sigma),
                               reverse=True)
        topSkipperTRSum = sum([p.tsr.ordinal(target=targetElo, alpha=200 / model.sigma)
                               for p in topSkippersTR[:numTops]])
        topSkippersTR = [{'name': p.name, 'key': p.key,
                          'mu': p.tsr.mu, 'sigma': p.tsr.sigma} for p in topSkippersTR]

        topCrewsTR = sorted([p for p in filtered_people
                             if p.tcr.mu != model.mu
                             if not p.hasTargetSeasons(targetSeasons, 'crew')],
                            key=lambda x: x.tcr.ordinal(
                                target=targetElo, alpha=200 / model.sigma),
                            reverse=True)
        topCrewTRSum = sum([p.tcr.ordinal(target=targetElo, alpha=200 / model.sigma)
                            for p in topCrewsTR[:numTops]])
        topCrewsTR = [{'name': p.name, 'key': p.key,
                       'mu': p.tcr.mu, 'sigma': p.tcr.sigma} for p in topCrewsTR]
        topRatingTR = (topSkipperTRSum + topCrewTRSum) / (numTops * 2)

        # WOMEN's Team Racing:

        topWomenSkippersTR = sorted([p for p in filtered_people
                                     if p.wtsr.mu != model.mu
                                     and p.gender == 'F'
                                     and not p.hasTargetSeasons(targetSeasons, 'skipper')],
                                    key=lambda x: x.wtsr.ordinal(
                                        target=targetElo, alpha=200 / model.sigma),
                                    reverse=True)
        topWomenSkipperTRSum = sum([p.wtsr.ordinal(target=targetElo, alpha=200 / model.sigma)
                                    for p in topWomenSkippersTR[:numTops]])
        topWomenSkippersTR = [{'name': p.name, 'key': p.key, 'mu': p.wtsr.mu,
                               'sigma': p.wtsr.sigma} for p in topWomenSkippersTR]

        topWomenCrewsTR = sorted([p for p in filtered_people
                                  if p.wtcr.mu != model.mu
                                  and p.gender == 'F'
                                  and not p.hasTargetSeasons(targetSeasons, 'crew')],
                                 key=lambda x: x.wtcr.ordinal(
                                     target=targetElo, alpha=200 / model.sigma),
                                 reverse=True)
        topWomenCrewTRSum = sum([p.wtcr.ordinal(target=targetElo, alpha=200 / model.sigma)
                                 for p in topWomenCrewsTR[:numTops]])
        topWomenCrewsTR = [{'name': p.name, 'key': p.key, 'mu': p.wtcr.mu,
                            'sigma': p.wtcr.sigma} for p in topWomenCrewsTR]
        topWomenRatingTR = (topWomenSkipperTRSum +
                            topWomenCrewTRSum) / (numTops * 2)

        return {'team': team,
                'topOpen': topRating,
                'topWomen': topWomenRating,
                'topOpenTR': topRatingTR,
                'topWomenTR': topWomenRatingTR,
                'skippers': topSkippers,
                'crews': topCrews,
                'wskippers': topWomenSkippers,
                'wcrews': topWomenCrews,
                'trskippers': topSkippersTR,
                'trcrews': topCrewsTR,
                'wtrskippers': topWomenSkippersTR,
                'wtrcrews': topWomenCrewsTR}


def getTeamRanks(people, season, lastRanks=None):
    trs = {}
    for team in teamRegions.keys():
        tr = getTeamRatings(people, team, [season])
        if tr is not None:
            trs[team] = tr
        # else:
        #     trs[team] = lastRanks[team]
    ranks = {}

    for type, type2 in zip(['topOpen', 'topWomen', 'topOpenTR', 'topWomenTR'], ['o', 'w', 'tr', 'wtr']):
        sorted_trs = sorted(
            trs.items(), key=lambda item: item[1][type], reverse=True)

        for i, (team, vals) in enumerate(sorted_trs):
            if team not in ranks.keys():
                ranks[team] = {}
            ranks[team][type2] = i + 1
    return ranks


def calculateFR(people, date, regatta, race, row, type, scoring, season, residuals, regattaAvg, womens):
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

    # Check for womens regatta
    partnerKeys = scores['PartnerLink']
    partnerNames = scores['Partner']
    # partnerKeys = row[row['Position'] != type]['key']
    # partnerNames = row[row['Position'] != type]['Sailor']
    # partners = [people[key] if key != 'Unknown'
    # and key is not None
    # else None for key in zip(partnerKeys, partnerNames)]

    # Seperate out the openskill rating objects for use in the model
    if not womens:
        ratings = [[r.sr] if type == 'Skipper' else [r.cr] for r in racers]
    else:
        ratings = [[r.wsr] if type == 'Skipper' else [r.wcr] for r in racers]

    # grab starting rating values for change calculation later
    startingRating = [r[0].ordinal(
        target=targetElo, alpha=200 / model.sigma) for r in ratings]

    # Rate using the model
    ratings = model.rate(ratings, scoreVals)

    # Make predictions
    predictions = model.predict_rank(ratings)

    # calculate error and add to list (residuals)
    for pred, score, racer in zip(predictions, scoreVals, racers):
        residuals.append(score - pred[0])

    # Update racers' ratings
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

    # Pre-calculate lists for sailor's race values
    if womens:
        changes = [(racers[i].wsr.ordinal(target=targetElo, alpha=200 / model.sigma) if type == 'Skipper' else racers[i].wcr.ordinal(
            target=targetElo, alpha=200 / model.sigma)) - startingRating[i] for i in range(len(racers))]
    else:
        changes = [(racers[i].sr.ordinal(target=targetElo, alpha=200 / model.sigma) if type == 'Skipper' else racers[i].cr.ordinal(
            target=targetElo, alpha=200 / model.sigma)) - startingRating[i] for i in range(len(racers))]

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
    doCr = race.split("/")[0] in targetSeasons and isCross == 1

    # Loop through each sailor and the associated values
    for sailor, score, pred, change, partnerKey, partnerName, oldRating, new_rating, team in zip(racers, scoreVals, predictions, changes, partnerKeys, partnerNames, startingRating, ratings, teams):
        outLinks = 0

        partnerKey = partnerKey if partnerKey not in merges.keys(
        ) else merges[partnerKey]

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
            'change': change,
            'regAvg': regattaAvg,
            'cross': isCross,
            'outLinks': outLinks,
            'sr': sailor.sr.ordinal(target=targetElo, alpha=200 / model.sigma),
            'srmu': sailor.sr.mu,
            'srsig': sailor.sr.sigma,
            'ratingType': ratingType,
            'oldRating': oldRating,
            'newRating': new_rating[0].ordinal(target=targetElo, alpha=200 / model.sigma),
            'womens': womens,
            'date': date,
            'partner': {'name': partnerName, 'key': partnerKey},
            'venue': venue,
            'raceID': actualID,
            'type': 'fleet',
            'scoring': scoring
        })


def calculateTR(people, date, regatta, race, row, type, scoring, season, regattaAvg, womens):
    venue = row['Venue'].iat[0]

    teamAName = row['teamAName'].iat[0]
    teamAKeys = [boat[type.lower() + 'Key'] if boat[type.lower() + 'Key']
                 is not None else 'Unknown' for boat in row['teamABoats'].iat[0]]
    # teamANames = [boat[type.lower() + 'Name'] for boat in row['teamABoats'].iat[0]]
    for oldkey, newkey in merges.items():
        if oldkey in teamAKeys:
            teamAKeys = [k if k != oldkey else newkey for k in teamAKeys]
    teamARacers = [people[key]
                   for key in teamAKeys if 'Unknown' not in key and key in people.keys()]

    teamBName = row['teamBName'].iat[0]
    teamBKeys = [boat[type.lower() + 'Key'] if boat[type.lower() + 'Key']
                 is not None else 'Unknown' for boat in row['teamBBoats'].iat[0]]
    # teamBNames = [boat[type.lower() + 'Name'] for boat in row['teamBBoats'].iat[0]]
    for oldkey, newkey in merges.items():
        if oldkey in teamBKeys:
            teamBKeys = [k if k != oldkey else newkey for k in teamBKeys]
    teamBRacers = [people[key]
                   for key in teamBKeys if 'Unknown' not in key and key in people.keys()]

    teamARatings = []
    if womens:
        teamARatings = [r.wtsr if type ==
                        'Skipper' else r.wtcr for r in teamARacers]
    else:
        teamARatings = [r.tsr if type ==
                        'Skipper' else r.tcr for r in teamARacers]

    teamBRatings = []
    if womens:
        teamBRatings = [r.wtsr if type ==
                        'Skipper' else r.wtcr for r in teamBRacers]
    else:
        teamBRatings = [r.tsr if type ==
                        'Skipper' else r.tcr for r in teamBRacers]

    startingARating = [
        r.ordinal(target=targetElo, alpha=200 / model.sigma) for r in teamARatings]
    startingBRating = [
        r.ordinal(target=targetElo, alpha=200 / model.sigma) for r in teamBRatings]

    if len(teamARatings) < 1 or len(teamBRatings) < 1:
        # print("not enough sailors in this race, skipping", row['raceID'].iat[0])
        return

    predictions = model.predict_rank([teamARatings, teamBRatings])

    ratings = model.rate([teamARatings, teamBRatings],
                         ranks=[1 if row['teamAOutcome'].iat[0] == 'win' else 2, 1 if row['teamBOutcome'].iat[0] == 'win' else 2])

    for team, name, newRatings in zip([teamARacers, teamBRacers], [teamAName, teamBName], ratings):
        for racer, new_rating in zip(team, newRatings):
            # print(new_rating.ordinal(target=targetElo, alpha=200 / model.sigma))
            # racer.teams = [name]
            if type == 'Skipper':
                if womens:
                    racer.wtsr = new_rating
                else:
                    racer.tsr = new_rating
            else:
                if womens:
                    racer.wtcr = new_rating
                else:
                    racer.tcr = new_rating

    if womens:
        teamARatings = [r.wtsr if type ==
                        'Skipper' else r.wtcr for r in teamARacers]
    else:
        teamARatings = [r.tsr if type ==
                        'Skipper' else r.tcr for r in teamARacers]

    if womens:
        teamBRatings = [r.wtsr if type ==
                        'Skipper' else r.wtcr for r in teamBRacers]
    else:
        teamBRatings = [r.tsr if type ==
                        'Skipper' else r.tcr for r in teamBRacers]

    endingARatings = [
        r.ordinal(target=targetElo, alpha=200 / model.sigma) for r in teamARatings]
    endingBRatings = [
        r.ordinal(target=targetElo, alpha=200 / model.sigma) for r in teamBRatings]

    AChanges = [e-s for s, e in zip(startingARating, endingARatings)]
    BChanges = [e-s for s, e in zip(startingBRating, endingBRatings)]

    for tscore, toutcome, oppt, oppn, index, racers, oppRacers, changes, boats, starting, new_ratings, teamName in zip(
        [row['teamAScore'].iat[0], row['teamBScore'].iat[0]],
        [row['teamAOutcome'].iat[0],
         row['teamBOutcome'].iat[0]],
        [row['teamBName'].iat[0],
         row['teamAName'].iat[0]],
        [row['teamBNick'].iat[0],
         row['teamANick'].iat[0]],
        [0, 1],
        [teamARacers, teamBRacers],
        [teamBRacers, teamARacers],
        [AChanges, BChanges],
        [row['teamABoats'].iat[0], row['teamBBoats'].iat[0]], 
        [startingARating, startingBRating], 
        [endingARatings, endingBRatings], [teamAName, teamBName]):

        partnerKeys = [boat['crewKey'] if boat['crewKey']
                       is not None else 'Unknown' for boat in boats]
        partnerNames = [boat['crewName'] if boat['crewName']
                        is not None else 'Unknown' for boat in boats]
        if type == 'Crew':
            partnerKeys = [boat['skipperKey'] if boat['skipperKey']
                           is not None else 'Unknown' for boat in boats]
            partnerNames = [boat['skipperName'] if boat['skipperName']
                            is not None else 'Unknown' for boat in boats]

        for racer, change, partnerKey, partnerName, oldRating, new_rating in zip(racers, changes, partnerKeys, partnerNames, starting, new_ratings):

            partnerKey = partnerKey if partnerKey not in merges.keys(
            ) else merges[partnerKey]

            # Calculate Rivals
            for opp in oppRacers:
                if type not in racer.rivals:
                    racer.rivals[type] = {}

                if opp.key not in racer.rivals[type]:
                    racer.rivals[type][opp.key] = {
                        'name': opp.name, 'races': {}, 'team': opp.teams[-1], 'wins': {}}

                if season not in racer.rivals[type][opp.key]['races'].keys():
                    racer.rivals[type][opp.key]['races'][season] = 0
                if season not in racer.rivals[type][opp.key]['wins'].keys():
                    racer.rivals[type][opp.key]['wins'][season] = 0

                racer.rivals[type][opp.key]['races'][season] += 1
                if toutcome == 'win':
                    racer.rivals[type][opp.key]['wins'][season] += 1

            # Make sure seasons are updated
            if season not in [s[0] for s  in racer.seasons[type.lower()]]:
                racer.seasons[type.lower()].append((season, teamName))

            ratingType = 'wtsr' if womens else 'tsr' if type == 'Skipper' else 'wtcr' if womens else 'tcr'
            racer.races.append({'raceID': row['raceID'].iat[0], 'raceNum': int(row['raceNum'].iat[0]), 'round':  row['round'].iat[0],
                                'pos': type,
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
                                'change': float(change),
                                'venue': venue,
                                'type': 'team'
                                })


def main(df_sailor_ratings, df_sailor_info, df_races_full):
    people = {}
    people, df_races_full = setupPeople(df_sailor_ratings, df_sailor_info, df_races_full)

    # Pre calculate the number of races to rate
    leng = len(df_races_full['adjusted_raceID'].unique())

    # List of residuals (errors)
    residuals = []

    dateRanks = {}

    # Current race count for print statement
    i = 0

    # First, group by regatta to calculate regatta-level metrics
    regatta_groups = df_races_full.groupby(['regatta'], sort=False)

    # Initialize counters for the outer loop
    regatta_count = len(regatta_groups)

    prevDate = None

    # Iterate through each regatta first
    for regatta_name, regatta_data in regatta_groups:

        # Calculate for each position scoring
        scoring = ""
        if isinstance(regatta_data.iloc[0]['Scoring'], str):
            scoring = regatta_data.iloc[0]['Scoring']
        else:
            scoring = regatta_data.iloc[0]['Scoring'].iat[0]

        season = regatta_data.iloc[0]['raceID'].split("/")[0]

        # Should calculate women's here
        womens = False

        regattAvg = 0

        # Filter down people once per regatta?
        if scoring != 'team':
            skipper_keys = regatta_data.loc[regatta_data['Position']
                                            == 'Skipper']['key'].unique()
            skippers = [people[k] for k in skipper_keys]

            crew_keys = regatta_data.loc[regatta_data['Position'] == 'Crew']['key'].unique(
            )
            crews = [people[k] for k in crew_keys if k in people.keys()]

            genders = [p.gender for p in skippers + crews]
            womenCount = sum([1 if g == "F" else 0 for g in genders])
            womens = 'M' not in genders and womenCount >= 4
            tempRating = 0
            for type, racers in zip(['Skiper', 'Crew'], [skippers, crews]):
                if womens:
                    ratings = [r.wsr if type ==
                               'Skipper' else r.wcr for r in racers]
                else:
                    ratings = [r.sr if type ==
                               'Skipper' else r.cr for r in racers]

                startingRating = [
                    r.ordinal(target=targetElo, alpha=200 / model.sigma) for r in ratings]
                tempRating += sum(startingRating)

            # Calculate regatta average
            regattaAvg = tempRating / len(skippers + crews)
        else:  # TR
            skipper_keys = [k for kl in regatta_data['allSkipperKeys']
                            for k in kl]
            crew_keys = [k for kl in regatta_data['allCrewKeys'] for k in kl]

            for oldkey, newkey in merges.items():
                if oldkey in skipper_keys:
                    skipper_keys = [
                        k if k != oldkey else newkey for k in skipper_keys]
                if oldkey in crew_keys:
                    crew_keys = [
                        k if k != oldkey else newkey for k in crew_keys]

            skippers = [people[k] for k in skipper_keys if k in people.keys()]
            crews = [people[k] for k in crew_keys if k in people.keys()]

            genders = [p.gender for p in skippers + crews]
            womenCount = sum([1 if g == "F" else 0 for g in genders])
            womens = 'M' not in genders and womenCount >= 4

            tempRating = 0
            for type, racers in zip(['Skiper', 'Crew'], [skippers, crews]):
                if womens:
                    ratings = [r.wtsr if type == 'Skipper' else r.wtcr for r in racers]
                else:
                    ratings = [r.tsr if type == 'Skipper' else r.tcr for r in racers]

                startingRating = [r.ordinal(target=targetElo, alpha=200 / model.sigma) for r in ratings]
                tempRating += sum(startingRating)

            # Calculate regatta average
            regattaAvg = tempRating / len(skippers + crews)

        race_groups = regatta_data.groupby(
            ['date', 'regatta', 'adjusted_raceID'], sort=False)
        race_count = len(race_groups)

        # Iterate through each race in this regatta
        for (date, regatta, race), row in race_groups:
            i += 1

            # Print status every 100 races
            if i % 1000 == 0:
                print(
                    f"Currently analyzing race {i}/{leng} in {regatta}, Date:{date}")

            for pos in ['Skipper', 'Crew']:
                if scoring == 'team':
                    calculateTR(people, date, regatta, race, row, pos, scoring, season, regattaAvg, womens)
                else:
                    calculateFR(people, date, regatta, race, row, pos, scoring, season, residuals, regattaAvg, womens)

        # Update the team ranking history every week
        # date = regatta_data['date'].iloc[0]
        # if date.year >= 2018 and date.strftime("%Y-%m-%d") != prevDate and date.weekday() == 5:
        #     prevDate = date.strftime("%Y-%m-%d")
        #     if prevDate is not None:
        #         ranks = getTeamRanks(people, season) #  dateRanks[-1]
        #         if (len(ranks)) > 0:
        #             dateRanks[prevDate] = ranks
    # Calculate statiscs about the accuracy of the model. (Lower is better)
    # me = np.array(residuals).mean()
    # mse = (np.array(residuals) ** 2).mean()
    # print(me, mse)
    return people, df_races_full


def calculateDateRanks(dateRanks):
    # list(dateRanks.keys())[0].strftime("%Y-%m-%d")
    final = {'dates': list(dateRanks.keys())}
    for team in list(dateRanks.values())[-1].keys():
        final[team] = [d[team] if team in d.keys(
        ) else None for d in dateRanks.values()]
    doc = db.collection('vars').document('dateRanks').set({"ranks": final})
    # final


def postCalcAdjust(people, df_races_full):
    # Filter sailors who have 'f24' in their seasons list
    eligible_skippers = [p for p in people.values()
                         if not p.hasTargetSeasons(targetSeasons, 'skipper')
                         and sum([race['outLinks'] for race in p.races if 'outLinks' in race.keys()]) > 70 
                         and (2000 + int(p.year.split()[0]) > gradCutoff if isinstance(p.year, str) and len(p.year.split()) > 1 else int(p.year) > gradCutoff)]

    eligible_crews = [p for p in people.values()
                      if not p.hasTargetSeasons(targetSeasons, 'crew')
                      and sum([race['outLinks'] for race in p.races if 'outLinks' in race.keys()]) > 70
                      and (2000 + int(p.year.split()[0]) > gradCutoff if isinstance(p.year, str) and len(p.year.split()) > 1 else int(p.year) > gradCutoff)]

    # TODO: Count tr and fr seasons seperately
    eligible_skippers_tr = [p for p in people.values()
                            if not p.hasTargetSeasons(targetSeasons, 'skipper')
                            and (2000 + int(p.year.split()[0]) > gradCutoff if isinstance(p.year, str) and len(p.year.split()) > 1 else int(p.year) > gradCutoff)]
    eligible_crews_tr = [p for p in people.values()
                         if not p.hasTargetSeasons(targetSeasons, 'crew')
                         and (2000 + int(p.year.split()[0]) > gradCutoff if isinstance(p.year, str) and len(p.year.split()) > 1 else int(p.year) > gradCutoff)]

    for p in people.values():
        p.skipperRank = 0
        p.crewRank = 0
        p.womenSkipperRank = 0
        p.womenCrewRank = 0
        p.skipperRankTR = 0
        p.crewRankTR = 0
        p.womenSkipperRankTR = 0
        p.womenCrewRankTR = 0

    for i, s in enumerate(sorted([p for p in eligible_skippers if p.sr.mu != model.mu], key=lambda p: p.sr.ordinal(), reverse=True)):
        s.skipperRank = i + 1
    for i, s in enumerate(sorted([p for p in eligible_crews if p.cr.mu != model.mu], key=lambda p: p.cr.ordinal(), reverse=True)):
        s.crewRank = i + 1

    for i, s in enumerate(sorted([p for p in eligible_skippers if p.wsr.mu != model.mu], key=lambda p: p.wsr.ordinal(), reverse=True)):
        s.womenSkipperRank = i + 1
    for i, s in enumerate(sorted([p for p in eligible_crews if p.wcr.mu != model.mu], key=lambda p: p.wcr.ordinal(), reverse=True)):
        s.womenCrewRank = i + 1

    for i, s in enumerate(sorted([p for p in eligible_skippers_tr if p.tsr.mu != model.mu], key=lambda p: p.tsr.ordinal(), reverse=True)):
        s.skipperRankTR = i + 1
    for i, s in enumerate(sorted([p for p in eligible_crews_tr if p.tcr.mu != model.mu], key=lambda p: p.tcr.ordinal(), reverse=True)):
        s.crewRankTR = i + 1

    for i, s in enumerate(sorted([p for p in eligible_skippers_tr if p.wtsr.mu != model.mu], key=lambda p: p.wtsr.ordinal(), reverse=True)):
        s.womenSkipperRankTR = i + 1
    for i, s in enumerate(sorted([p for p in eligible_crews_tr if p.wtcr.mu != model.mu], key=lambda p: p.wtcr.ordinal(), reverse=True)):
        s.womenCrewRankTR = i + 1

    allRows = []
    for sailor, p in people.items():
        avgSkipperRatio = float(np.array(
            [r['ratio'] for r in p.races if r['pos'] == 'Skipper' and 'ratio' in r.keys()]).mean())
        avgCrewRatio = float(np.array(
            [r['ratio'] for r in p.races if r['pos'] == 'Crew' and 'ratio' in r.keys()]).mean())
        p.avgSkipperRatio = avgSkipperRatio
        p.avgCrewRatio = avgCrewRatio

        allRows.append([p.name, len(p.races), sailor,
                        p.skipperRank, p.crewRank, p.womenSkipperRank,
                        p.womenCrewRank, p.skipperRankTR, p.womenSkipperRankTR,
                        p.teams,
                        p.gender,
                        p.sr.ordinal(target=targetElo,
                                     alpha=200 / model.sigma),
                        p.cr.ordinal(target=targetElo,
                                     alpha=200 / model.sigma),
                        p.wsr.ordinal(target=targetElo,
                                      alpha=200 / model.sigma),
                        p.wcr.ordinal(target=targetElo,
                                      alpha=200 / model.sigma),
                        p.tsr.ordinal(target=targetElo,
                                      alpha=200 / model.sigma),
                        p.tcr.ordinal(target=targetElo,
                                      alpha=200 / model.sigma),
                        p.wtsr.ordinal(target=targetElo,
                                       alpha=200 / model.sigma),
                        p.wtcr.ordinal(target=targetElo,
                                       alpha=200 / model.sigma),
                        p.sr.mu, p.sr.sigma,
                        p.cr.mu, p.cr.sigma,
                        p.wsr.mu, p.wsr.sigma,
                        p.wcr.mu, p.wcr.sigma,
                        p.tsr.mu, p.tsr.sigma,
                        p.tcr.mu, p.tcr.sigma,
                        p.wtsr.mu, p.wtsr.sigma,
                        p.wtcr.mu, p.wtcr.sigma,
                        sum([race['outLinks']
                            for race in p.races if 'outLinks' in race.keys()]),
                        p.year, p.links,
                        p.seasons,
                        sum([race['cross']
                            for race in p.races if 'cross' in race.keys()]),
                        p.races, p.rivals, avgSkipperRatio, avgCrewRatio])

    df_sailors = pd.DataFrame(allRows, columns=['Sailor', 'numRaces', 'key', 'SkipperRank', 'CrewRank', 'WomenSkipperRank', 'WomenCrewRank', 'TRSkipperRank', 'TRWomenSkipperRank', 'Teams', 'gender',
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
                                                'Seasons', 'Cross', 'Races',  'Rivals', 'skipperAvgRatio', 'crewAvgRatio'])

    df_sailors.to_json(f'sailors-{date.today().strftime("%Y%m%d")}.json', index=False)
    df_sailors.to_json(f'sailors-latest.json', index=False)
    df_sailors = df_sailors.sort_values(
        by='numRaces', ascending=False).reset_index(drop=True)
    # print(len(df_races_full))
    
    for oldkey, newkey in merges.items():
        # replace merged names in dataset
        df_races_full['Link'] = df_races_full['Link'].replace(oldkey, newkey)
        
    print(len(people))
    return people, df_sailors, df_races_full


def getCounts(races):
    # season_counts = defaultdict(int)
    season_counts = {}

    for race in races:
        season = race["raceID"].split("/")[0]
        if season not in season_counts.keys():
            season_counts[season] = {}
        if race['pos'] not in season_counts[season].keys():
            season_counts[season][race['pos']] = 0
        season_counts[season][race['pos']] += 1

    return dict(season_counts)


def uploadSailors(people, cursor, connection, batch_size=300):
    
    # eligible = [p for p in people.values() if (targetSeasons[-1] in p.seasons['skipper']
    #                                            or targetSeasons[-1] in p.seasons['crew'])
    #             and len(p.races) > 0
    #             and type(p.races[-1]['date']) != type("hi")
    #             and (today - p.races[-1]['date']).days < 14]
    eligible = list(people.values())
    print(len(eligible))

    sailor_rows = []
    rival_rows = []
    sailor_teams_rows = []

    for i, p in enumerate(eligible):
        if p.key is None:
            print("No key for", p.name)
            continue

        avg_sk = 0 if p.avgSkipperRatio is None or np.isnan(p.avgSkipperRatio) else p.avgSkipperRatio
        avg_cr = 0 if p.avgCrewRatio is None or np.isnan(p.avgCrewRatio) else p.avgCrewRatio

        sailor_rows.append((
            p.key.replace("/", "-"),
            p.name,
            p.gender,
            int(p.sr.ordinal(target=targetElo, alpha=200 / model.sigma)),
            int(p.cr.ordinal(target=targetElo, alpha=200 / model.sigma)),
            int(p.wsr.ordinal(target=targetElo, alpha=200 / model.sigma)),
            int(p.wcr.ordinal(target=targetElo, alpha=200 / model.sigma)),
            int(p.tsr.ordinal(target=targetElo, alpha=200 / model.sigma)),
            int(p.tcr.ordinal(target=targetElo, alpha=200 / model.sigma)),
            int(p.wtsr.ordinal(target=targetElo, alpha=200 / model.sigma)),
            int(p.wtcr.ordinal(target=targetElo, alpha=200 / model.sigma)),
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
            p.year
        ))
        
        raceCounts = (lambda rc_norm, ps: {
                        'skipper': {season: rc_norm.get(season, {}).get('Skipper', 0) for season in [s[0] for s in list(ps['skipper'])]},
                        'crew': {season: rc_norm.get(season, {}).get('Crew', 0) for season in [s[0] for s in list(ps['crew'])]}
                    })( (lambda rc: {s: {pos.title(): cnt for pos, cnt in posd.items()} for s, posd in rc.items()})(getCounts(p.races)), p.seasons ) 

        for position in ['Skipper', 'Crew']:
            if position not in p.rivals:
                continue
            for key, values in p.rivals[position].items():
                for season in values['races']:
                    rival_rows.append((
                        p.key.replace("/", "-"),
                        key,
                        values['name'],
                        values['team'],
                        position,
                        season,
                        values['races'][season],
                        values['wins'][season]
                    ))
                    
        for position in ['skipper', 'crew']:
            if p.key is None:
                continue
            for season, team in set(p.seasons[position]):
                sailor_teams_rows.append((
                    p.key.replace("/", "-"),
                    team,
                    season,
                    position,
                    raceCounts[position][season]
                ))

        # Commit in batches
        if (i + 1) % batch_size == 0:
            print(f"Uploading sailors {i - batch_size + 1} to {i}...", len(sailor_teams_rows))
            cursor.executemany("""
                INSERT IGNORE INTO Sailors (
                    sailorID, name, gender, sr, cr, wsr, wcr, tsr, tcr, wtsr, wtcr,
                    sRank, cRank, wsRank, wcRank, tsRank, tcRank, wtsRank, wtcRank,
                    avgSkipperRatio, avgCrewRatio, year
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, sailor_rows)
            sailor_rows.clear()

            if rival_rows:
                cursor.executemany("""
                    INSERT IGNORE INTO SailorRivals (
                        sailorID, rivalID, rivalName, rivalTeam, position, season, raceCount, winCount
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, rival_rows)
                rival_rows.clear()
            
            if sailor_teams_rows:
                try:
                    cursor.executemany("""
                        INSERT IGNORE INTO SailorTeams(sailorID, teamID, season, position, raceCount)
                        VALUES(%s,%s,%s,%s,%s)
                    """, sailor_teams_rows)
                    sailor_teams_rows.clear()
                except Exception as e:
                    print(sailor_teams_rows)
                    raise e

            connection.commit()

    # Final flush
    if sailor_rows:
        cursor.executemany("""
                INSERT IGNORE INTO Sailors (
                    sailorID, name, gender, sr, cr, wsr, wcr, tsr, tcr, wtsr, wtcr,
                    sRank, cRank, wsRank, wcRank, tsRank, tcRank, wtsRank, wtcRank,
                    avgSkipperRatio, avgCrewRatio, year
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, sailor_rows)
        connection.commit()
    if rival_rows:
        cursor.executemany("""
                    INSERT INTO SailorRivals (
                        sailorID, rivalID, rivalName, rivalTeam, position, season, raceCount, winCount
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, rival_rows)
    if sailor_teams_rows:
        cursor.executemany("""
                INSERT IGNORE INTO SailorTeams(sailorID, teamID, season, position, raceCount)
                VALUES(%s,%s,%s,%s,%s)
            """, sailor_teams_rows)

        connection.commit()

    print("✅ All sailors uploaded successfully!")

def uploadTeams(df_sailors, df_races_full, people, cursor, connection):

    def getCounts(races):
        # season_counts = defaultdict(int)
        season_counts = {}

        for race in races:
            season = race["raceID"].split("/")[0]
            if season not in season_counts.keys():
                season_counts[season] = {}
            if race['pos'] not in season_counts[season].keys():
                season_counts[season][race['pos']] = 0
            season_counts[season][race['pos']] += 1

        return dict(season_counts)

    # batch = db.batch()
    # col = db.collection('eloTeams')
    teams = []
    predteams = []
    scrape = False
    teamNames = teamRegions.keys()
    lenteams = len(teamNames)

    season_mask = df_sailors['Seasons'].apply(lambda x: not set(x['skipper']).isdisjoint(
        targetSeasons) or not set(x['crew']).isdisjoint(targetSeasons))

    # Explode the Teams column to enable grouping
    df_exploded = df_sailors.loc[season_mask].explode('Teams')

    # Group by team and compute necessary aggregates
    team_stats = df_exploded.groupby('Teams').agg(
        numCurMembers=('Teams', 'count'),
        avgSkipperOrdinal=('srOrd', 'mean'),
        avgCrewOrdinal=('crOrd', 'mean'),
        avgSkipperRatio=('skipperAvgRatio', 'mean'),
        avgCrewRatio=('crewAvgRatio', 'mean')
    )

    # Calculate the average values as in the original code
    team_stats['avg'] = (team_stats['avgSkipperOrdinal'] +
                         team_stats['avgCrewOrdinal']) / 2
    team_stats['avgRatio'] = (
        team_stats['avgSkipperRatio'] + team_stats['avgCrewRatio']) / 2

    team_stats = team_stats.reindex(teamNames, fill_value=0)
    today = datetime.today()

    # team_link_map = df_races_full.dropna().drop_duplicates('Team').set_index('Team')['Teamlink'].to_dict()

    df_cleaned = df_races_full.dropna(
        subset=['Team', 'Teamlink']).drop_duplicates(subset='Team', keep='first')

    # Create a dictionary with 'Team' as the key and 'TeamLink' as the value
    team_link_map = pd.Series(
        df_cleaned.Teamlink.values, index=df_cleaned.Team).to_dict()


    for i, (team, row) in enumerate(team_stats.iterrows()):
        # if team != "MIT":
        #     continue
        print(f"{i}/{len(team_stats)} {team}")
        avg = row['avg']
        avgRatio = row['avgRatio']
        numCurMembers = row['numCurMembers']

        region = teamRegions[team]
        # teamLink = df_races.loc[df_races['Team'] == team, 'Teamlink'].iloc[0]
        # Default to '' if team not found
        teamLink = team_link_map.get(team, '')
        url = f"https://scores.collegesailing.org/schools/{teamLink}"

        filtered_people = [p for p in people.values() if team in p.teams]

        members = [{"name": str(p.name),
                    "key": p.key,
                    "gender": p.gender,
                    "year": str(p.year),
                    'teams': list(p.teams),
                    'skipperRating': int(p.sr.ordinal(target=targetElo, alpha=200 / model.sigma)),
                    # 'sr': {'mu': p.sr.mu, 'sigma': p.sr.sigma},
                    'crewRating': int(p.cr.ordinal(target=targetElo, alpha=200 / model.sigma)),
                    'womenSkipperRating': int(p.wsr.ordinal(target=targetElo, alpha=200 / model.sigma)),
                    'womenCrewRating': int(p.wcr.ordinal(target=targetElo, alpha=200 / model.sigma)),
                    'tsr': p.tsr.ordinal(target=targetElo, alpha=200 / model.sigma),
                    'wtsr': p.wtsr.ordinal(target=targetElo, alpha=200 / model.sigma),
                    'tcr': p.tcr.ordinal(target=targetElo, alpha=200 / model.sigma),
                    'wtcr': p.wtcr.ordinal(target=targetElo, alpha=200 / model.sigma),
                    'avgSkipperRatio': float(p.avgSkipperRatio),
                    'avgCrewRatio': float(p.avgCrewRatio),
                    # season -> position -> count (normalized position names)
                    'raceCount': (lambda rc: {s: {pos.title(): cnt for pos, cnt in posd.items()} for s, posd in rc.items()})(getCounts(p.races)),
                    # position -> season -> count (used later when inserting SailorTeams)
                    'raceCounts': (lambda rc_norm, ps: {
                        'skipper': {season: rc_norm.get(season, {}).get('Skipper', 0) for season in list(ps['skipper'])},
                        'crew': {season: rc_norm.get(season, {}).get('Crew', 0) for season in list(ps['crew'])}
                    })( (lambda rc: {s: {pos.title(): cnt for pos, cnt in posd.items()} for s, posd in rc.items()})(getCounts(p.races)), p.seasons ),
                    'seasons': p.seasons,
                    'cross': sum([race.get('cross', 0) for race in p.races]),
                    'outLinks': sum([race['outLinks'] for race in p.races if 'outLinks' in race.keys()]),
                    'skipperRank': int(p.skipperRank),
                    'crewRank': int(p.crewRank),
                    'womenSkipperRank': int(p.womenSkipperRank),
                    'womenCrewRank': int(p.womenCrewRank)
                    } for p in filtered_people]
        

        topRating = 0
        topWomenRating = 0
        topRatingTR = 0
        topWomenRatingTR = 0
        
        topRating = 0
        topSkipperSum = 0
        topCrewsSum = 0
        topWomenRating = 0
        topRatingTR = 0
        topWomenRatingTR = 0

        numTops = 3
        if numCurMembers > 0:
            topSkippers = sorted([p for p in filtered_people
                                  if validPerson(p, 'skipper')
                                  #   if sum([race['cross'] for race in p.races if 'cross' in race.keys()]) > 20
                                  #   and sum([race['outLinks'] for race in p.races if 'outLinks' in race.keys()]) > 70
                                  #   and not set(p.seasons['skipper']).isdisjoint(targetSeasons)
                                  # and sum([p['raceCount'][seas] for seas in targetSeasons if seas in p['raceCount'].keys()]) > 5
                                  ],
                                 key=lambda x: x.sr.ordinal(
                                     target=targetElo, alpha=200 / model.sigma),
                                 reverse=True)
            topSkipperSum = sum([p.sr.ordinal(target=targetElo, alpha=200 / model.sigma)
                                 for p in topSkippers[:numTops]])
            topSkippers = [{'name': p.name, 'key': p.key,
                            'mu': p.sr.mu, 'sigma': p.sr.sigma} for p in topSkippers]

            topCrews = sorted([p for p in filtered_people if validPerson(p, 'crew')
                               #   if sum([race['cross'] for race in p.races if 'cross' in race.keys()]) > 20
                               #   and sum([race['outLinks'] for race in p.races if 'outLinks' in race.keys()]) > 70
                               #   and not set(p.seasons['crew']).isdisjoint(targetSeasons)
                               ],
                              key=lambda x: x.cr.ordinal(
                                  target=targetElo, alpha=200 / model.sigma),
                              reverse=True)
            topCrewsSum = sum([p.cr.ordinal(target=targetElo, alpha=200 / model.sigma)
                               for p in topCrews[:numTops]])
            topCrews = [{'name': p.name, 'key': p.key,
                         'mu': p.cr.mu, 'sigma': p.cr.sigma} for p in topCrews]

            topRating = (topSkipperSum + topCrewsSum) / (numTops * 2)
            # print(topSkippers, topCrews, topRating)

            # Women's
            numTops = 2
            topWomenSkippers = sorted([p['womenSkipperRating'] for p in members
                                       #   if p['skipperRank'] != 0
                                      if p['cross'] > 20
                                      and p['gender'] == 'F'
                                      and not set(p['raceCount'].keys()).isdisjoint(targetSeasons)
                                      and all(['Skipper' in p['raceCount'][seas].keys() for seas in targetSeasons if seas in p['raceCount'].keys()])
                                      and sum([p['raceCount'][seas]['Skipper'] for seas in targetSeasons if seas in p['raceCount'].keys()]) > 5
                                       ], reverse=True)[:numTops]
            topWomenSkipperSum = sum(topWomenSkippers)

            topWomenCrews = sorted([p['womenCrewRating'] for p in members
                                    #    if p['crewRank'] != 0
                                    if p['cross'] > 20
                                    and p['gender'] == 'F'
                                    and not set(p['raceCount'].keys()).isdisjoint(targetSeasons)
                                    and all(['Crew' in p['raceCount'][seas].keys() for seas in targetSeasons if seas in p['raceCount'].keys()])
                                    and sum([p['raceCount'][seas]['Crew'] for seas in targetSeasons if seas in p['raceCount'].keys()]) > 5
                                    ], reverse=True)[:numTops]
            topWomenCrewsSum = sum(topWomenCrews)

            topWomenRating = (topWomenSkipperSum +
                              topWomenCrewsSum) / (numTops * 2)

            # Team racing:

            numTops = 3

            topSkippersTR = sorted([p for p in filtered_people
                                    if p.tsr.mu != model.mu
                                    if not p.hasTargetSeasons(targetSeasons, 'skipper')],
                                   key=lambda x: x.tsr.ordinal(
                                       target=targetElo, alpha=200 / model.sigma),
                                   reverse=True)
            topSkipperTRSum = sum([p.tsr.ordinal(target=targetElo, alpha=200 / model.sigma)
                                   for p in topSkippersTR[:numTops]])
            topSkippersTR = [{'name': p.name, 'key': p.key,
                              'mu': p.tsr.mu, 'sigma': p.tsr.sigma} for p in topSkippersTR]

            topCrewsTR = sorted([p for p in filtered_people
                                 if p.tcr.mu != model.mu
                                 if not p.hasTargetSeasons(targetSeasons, 'crew')],
                                key=lambda x: x.tcr.ordinal(
                                    target=targetElo, alpha=200 / model.sigma),
                                reverse=True)
            topCrewTRSum = sum([p.tcr.ordinal(target=targetElo, alpha=200 / model.sigma)
                                for p in topCrewsTR[:numTops]])
            topCrewsTR = [{'name': p.name, 'key': p.key,
                           'mu': p.tcr.mu, 'sigma': p.tcr.sigma} for p in topCrewsTR]
            topRatingTR = (topSkipperTRSum + topCrewTRSum) / (numTops * 2)

            # WOMEN's Team Racing:

            topWomenSkippersTR = sorted([p for p in filtered_people
                                         if p.wtsr.mu != model.mu
                                         and p.gender == 'F'
                                         and not p.hasTargetSeasons(targetTRSeasons, 'skipper')
                                         ],
                                        key=lambda x: x.wtsr.ordinal(
                                            target=targetElo, alpha=200 / model.sigma),
                                        reverse=True)
            topWomenSkipperTRSum = sum([p.wtsr.ordinal(target=targetElo, alpha=200 / model.sigma)
                                        for p in topWomenSkippersTR[:numTops]])
            topWomenSkippersTR = [{'name': p.name, 'key': p.key, 'mu': p.wtsr.mu,
                                   'sigma': p.wtsr.sigma} for p in topWomenSkippersTR]

            topWomenCrewsTR = sorted([p for p in filtered_people
                                      if p.wtcr.mu != model.mu
                                      and p.gender == 'F'
                                      and not p.hasTargetSeasons(targetTRSeasons, 'crew')
                                      ],
                                     key=lambda x: x.wtcr.ordinal(
                                         target=targetElo, alpha=200 / model.sigma),
                                     reverse=True)
            topWomenCrewTRSum = sum([p.wtcr.ordinal(target=targetElo, alpha=200 / model.sigma)
                                     for p in topWomenCrewsTR[:numTops]])
            topWomenCrewsTR = [{'name': p.name, 'key': p.key, 'mu': p.wtcr.mu,
                                'sigma': p.wtcr.sigma} for p in topWomenCrewsTR]
            topWomenRatingTR = (topWomenSkipperTRSum +
                                topWomenCrewTRSum) / (numTops * 2)


        cursor.execute("""
                    INSERT INTO Teams (teamID, teamName, topFleetRating, topWomenRating, topTeamRating, topWomenTeamRating, avgRating, avgRatio, region, link)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,   (team, team, topRating, topWomenRating, topRatingTR, topWomenRatingTR, avg, avgRatio, region, url))            
        
        print("inserted team", team)
        
        batch_size = 200  # adjust for your DB/network limits
        rows_to_insert = []

        for member in members:
            if member['key'] is None:
                print("No key for", member['name'])
                continue

            for position in ['skipper', 'crew']:
                for season in set(member['seasons'][position]):
                    rows_to_insert.append((
                        member['key'].replace("/", "-"),
                        team,
                        season,
                        position,
                        member['raceCounts'][position][season]
                    ))

        # Insert in batches
        for start in range(0, len(rows_to_insert), batch_size):
            batch = rows_to_insert[start:start + batch_size]
            try:
                cursor.executemany("""
                    INSERT IGNORE INTO SailorTeams(sailorID, teamID, season, position, raceCount)
                    VALUES(%s,%s,%s,%s,%s)
                """, batch)
                connection.commit()
            except mysql.connector.errors.IntegrityError as e:
                print("Batch insert failed:", e)
                raise e

        connection.commit()
        
    newTeams = sorted(teams, key=lambda x: x['topRating'], reverse=True)
    return newTeams


if __name__ == "__main__":
    start = time.time()

    doScrape = False
    doUpload = False

    if doScrape:
        df_races = runFleetScrape()
    else:
        # if running scrapers seperately
        df_races = pd.read_json("racesfr.json")

    df_races['raceNum'] = df_races['raceID'].apply(lambda id: int(id.split("/")[2][:-1]))  # Numeric part
    df_races['raceDiv'] = df_races['raceID'].apply(lambda id: id.split("/")[2][-1])  # Division part (e.g., 'A', 'B')
    df_races['adjusted_raceID'] = df_races.apply(adjust_race_id, axis=1)  # to make combined division combined
    df_races['Link'] = df_races['Link'].fillna('Unknown')  # fill empty links
    # df_races['key'] = np.where(df_races['Link'] == 'Unknown', df_races['Sailor'], df_races['Link'])
    df_races['key'] = df_races.apply(
        lambda row: row['Sailor'] + "-" +
        row['Team'] if row['Link'] == 'Unknown' else row['Link'],
        axis=1
    )

    df_races['partnerKey'] = df_races.apply(
        lambda row: row['Partner'] + "-" +
        row['Team'] if row['PartnerLink'] == 'Unknown' else row['PartnerLink'],
        axis=1
    )

    if doScrape:
        df_races_tr, trSailors = scrapeTR()
    else:
        # if running scrapers seperately
        df_races_tr = pd.read_json("racesTR.json")

    df_races_tr['adjusted_raceID'] = df_races_tr['raceID']
    df_races_tr['Scoring'] = 'team'
    df_races = df_races.rename(
        {'Date': 'date', 'Regatta': 'regatta'}, axis='columns')
    df_races_full = pd.concat([df_races, df_races_tr])

    # clean up memory
    del df_races, df_races_tr

    df_races_full = df_races_full.sort_values(
        ['date', 'raceNum', 'raceDiv']).reset_index(drop=True)

    df_sailor_ratings = None
    if not calc_all:
        cutoff = (datetime.now() - timedelta(weeks=2))
        df_races_full = df_races_full.loc[df_races_full['date'] > cutoff]
        df_sailor_ratings = pd.read_json("sailors-latest.json")      
        # df_sailor_ratings = pd.read_json("sailors-20250424.json")

    if doScrape:
        df_sailor_info = runSailorData()
    else:
        df_sailor_info = pd.read_json("sailor_data2.json")

    people = main(df_sailor_ratings, df_sailor_info, df_races_full)
    people, df_sailors = postCalcAdjust(people)

    if doUpload:
        uploadSailors(people, cursor)
        # connection.commit()
        teams = uploadTeams(df_sailors, df_races_full, people, cursor, connection)
        connection.commit()

    cursor.close()
    connection.close()

    end = time.time() 
    print(f"{int((end-start) // 60)}:{int((end-start) % 60)}")