from AsyncScraper import runFleetScrape
from TRScraper import scrapeTR
from dataScraper import runSailorData

from calculationsFR import calculateFR
from calculationsTR import calculateTR

from chatRivals import buildRivals, uploadRivals

from uploadScores import uploadScoresBySailor, uploadAllScores
from Teams import uploadTeams

from Sailors import Sailor, setupPeople, handleMerges, outputSailorsToFile, calculateSailorRanks, uploadSailors, updateSailorRatios
from config import Config

import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import mysql.connector
from dotenv import load_dotenv
import os
import json

import cProfile
import pstats

def getScoring(regatta_data):
    scoring = ""
    if isinstance(regatta_data.iloc[0]['Scoring'], str):
        scoring = regatta_data.iloc[0]['Scoring']
    else:
        scoring = regatta_data.iloc[0]['Scoring'].iat[0]
    return scoring

def getWomensAndRegAvgFR(people, regatta_data, config: Config):
    skipper_keys = regatta_data.loc[regatta_data['Position'] == 'Skipper']['key'].unique()
    skippers = [people[k] for k in skipper_keys if k in people.keys()]

    crew_keys = regatta_data.loc[regatta_data['Position'] == 'Crew']['key'].unique()
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
            r.ordinal(target=config.targetElo, alpha=config.alpha) for r in ratings]
        tempRating += sum(startingRating)

    regattaAvg = tempRating / len(skippers + crews)
    
    return womens, regattaAvg

def getWomensAndRegAvgTR(people, regatta_data, config : Config):
    skipper_keys = [k for kl in regatta_data['allSkipperKeys']
                    for k in kl]
    crew_keys = [k for kl in regatta_data['allCrewKeys'] for k in kl]

    for oldkey, newkey in config.merges.items():
        if oldkey in skipper_keys:
            skipper_keys = [k if k != oldkey else newkey for k in skipper_keys]
        if oldkey in crew_keys:
            crew_keys = [k if k != oldkey else newkey for k in crew_keys]

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

        startingRating = [r.ordinal(target=config.targetElo, alpha=config.alpha) for r in ratings]
        tempRating += sum(startingRating)

    regattaAvg = tempRating / len(skippers + crews)
    
    return womens, regattaAvg
    
def resetPeopleToBeforeSeason(people : list[Sailor], season : str, ratingType: str):
    for person in people:
        person.resetRatingToBeforeSeason(season, ratingType)

def calculateAllRegattaInfo(people, df_races, calculatedAtDict, config: Config):
    print(f"Calculating regatta info (womens and regavg)")
    regattaDict = {}
    
    regatta_groups = df_races.groupby(['Regatta'], sort=False)
    for regatta_name, regatta_data in regatta_groups:
        scoring = getScoring(regatta_data)

        if scoring == 'team':
            womens, regattaAvg = getWomensAndRegAvgTR(people, regatta_data, config)
        else:
            womens, regattaAvg = getWomensAndRegAvgFR(people, regatta_data, config)
        
        ratingType = 'w' if womens else '' + 'tr' if scoring == 'team' else 'fr'
        
        regattaDict[regatta_name] = {'scoring': scoring, 'womens': womens, 'regAvg': regattaAvg, 'ratingType': ratingType}    

    return regattaDict

def calcAllRacesForRT(ratingType, people, df_races, allFrRaces, allTrRaces, calculatedAtDict: dict, config: Config):
    print(f"Calculating all {ratingType} races")
    leng = len(df_races['adjusted_raceID'].unique())
    regatta_groups = df_races.groupby(['Regatta'], sort=False)
    
    i = 0
    canSkipCalc = not config.calcAll
    resetDate = None
    
    for regatta_name, regatta_data in regatta_groups:
        season = regatta_data.iloc[0]['raceID'].split("/")[0]
        scoring = regatta_data.iloc[0]['scoring']
        regattaAvg = regatta_data.iloc[0]['regAvg']
        womens = regatta_data.iloc[0]['womens']
        date = regatta_data.iloc[0]['Date']
        
        if canSkipCalc:
            updatedAt = regatta_data.iloc[0]['updatedAt']
            calculatedAt = calculatedAtDict.setdefault(season)
            
            if calculatedAt is None or updatedAt > calculatedAt:
                canSkipCalc = False
                resetDate = date
                print("Found point that needs to be calculated, resetting to before", date, ratingType)
            else:
                continue

        race_groups = regatta_data.groupby(['adjusted_raceID'], sort=False)
        
        calculatedAtDict[season] = time.time()

        # Iterate through each race in this regatta
        for raceID, row in race_groups:
            i += 1
            if i % 1000 == 0:
                print(f"Currently analyzing race {i}/{leng} in {regatta_name[0]}, Date:{date}")

            for pos in ['Skipper', 'Crew']:
                if scoring == 'team':
                    calculateTR(allTrRaces, people, resetDate, date, row, pos, season, regattaAvg, womens, config)
                else:
                    calculateFR(allFrRaces, people, resetDate, date, regatta_name, raceID[0], row, pos, scoring, season, regattaAvg, womens, ratingType, config)
    return people, allFrRaces, allTrRaces
    
def calculateAllRaces(people, df_races, regatta_info, calculatedAtDict: dict, config: Config):
    allFrRaces = []
    allTrRaces = []
    
    df_regatta_info = pd.DataFrame(regatta_info).T
    df_regatta_info = df_regatta_info.reset_index().rename(columns={'level_0': 'Regatta'})

    for rt in ['fr', 'wfr', 'tr', 'wtr']:
        df_races_filtered = df_races.merge(
            df_regatta_info,
            on='Regatta',
            how='inner'
        )
        df_races_filtered = df_races_filtered[df_races_filtered['ratingType'] == rt]
                
        people, allFrRaces, allTrRaces = calcAllRacesForRT(rt, people, df_races_filtered, allFrRaces, allTrRaces, calculatedAtDict, config)
    
    return people, allFrRaces, allTrRaces

def upload(people : dict[str, Sailor], allFrRows, allTrRows, df_rivals, config: Config):
    # Create a connection
    connection = mysql.connector.connect(
        host=os.getenv('DB_HOST'),
        port=os.getenv('DB_PORT'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASS'),
        database=os.getenv('DB_NAME')
    )

    uploadSailors(people, connection, config)
    uploadTeams(people, connection, config)
    uploadAllScores(allFrRows, allTrRows, connection)
    uploadRivals(df_rivals, connection)
    
    connection.close()
    
def load(rootDir : str, config: Config):
    load_dotenv()
    
    if config.doScrape:
        df_races_fr = runFleetScrape("racesfrtest.parquet", "racesfrtest.parquet") 
        df_races_tr = scrapeTR("racesTR.json","racesTR.json", "trSailorInfoAll.json")
        df_sailor_info = runSailorData("racesfrtest.parquet", "trSailorInfoAll.json", "sailor_data2.parquet", "sailor_data2.parquet")
    else: 
        print("Reading from files.")
        df_races_fr = pd.read_parquet(rootDir + "racesfrtest.parquet")
        df_races_tr = pd.read_parquet(rootDir + "racesTR.parquet")
        df_sailor_info = pd.read_parquet(rootDir + "sailor_data2.parquet")
        
    df_races_full = pd.concat([df_races_fr, df_races_tr])
    
    # clean up memory
    del df_races_fr, df_races_tr
    
    df_races_full = df_races_full.sort_values(['Date', 'raceNum', 'Div']).reset_index(drop=True)
    
    df_sailor_ratings = None
    if not config.calcAll:
        # cutoff = (datetime.now() - timedelta(weeks=2))
        # df_races_full = df_races_full.loc[df_races_full['Date'] > cutoff]
        df_sailor_ratings = pd.read_json(rootDir + "sailors-latest.json")

    try:
        with open("calculated_at_dict.json", "r") as f:
            calculatedAtDict = json.load(f)
    except FileNotFoundError:
        calculatedAtDict = {}
        
        
    df_oldFrPostCalcRaces = pd.read_parquet(rootDir + 'postcalcfrraces.parquet')
    df_oldTrPostCalcRaces = pd.read_parquet(rootDir + 'postcalctrraces.parquet')
    
    return df_races_full, df_sailor_info, df_sailor_ratings, calculatedAtDict, df_oldFrPostCalcRaces, df_oldTrPostCalcRaces

def main(rootDir : str = "", jupyter = False):
    
    config : Config = Config()

    df_races_full, df_sailor_info, df_sailor_ratings, calculatedAtDict, df_oldFrPostCalcRaces, df_oldTrPostCalcRaces = load(rootDir, config)

    print("Loading complete.\nStarting setup.")

    people = setupPeople(df_sailor_ratings, df_sailor_info, config)
    people, df_races_full = handleMerges(df_races_full, people, config)

    print("Setup complete.\nStarting calculations.")
    
    regatta_info = calculateAllRegattaInfo(people, df_races_full, calculatedAtDict, config)
    people, allFrRaces, allTrRaces = calculateAllRaces(people, df_races_full, regatta_info, calculatedAtDict, config)
    people = calculateSailorRanks(people, config)
    updateSailorRatios(people)
    df_rivals = buildRivals(df_races_full, config)
    
    existing_race_ids = set(race.get('raceID') for race in allFrRaces)
    new_rows = df_oldFrPostCalcRaces[~df_oldFrPostCalcRaces['raceID'].isin(existing_race_ids)]
    allFrRaces.extend(new_rows.to_dict('records'))
    
    existing_race_ids = set(race.get('raceID') for race in allTrRaces)
    new_rows = df_oldTrPostCalcRaces[~df_oldTrPostCalcRaces['raceID'].isin(existing_race_ids)]
    allTrRaces.extend(new_rows.to_dict('records'))
    
    print("Calculations finished.\nOutputting to files")
    
    with open("calculated_at_dict.json", "w") as f:
        json.dump(calculatedAtDict, f)
    
    if jupyter:
        return people, df_races_full
    
    outputSailorsToFile(people, config)
    
    df_rivals.to_parquet('rivalstesting.parquet')
    
    df_frAfter = pd.DataFrame(allFrRaces)
    df_frAfter.to_parquet("postcalcFRraces.parquet")
    
    df_trAfter = pd.DataFrame(allTrRaces)
    df_trAfter.to_parquet("postcalcTRraces.parquet")
    
    print("File output finished.")
    
    if config.doUpload:
        print("Uploading to db")
        upload(people, df_rivals, config)

if __name__ == "__main__":
    # with cProfile.Profile() as profile:
    start = time.time()
    main()
    end = time.time()
    print(f"{int((end-start) // 60)}:{int((end-start) % 60)}")
    
    # results = pstats.Stats(profile)
    # results.sort_stats(pstats.SortKey.TIME)
    # results.print_stats()
    # results.dump_stats('profiling.prof')