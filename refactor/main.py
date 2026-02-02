from AsyncScraper import runFleetScrape
from TRScraper import scrapeTR
from dataScraper import runSailorData

from calculationsFR import calculateFR
from calculationsTR import calculateTR

from uploadScores import uploadScoresBySailor
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

        startingRating = [r.ordinal(target=config.targetElo, alpha=config.alpha) for r in ratings]
        tempRating += sum(startingRating)

    regattaAvg = tempRating / len(skippers + crews)
    
    return womens, regattaAvg
    
def calculateAllRaces(people, df_races, config: Config):
    leng = len(df_races['adjusted_raceID'].unique())
    regatta_groups = df_races.groupby(['Regatta'], sort=False)
    i = 0
    
    for regatta_name, regatta_data in regatta_groups:
        scoring = getScoring(regatta_data)
        season = regatta_data.iloc[0]['raceID'].split("/")[0]

        if scoring == 'team':
            womens, regattaAvg = getWomensAndRegAvgTR(people, regatta_data, config)
        else:
            womens, regattaAvg = getWomensAndRegAvgFR(people, regatta_data, config)

        race_groups = regatta_data.groupby(['Date', 'adjusted_raceID'], sort=False)

        # Iterate through each race in this regatta
        for (date, race), row in race_groups:
            i += 1
            if i % 1000 == 0:
                print(f"Currently analyzing race {i}/{leng} in {regatta_name}, Date:{date}")

            for pos in ['Skipper', 'Crew']:
                if scoring == 'team':
                    calculateTR(people, date, row, pos, season, regattaAvg, womens, config)
                else:
                    calculateFR(people, date, regatta_name, race, row, pos, scoring, season, regattaAvg, womens, config)
    return people

def upload(people : dict[str, Sailor], config: Config):
    # Create a connection
    connection = mysql.connector.connect(
        host=os.getenv('DB_HOST'),
        port=os.getenv('DB_PORT'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASS'),
        database=os.getenv('DB_NAME')
    )

    # uploadSailors(people, connection, config)
    # uploadTeams(people, connection, config)
    uploadScoresBySailor(people, connection)
    
    connection.close()
    
def load(rootDir : str, config: Config):
    load_dotenv()
    
    if config.doScrape:
        df_races_fr = runFleetScrape("racesfrtest.json", "racesfrtest.json") 
        df_races_tr = scrapeTR("racesTR.json","racesTR.json", "trSailorInfoAll.json")
        df_sailor_info = runSailorData("racesfrtest.json", "trSailorInfoAll.json", "sailor_data2.json")
    else: 
        print("Reading from files.")
        df_races_fr = pd.read_json(rootDir + "racesfrtest.json")
        df_races_tr = pd.read_json(rootDir + "racesTR.json")
        df_sailor_info = pd.read_json(rootDir + "sailor_data2.json")
        
    df_races_full = pd.concat([df_races_fr, df_races_tr])
    
    # clean up memory
    del df_races_fr, df_races_tr
    
    df_races_full = df_races_full.sort_values(['Date', 'raceNum', 'Div']).reset_index(drop=True)    
    
    df_sailor_ratings = None
    if not config.calcAll:
        cutoff = (datetime.now() - timedelta(weeks=2))
        df_races_full = df_races_full.loc[df_races_full['Date'] > cutoff]
        df_sailor_ratings = pd.read_json(rootDir + "sailors-latest.json")
    
    return df_races_full, df_sailor_info, df_sailor_ratings


def main(rootDir : str = "", jupyter = False):
    
    config : Config = Config()

    df_races_full, df_sailor_info, df_sailor_ratings = load(rootDir, config)

    print("Loading complete.\nStarting setup.")

    people = setupPeople(df_sailor_ratings, df_sailor_info, config)
    people, df_races_full = handleMerges(df_races_full, people, config)

    print("Setup complete.\nStarting calculations.")
    
    people = calculateAllRaces(people, df_races_full, config)
    people = calculateSailorRanks(people, config)
    updateSailorRatios(people)
    
    print("Calculations finished.\nOutputting to files")
    
    if jupyter:
        return people, df_races_full
    
    outputSailorsToFile(people, config)
    
    print("File output finished.")
    if config.doUpload:
        print("Uploading to db")
        upload(people, config)

if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"{int((end-start) // 60)}:{int((end-start) % 60)}")