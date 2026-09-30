# %% Imports
# %load_ext autoreload
# %autoreload 2
# rootDir = "./../"

from AsyncScraper import runFleetScrape
from TRScraper import scrapeTR
from dataScraper import runSailorData

from calculationsFR import calculateFR
from calculationsTR import calculateTR

from chatRivals import buildRivals, uploadRivals
from regionOffsets import computeRegionOffsets, applyRegionOffsets

from uploadScores import uploadAllScores, updateHomepageStats
from Teams import uploadTeams

from Sailors import Sailor, setupPeople, handleMerges, loadRatingHistory, outputSailorsToFile, calculateSailorRanks, uploadSailors, updateSailorRatios
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

def getPeopleFR(people, regatta_data,config):
    skipper_keys = regatta_data.loc[regatta_data['Position'] == 'Skipper']['key'].unique()
    skippers = [people[k] for k in skipper_keys if k in people.keys()]

    crew_keys = regatta_data.loc[regatta_data['Position'] == 'Crew']['key'].unique()
    crews = [people[k] for k in crew_keys if k in people.keys()]
    return skippers, crews,

def getWomensFR(people, regatta_data, config: Config):
    skippers, crews = getPeopleFR(people, regatta_data, config)

    genders = [p.gender for p in skippers + crews]
    womenCount = sum([1 if g == "F" else 0 for g in genders])
    womens = 'M' not in genders and womenCount >= 4
    return womens

def getRegAvgFR(people, womens, regatta_data, config: Config):
    skippers, crews = getPeopleFR(people, regatta_data, config)
    
    tempRating = 0
    for type, racers in zip(['Skipper', 'Crew'], [skippers, crews]):
        if womens:
            ratings = [r.wsr if type ==
                        'Skipper' else r.wcr for r in racers]
        else:
            ratings = [r.sr if type ==
                        'Skipper' else r.cr for r in racers]

        startingRatings = [
            r.ordinal(target=config.targetElo, alpha=config.alpha) for r in ratings]
        tempRating += sum(startingRatings)

    regattaAvg = tempRating / len(skippers + crews)
    
    return regattaAvg

def getPeopleInTR(people, regatta_data, config: Config):
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
    return skippers, crews

def getWomensTR(people, regatta_data, config: Config):
    skippers, crews = getPeopleInTR(people,regatta_data, config)

    genders = [p.gender for p in skippers + crews]
    womenCount = sum([1 if g == "F" else 0 for g in genders])
    womens = 'M' not in genders and womenCount >= 4
    return womens

def getRegAvgTR(people, womens, regatta_data, config : Config):
    skippers, crews = getPeopleInTR(people, regatta_data, config)

    tempRating = 0
    for type, racers in zip(['Skipper', 'Crew'], [skippers, crews]):
        if womens:
            ratings = [r.wtsr if type == 'Skipper' else r.wtcr for r in racers]
        else:
            ratings = [r.tsr if type == 'Skipper' else r.tcr for r in racers]

        startingRating = [r.ordinal(target=config.targetElo, alpha=config.alpha) for r in ratings]
        tempRating += sum(startingRating)

    regattaAvg = tempRating / len(skippers + crews)
    
    return regattaAvg
    
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
            womens = getWomensTR(people, regatta_data, config)
        else:
            womens = getWomensFR(people, regatta_data, config)
        
        ratingType = ('w' if womens else '') + ('tr' if scoring == 'team' else 'fr')
        
        regattaDict[regatta_name] = {'scoring': scoring, 'womens': womens, 'ratingType': ratingType}    

    return regattaDict

def regattaDate(regatta_data):
    """Timestamp of a regatta, from whichever type the Date column happens to hold."""
    date = regatta_data.iloc[0]['Date']
    if type(date) == str:
        fmt = "%Y-%m-%d %H:%M:%S" if len(date) != 10 else "%Y-%m-%d"
        return datetime.strptime(date, fmt).timestamp()
    if type(date) != float:
        return date.timestamp()
    return date

def findResetDate(regatta_groups, ratingType, calculatedAtDict):
    """Earliest date whose regatta data changed since it was last calculated.

    Found up front, over every regatta, rather than latching onto the first stale one
    encountered. Most dates carry several regattas (up to 19 in f25), and the sweep
    resumes at a regatta boundary while the rollback happens at a date boundary. Taking
    the minimum makes the two agree: everything from resetDate onward is recalculated,
    and everything before it is left alone.
    """
    resetDate = None
    for regatta_name, regatta_data in regatta_groups:
        season = regatta_data.iloc[0]['raceID'].split("/")[0]
        calculatedAt = calculatedAtDict.get(f"{ratingType}/{season}")
        updatedAt = regatta_data.iloc[0]['updatedAt']

        if calculatedAt is None or updatedAt > calculatedAt:
            date = regattaDate(regatta_data)
            if resetDate is None or date < resetDate:
                resetDate = date
    return resetDate

def calcAllRacesForRT(ratingType, people, df_races, allFrRaces, allTrRaces, calculatedAtDict: dict, config: Config):
    print(f"Calculating all {ratingType} races")
    leng = len(df_races['adjusted_raceID'].unique())
    regatta_groups = df_races.groupby(['Regatta'], sort=False)
    
    i = 0

    if config.calcAll:
        resetDate = None
    else:
        resetDate = findResetDate(regatta_groups, ratingType, calculatedAtDict)
        if resetDate is None:
            print(f"Nothing to recalculate for {ratingType}")
            return people, allFrRaces, allTrRaces
        print(f"Recalculating {ratingType} from {resetDate} onward")

    for regatta_name, regatta_data in regatta_groups:
        season = regatta_data.iloc[0]['raceID'].split("/")[0]
        scoring = regatta_data.iloc[0]['scoring']
        womens = regatta_data.iloc[0]['womens']
        date = regattaDate(regatta_data)

        # Same boundary the ratings are rolled back to, so a regatta is never rolled
        # back without also being recomputed.
        if resetDate is not None and date < resetDate:
            continue

        race_groups = regatta_data.groupby(['adjusted_raceID'], sort=False)

        # Keyed by rating type AND season. Keying on season alone meant the first of the
        # four passes (fr) stamped the checkpoint, so wfr/tr/wtr then saw it already set
        # and skipped every regatta.
        calculatedAtDict[f"{ratingType}/{season}"] = time.time()
        
        if scoring == 'team':
            regattaAvg = getRegAvgTR(people, womens,regatta_data, config)
        else: 
            regattaAvg = getRegAvgFR(people, womens,regatta_data, config)
            

        # Iterate through each race in this regatta
        for raceID, row in race_groups:
            i += 1
            if i % 1000 == 0:
                print(f"Currently analyzing race {i}/{leng} in {regatta_name[0]}, Date:{date}")

            for pos in ['Skipper', 'Crew']:
                if scoring == 'team':
                    calculateTR(allTrRaces, people, resetDate, date, row, pos, season, regattaAvg, womens, ratingType, config)
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

def upload(people : dict[str, Sailor], df_frAfter, df_trAfter, df_rivals, outlinks_dict, racecounts_dict, winp_dict, team_link_map, config: Config):
    # Create a connection
    connection = mysql.connector.connect(
        host=os.getenv('DB_HOST'),
        port=os.getenv('DB_PORT'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASS'),
        database=os.getenv('DB_NAME'), 
        allow_local_infile=True
    )

    uploadSailors(people, connection, config)
    uploadTeams(people, outlinks_dict, racecounts_dict, winp_dict, team_link_map, connection, config)
    uploadAllScores(df_frAfter, df_trAfter, connection)
    uploadRivals(df_rivals, connection)
    updateHomepageStats(connection)
    
    connection.close()
    
def load(rootDir : str, config: Config):
    load_dotenv()
    
    if config.doScrape:
        df_races_fr = runFleetScrape("racesfr.parquet", "racesfr.parquet") 
        df_races_tr = scrapeTR("racesTR.parquet", "racesTR.parquet", "trSailorInfoAll.json")
        df_sailor_info = runSailorData("racesfr.parquet", "trSailorInfoAll.json", "sailor_data2.parquet", "sailor_data2.parquet")
    else: 
        print("Reading from files.")
        df_races_fr = pd.read_parquet(rootDir + "racesfr.parquet")
        df_races_tr = pd.read_parquet(rootDir + "racesTR.parquet")
        df_sailor_info = pd.read_parquet(rootDir + "sailor_data2.parquet")
        
    df_races_full = pd.concat([df_races_fr, df_races_tr])
    
    # clean up memory
    del df_races_fr, df_races_tr
    
    df_races_full = df_races_full.sort_values(['Date', 'raceNum', 'Div']).reset_index(drop=True)
    
    df_sailor_ratings = None
    df_oldFrPostCalc = None
    df_oldTrPostCalc = None
    if not config.calcAll:
        # cutoff = (datetime.now() - timedelta(weeks=2))
        # df_races_full = df_races_full.loc[df_races_full['Date'] > cutoff]
        df_sailor_ratings = pd.read_json(rootDir + "sailors-latest.json")
        # Read once here: needed both to rebuild rating history before the sweep and to
        # carry forward rows afterwards.
        df_oldFrPostCalc = pd.read_parquet(rootDir + 'postcalcFRraces.parquet')
        df_oldTrPostCalc = pd.read_parquet(rootDir + 'postcalcTRraces.parquet')

    try:
        with open(rootDir + "calculated_at_dict.json", "r") as f:
            calculatedAtDict = json.load(f)
    except FileNotFoundError:
        calculatedAtDict = {}
    
    return df_races_full, df_sailor_info, df_sailor_ratings, df_oldFrPostCalc, df_oldTrPostCalc, calculatedAtDict
# %%
def main(rootDir : str = "", jupyter = False):
    # %% Load Files
    
    config : Config = Config()

    df_races_full, df_sailor_info, df_sailor_ratings, df_oldFrPostCalc, df_oldTrPostCalc, calculatedAtDict = load(rootDir, config)
    
    # df_races_full = df_races_full.loc[df_races_full['season'] == 'f25']

    print("Loading complete.\nStarting setup.")

    # %% Setup people
    people = setupPeople(df_sailor_ratings, df_sailor_info, config)
    del df_sailor_info, df_sailor_ratings
    people, df_races_full = handleMerges(df_races_full, people, config)

    if not config.calcAll:
        people = loadRatingHistory(people, pd.concat([df_oldFrPostCalc, df_oldTrPostCalc]), config)
    
    print("Setup complete.\nStarting calculations.")
    
    # %% Calculations
    regatta_info = calculateAllRegattaInfo(people, df_races_full, calculatedAtDict, config)
    people, allFrRaces, allTrRaces = calculateAllRaces(people, df_races_full, regatta_info, calculatedAtDict, config)
    
    del regatta_info
    
    df_cleaned = df_races_full.dropna(
        subset=['Team', 'Teamlink']).drop_duplicates(subset='Team', keep='first')
    team_link_map = pd.Series(df_cleaned.Teamlink.values, index=df_cleaned.Team).to_dict()
    
    df_rivals = buildRivals(df_races_full, config)

    # Every post-calc row has to still correspond to a scraped race entry. A team that
    # fixes the wrong sailors it entered leaves no scraped row for the people taken off,
    # so races not recalculated this run would otherwise carry their old rows forward.
    validFrRows = set()
    df_frSource = df_races_full.loc[df_races_full['Scoring'] != 'team']
    if not config.calcAll:
        validFrRows = set(zip(df_frSource['raceID'], df_frSource['key']))

    # Kept past the del: the region-offset fit needs each sailor's contemporaneous team.
    df_regionSource = df_frSource[['raceID', 'key', 'Team']].copy()
    del df_frSource

    del df_races_full

    if not config.calcAll:
        existing_race_ids = set(race.get('raceID') for race in allFrRaces)
        new_rows = df_oldFrPostCalc[~df_oldFrPostCalc['raceID'].isin(existing_race_ids)]
        carried = [r for r in new_rows.to_dict('records')
                   if (r['raceID'], r['sailorID']) in validFrRows]
        if len(carried) != len(new_rows):
            print(f"Dropped {len(new_rows) - len(carried)} carried-forward fleet rows no longer in the scrape")
        allFrRaces.extend(carried)

        existing_race_ids = set(race.get('raceID') for race in allTrRaces)
        new_rows = df_oldTrPostCalc[~df_oldTrPostCalc['raceID'].isin(existing_race_ids)]
        allTrRaces.extend(new_rows.to_dict('records'))

    del df_oldFrPostCalc, df_oldTrPostCalc
    
    df_frAfter = pd.DataFrame(allFrRaces)
    df_trAfter = pd.DataFrame(allTrRaces)
    
    del allFrRaces, allTrRaces

    # Region offsets, then ranks. Ranking happens after this so it sees the corrected
    # ratings; the fit needs the fully assembled frame, which is why it sits here rather
    # than beside the sweep.
    if config.useRegionOffsets:
        print("Fitting region offsets")
        offsets = computeRegionOffsets(df_frAfter, df_regionSource, config)
        people = applyRegionOffsets(people, offsets, config)
    del df_regionSource

    people = calculateSailorRanks(people, config)

    # %%
    outlinks_dict = df_frAfter.groupby('sailorID')['outLinks'].sum().to_dict()
    
    df_trAfter['ratio'] = df_trAfter['outcome'].map({'win': 1, 'lose': 0, 'tie': 0.5})
    combined = pd.concat([df_frAfter, df_trAfter])
    combined['position'] = combined['position'].apply(lambda x: x.lower())
    
    winp_dict = combined.loc[combined['ratio'] >= 0].groupby(['sailorID', 'position', 'season'])['ratio'].mean().to_dict()

    # Needs `combined`, so it runs here rather than beside calculateSailorRanks.
    updateSailorRatios(people, combined)
    
    counts = combined.groupby(['sailorID', 'position', 'season']).size()

    racecounts_dict = {}
    for (s_id, pos, season), count in counts.items():
        racecounts_dict.setdefault(s_id, {}).setdefault(pos.lower(), {})[season] = count
    
    del counts
    
    print("Calculations finished.\nOutputting to files")
    
    # %% File Output
    
    outputSailorsToFile(people, rootDir, config)
    
    df_rivals.to_parquet(rootDir + 'rivalstesting.parquet')
    
    df_frAfter.to_parquet(rootDir + "postcalcFRraces.parquet")
    df_trAfter.to_parquet(rootDir + "postcalcTRraces.parquet")
    
    with open("calculated_at_dict.json", "w") as f:
        json.dump(calculatedAtDict, f) 
    
    print("File output finished.")
    
    # %% Upload data
    if config.doUpload:
        print("Uploading to db")
        upload(people, df_frAfter, df_trAfter, df_rivals, outlinks_dict, racecounts_dict, winp_dict, team_link_map, config)

# %% Main 
if __name__ == "__main__":
    # with cProfile.Profile() as profile:
    start = time.time()
    try:
        main()
    except Exception as e:
        end = time.time()
        print(f"{int((end-start) // 60)}:{int((end-start) % 60)}")
        raise e
        
    end = time.time()
    print(f"{int((end-start) // 60)}:{int((end-start) % 60)}")
    
    # results = pstats.Stats(profile)
    # results.sort_stats(pstats.SortKey.TIME)
    # results.print_stats()
    # results.dump_stats('profiling.prof')  