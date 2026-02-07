from bs4 import BeautifulSoup
import requests
import httpx
import asyncio
import pandas as pd
import numpy as np
import time
from datetime import date, datetime, timezone
import os
from concurrent.futures import ProcessPoolExecutor
import json

from TRScraper import makeSailorList, makeRaceList, getSailorLinks, getTeamReportInfo, getTeamIDs

def getRaceNums(oldNums, scoresLen):
    newNums = []
    if oldNums == [['']]:
        newNums = list(range(1, scoresLen + 1))
    elif len(oldNums) > 0:
        for i, num in enumerate(oldNums):
            if len(num) > 1:
                for j in range(int(num[0]), int(num[1]) + 1):
                    newNums.append(j)
            else:
                newNums.append(int(num[0]))
    return newNums

def makeRaceSeries(score, team, raceNum, division, name, link, gradYear, position, partner, partnerLink, venue, regatta, date, teamlink, scoring, boat, teamBoatName):
    raceSeries = pd.Series()
    raceSeries["Score"] = score
    raceSeries["Div"] = division
    raceSeries["Sailor"] = name
    raceSeries["Link"] = link if link is not None else "Unknown"
    raceSeries["key"] = link if link is not None else name + "-" + team
    raceSeries["GradYear"] = gradYear
    raceSeries["Position"] = position
    raceSeries["Partner"] = partner
    raceSeries["PartnerLink"] = partnerLink if partnerLink is not None else "Unknown"
    raceSeries["partnerKey"] = partnerLink if partnerLink is not None else partner + "-" + team
    raceSeries["Venue"] = venue
    raceSeries["Regatta"] = regatta
    raceSeries["Scoring"] = scoring
    raceSeries['raceID'] = "" + regatta + "/" + str(raceNum) + division
    raceSeries['adjusted_raceID'] = "" + regatta + "/" + str(raceNum) if scoring == "Combined" else "" + regatta + "/" + str(raceNum) + division
    raceSeries["Date"] = date
    raceSeries["raceNum"] = str(raceNum)
    raceSeries["Team"] = team
    raceSeries["Teamlink"] = teamlink
    raceSeries["Boat"] = boat
    raceSeries["TeamBoatName"] = teamBoatName
    return raceSeries

async def conditional_get(client, link, page_url_ending, full_meta={}, page_type=''):
    if page_type == '':
        page_type = page_url_ending
        
    url = f"https://scores.collegesailing.org/{link}/{page_url_ending}{"/" if page_url_ending != '' else ''}"
    html_path = f"pages/{link}-{page_type}.html"
    meta = full_meta[page_type]

    headers = {}
    if "etag" in meta:
        headers["If-None-Match"] = meta["etag"]
    if "last_modified" in meta:
        headers["If-Modified-Since"] = meta["last_modified"]

    r = await client.get(url, headers=headers)

    # Not modified -> load cached HTML
    if r.status_code == 304:
        with open(html_path, "r") as f:
            return f.read(), meta, False

    r.raise_for_status()

    html = r.text
    with open(html_path, "w") as f:
        f.write(html)

    # Update global metadata
    if "ETag" in r.headers:
        meta["etag"] = r.headers["ETag"]
    if "Last-Modified" in r.headers:
        meta["last_modified"] = r.headers["Last-Modified"]

    meta["last_checked"] = datetime.now(timezone.utc).isoformat()

    return html, meta, True

async def cleanup_semaphore(semaphore):
    if semaphore.locked():
        # Forcefully release if it was accidentally left locked
        semaphore.release()
        print("Semaphore was released manually.")

async def fetchData(client, semaphore, regattaID, link, scoring, date, meta, missing):
    retries = 10
    backoff = 1
    if not os.path.exists(f"pages/{link.split("/")[0]}"):
        os.makedirs(f"pages/{link.split("/")[0]}")
    for attempt in range(retries):
        try:
            async with semaphore:  # Limit concurrent requests
                if scoring == "Team":
                    # All races
                    html, newMeta, achanged = await conditional_get(client, link, "all", meta)
                    fullScores = BeautifulSoup(html, 'html.parser')

                    # sailors
                    html, newSMeta, schanged = await conditional_get(client, link, "sailors", meta)
                    sailors = BeautifulSoup(html, 'html.parser')
                    
                    # report
                    html, newRMeta, rchanged = await conditional_get(client, link, "", meta, "report")
                    report = BeautifulSoup(html, 'html.parser')
                    
                    full_meta = {"all": newMeta, "sailors": newSMeta, 'report': newRMeta}
                    process = achanged or schanged or rchanged or missing
                    
                    await cleanup_semaphore(semaphore)
                    return {'regattaID': regattaID, 'allRaces': fullScores, "sailors": sailors, 'report': report, 'scoring':scoring, 'date': date, 'meta': full_meta, 'process': process}
                else:
                    # full scores
                    html, newMeta, changed = await conditional_get(client, link, "full-scores", meta)
                    fullScores = BeautifulSoup(html, 'html.parser')

                    # sailors
                    html, newSMeta, schanged = await conditional_get(client, link, "sailors", meta)
                    sailors = BeautifulSoup(html, 'html.parser')
                    
                    full_meta = {"full-scores": newMeta, "sailors": newSMeta}
                    
                    await cleanup_semaphore(semaphore)
                    return {'regattaID': regattaID, 'fullScores': fullScores, "sailors": sailors, 'scoring':scoring, 'date': date, 'meta': full_meta, 'process': changed or schanged or missing}
        except httpx.ConnectTimeout as e:
            print(f"Connection timeout when fetching {link}. Retrying... ({attempt + 1}/{retries})")
            await asyncio.sleep(backoff)  # Wait before retrying
            backoff *= 2  # Exponential backoff
        except httpx.ReadError as e:
            print(f"Read error when fetching {link}. Retrying... ({attempt + 1}/{retries})")
            await asyncio.sleep(backoff)  # Wait before retrying
            backoff *= 2  # Exponential backoff
        except httpx.HTTPStatusError as e:
            print(f"HTTP error {e.response.status_code} when fetching {link}. Skipping...")
            await cleanup_semaphore(semaphore)
            return None
        except httpx.RequestError as e:
            print(f"Request error {e}. Skipping...")
            await cleanup_semaphore(semaphore)
            return None

# need to deal with redress
def parseScore(scoreString):
    if scoreString.text.isdigit():
        return int(scoreString.text)
    elif scoreString.has_attr('title'):
        return int(scoreString['title'][1:-1].split(",")[0].split(":")[0])

def addRaces(finalRaces, teamScores, sailors, others, pos, teamHome, host, regatta, teamLink, scoring, boat_type,teamBoatName, raceDate):
    for sailor in sailors:
        partners = [other['name'] for race in sailor['races'] for other in others if other['div'] == sailor['div'] and race in other['races']]
        partnerLinks = [other['link'] for race in sailor['races'] for other in others if other['div'] == sailor['div'] and race in other['races']]
        for i, score in enumerate(teamScores[sailor['div']]):
            if i + 1 in sailor['races']:
                partner = partners[sailor['races'].index(i + 1)] if sailor['races'].index(i + 1) < len(partners) else "Unknown"
                partnerLink = partnerLinks[sailor['races'].index(i + 1)] if sailor['races'].index(i + 1) < len(partners) else "Unknown"
                finalRaces.append(makeRaceSeries(score, teamHome, i + 1, sailor['div'], sailor['name'], sailor['link'],sailor['year'], pos, partner, partnerLink, host, regatta, raceDate, teamLink, scoring, boat_type, teamBoatName))



def processData(regattaData):
    if regattaData is None:
        print("None soup...?")
        return []
    
    regattaID = regattaData['regattaID']
    scoring = regattaData['scoring']
    raceDate = regattaData['date']
    
    if scoring == 'Team':
        return []
        allRaces = regattaData['allRaces']
        sailors = regattaData['sailors']
        reportPage = regattaData['report']
        return processTeamRegatta(regattaID, allRaces, sailors, reportPage, raceDate)
    else:
        fullScores = regattaData['fullScores']
        sailors = regattaData['sailors']
        return processFleetRegatta(regattaID, fullScores, sailors, scoring, raceDate)

def processTeamRegatta(regattaID, allRaces, sailors, reportPage, raceDate):
    data = []
    totalSailors = {}
    
    if len(allRaces.find_all('table', class_="teamscorelist")) == 0: 
        print(f"no scores entered for {regattaID}, skipping")
        return

    scoreData = allRaces.find_all('table', class_="teamscorelist")[
        0].contents

    host = allRaces.find("span", itemprop='location').text
    date = allRaces.find("time").attrs['datetime']
    date = date[:10]

    regattaType = allRaces.find("span", itemprop='description').text
    
    host = allRaces.find("span", itemprop='location').text

    raceRows = [row for table in scoreData if table for row in table.contents][1:]

    df_races = pd.DataFrame(makeRaceList(raceRows))

    df_sailors = pd.DataFrame(makeSailorList(sailors, regattaID))
    if len(df_sailors['name'].unique()) < 2:
        print("No rp ented for this regatta", regattaID)
        return

    df_sailorLinks = pd.DataFrame(getSailorLinks(reportPage))

    df_teamReportInfo = pd.DataFrame(getTeamReportInfo(reportPage))
    df_sailorteamInfo = pd.DataFrame(getTeamIDs(sailors))

    # merge dfs 
    df_sailors = pd.merge(df_sailors, df_sailorLinks, how='left', on='name')
    # df_totalSailors = pd.concat([df_totalSailors, df_sailors])

    for _, race_result in df_races.iterrows():
        raceNum = race_result['raceNum']
        round = race_result['round']
        
        teamA = race_result['teamA']
        teamAID = df_teamReportInfo.loc[df_teamReportInfo['uniName'] == teamA['name'], 'teamID'].iat[0]
        # print(teamAID,df_sailorteamInfo)
        teamAName = df_sailorteamInfo.loc[df_sailorteamInfo['teamID'] == teamAID, 'teamName'].iat[0]
        
        teamB = race_result['teamB']
        teamBID = df_teamReportInfo.loc[df_teamReportInfo['uniName'] == teamB['name'], 'teamID'].iat[0]
        teamBName = df_sailorteamInfo.loc[df_sailorteamInfo['teamID'] == teamBID, 'teamName'].iat[0]

        allSkipperKeys = []
        allCrewKeys = []

        teamASailors = df_sailors.loc[(df_sailors['round'] == round) & (df_sailors['teamID'] == teamAID) & (df_sailors['oppID'] == teamBID)]
        teamABoats = []
        if len(teamASailors) > 0:
            skippers = teamASailors.loc[teamASailors['pos'] == 'skipper']
            crews = teamASailors.loc[teamASailors['pos'] == 'crew']
            for _, skipper in skippers.iterrows():
                crew = crews.loc[crews['partner'] == skipper['name']].iloc[0]
                
                skipperKey = skipper['link'] if skipper['link'] != 'Unknown' else skipper['name'] + "-" + teamAName
                allSkipperKeys.append(skipperKey)
                
                crewKey = crew['link'] if crew['link'] != 'Unknown' else crew['name'] + "-" + teamAName
                allCrewKeys.append(crewKey)
                
                totalSailors[skipperKey] = {'name': skipper['name'], 'year': skipper['year'], 'link': skipper['link'], 'key': skipperKey, 'team': teamAName}
                totalSailors[crewKey] = {'name': crew['name'], 'year': crew['year'], 'link': crew['link'], 'key': crewKey, 'team': teamAName}
                
                teamABoats.append({'skipperName': skipper['name'], 'skipperLink': skipper['link'],'skipperKey': skipperKey, 
                                    'crewName': crew['name'], 'crewLink': crew['link'], 'crewKey': crewKey})
        
        teamBSailors = df_sailors.loc[(df_sailors['round'] == round) & (df_sailors['teamID'] == teamBID) & (df_sailors['oppID'] == teamAID)]
        teamBBoats = []

        if len(teamBSailors) > 0:
            skippers = teamBSailors.loc[teamBSailors['pos'] == 'skipper']
            crews = teamBSailors.loc[teamBSailors['pos'] == 'crew']
            for _, skipper in skippers.iterrows():
                crew = crews.loc[crews['partner'] == skipper['name']].iloc[0]
                
                skipperKey = skipper['link'] if skipper['link'] != 'Unknown' else skipper['name'] + "-" + teamBName
                allSkipperKeys.append(skipperKey)
                
                crewKey = crew['link'] if crew['link'] != 'Unknown' else crew['name'] + "-" + teamBName
                allCrewKeys.append(crewKey)
                
                totalSailors[skipperKey] = {'name': skipper['name'], 'year': skipper['year'], 'link': skipper['link'], 'key': skipperKey, 'team': teamBName}
                totalSailors[crewKey] = {'name': crew['name'], 'year': crew['year'], 'link': crew['link'], 'key': crewKey, 'team': teamBName}
                
                teamBBoats.append({'skipperName': skipper['name'], 'skipperLink': skipper['link'],'skipperKey': skipperKey, 
                                    'crewName': crew['name'], 'crewLink': crew['link'], 'crewKey': crewKey})
        
        data.append({'raceID': f"{regattaID}/{raceNum}",
                    'adjusted_raceID': f"{regattaID}/{raceNum}",
                    'Regatta': regattaID,
                    'raceNum': raceNum, 'round': round,
                    'Date': date,
                    'Venue': host,
                    'Scoring' : 'team',
                    'allSkipperKeys': allSkipperKeys,
                    'allCrewKeys': allCrewKeys,
                    'teamAName': teamAName,
                    'teamAUni': teamA['name'],
                    'teamANick': teamA['nick'],
                    'teamALink': teamA['link'],
                    'teamAID': teamAID,
                    'teamABoats': teamABoats,
                    'teamAScore': teamA['score'],
                    'teamAOutcome': teamA['outcome'],
                    'teamBName': teamBName,
                    'teamBUni': teamB['name'],
                    'teamBNick': teamB['nick'],
                    'teamBLink': teamB['link'],
                    'teamBID': teamBID,
                    'teamBBoats': teamBBoats,
                    'teamBScore': teamB['score'],
                    'teamBOutcome': teamB['outcome'],
                    })
    return data, totalSailors

def processFleetRegatta(regattaID, fullScores, sailors, scoring, raceDate):
    
    finalRaces = []
    
    if len(fullScores.find_all('table', class_="results")) == 0: 
        print(f"no scores entered for {regattaID}, skipping")
        return
    
    page_keys = fullScores.find_all('span',class_='page-info-key')
    boat_key = [key for key in page_keys if 'Boat' in key.text][0]
    boat_type = boat_key.next_sibling.text
    
    scoreData = fullScores.find_all('table', class_="results")[0].contents[1].contents
    # sailorData = sailors.find('table', class_="sailors").contents[1].contents
    header = fullScores.find('table', class_="results").find_all('th', class_="right")
    raceCount = int(header[len(header) - 2].text)
        
    numDivisions = 1
    if scoreData[1]['class'][0] == 'divB' and scoreData[2]['class'][0] == 'totalrow':
        numDivisions = 2
    if scoreData[2]['class'][0] == 'divC':
        numDivisions = 3

    teamCount = int(len(scoreData) / (numDivisions + 1))
    
    teamHomes = [(scoreData[(k*(numDivisions + 1)) - (numDivisions + 1)].find('a').text) for k in range(teamCount)]
    
    host = fullScores.find("span", itemprop='location').text
    
    if scoring == "Combined":
        teamHomes = teamHomes * numDivisions

    # loop through teams
    for i in range(1, teamCount + 1):
        teamHome = scoreData[(i*(numDivisions + 1)) - (numDivisions + 1)].find('a').text
        teamName = scoreData[(i*(numDivisions + 1)) - (numDivisions + 1) + 1].contents[2].text
        teamLink = scoreData[(i*(numDivisions + 1)) - (numDivisions + 1)].find('a')['href'].split("/")[2]
        teamScores = {'A': [], 'B': [], 'C':[]}

        teamScores["A"] = [parseScore(scoreData[(i*(numDivisions + 1)) - (numDivisions + 1)].contents[j]) for j in range(4, (4 + raceCount))]
        if numDivisions > 1:
            teamScores["B"] = [parseScore(scoreData[(i*(numDivisions + 1)) - (numDivisions + 1) + 1].contents[j]) for j in range(4, (4 + raceCount))]
        if numDivisions > 2:
            teamScores["C"] = [parseScore(scoreData[(i*(numDivisions + 1)) - (numDivisions + 1) + 2].contents[j]) for j in range(4, (4 + raceCount))]

        # teamNameEls = [i.parent for i in sailors.find_all('a') if i['href'] == teamLink] # this actually doesnt work because what if teams have two boats...
        teamNameEls = [i for i in sailors.find_all('td', class_="teamname") if i.text == teamName and i.previous_sibling.find('a')['href'].split("/")[2] == teamLink]
        
        if len(teamNameEls) == 0:
            print("team name entered wrong. Skipping team", teamName, regattaID)
            continue
        
        teamNameEl = teamNameEls[0]

        index = 0
        row = teamNameEl.parent
        
        skippers = []
        crews = []
        
        # only chooses active sailor rows (.next_sibling is not None)
        while row is not None and row['class'][0] != "topborder" and row['class'][0] != "reserves-row" or index == 0:
            curRow = row
            while curRow.find_all('td', class_="division-cell") == []:
                curRow = curRow.previous_sibling
            division = curRow.find_all('td', class_="division-cell")[0].text

            # Get Skipper
            skipper = row.contents[len(row.contents) - 4]
            skipperName = skipper.text.split(" '")[0]
            
            # Get Crew
            crew = row.contents[len(row.contents) - 2]
            crewName = crew.text.split(" '", 1)[0]
            
            if skipperName != "" and skipperName != "No show":
                skipperRaceNums = skipper.next_sibling.text.split(",")
                skipperYear = skipper.text.split(" '")[1]
                skipperLink = skipper.find('a')
                if skipperLink != None:
                    skipperLink = skipperLink['href'].split("/")[2].split(" ")[0]
                else:
                    skipperLink = None
                    
                skippers.append({'name':skipperName, 'year': skipperYear, 'link':skipperLink, 'races': getRaceNums([i.split("-", 1) for i in skipperRaceNums], len(teamScores[division])), 'div':division})
            
            if crewName != "" and crewName != "No show":
                crewRaceNums = crew.next_sibling.text.split(",")
                crewYear = crew.text.split(" '")[1]
                crewLink = crew.find('a')
                if crewLink != None:
                    crewLink = crewLink['href'].split("/")[2].split(" ")[0]
                else:
                    crewLink = None
                    
                crews.append({'name':crewName, 'year': crewYear, 'link': crewLink, 'races': getRaceNums([i.split("-", 1) for i in crewRaceNums], len(teamScores[division])), 'div':division})
                
            row = row.next_sibling
            index += 1
        
        # check for same person in 2 places at once and discard
        # skipper and crew for same boat in same races?
        for skipper in skippers:
            for crew in crews:
                for i,race in enumerate(skipper['races']):
                    if skipper['name'] == crew['name'] and race in crew['races'] and skipper['div'] == crew['div']:
                        skippers.remove(skipper)
                        crews.remove(crew)
                        print('removed duplicate skipper/crew',skipper['name'],crew['name'], regattaID)
                        break
                        
        # Skipper for both A and B maybe shouldnt be removed? it is legal to do once
        for skipper in skippers:
            # self = 0
            # for other in skippers: 
            #     if other['name'] == skipper['name']:
            #         self +=1
            # if self > 1 and skipper in skippers:
            #     skippers.remove(skipper)
            #     print('removed duplicate skipper',skipper['name'], regatta)
            #     continue
            
            # sail for two different boats at the same time (impossible)
            for i, score in enumerate(teamScores[skipper['div']]):
                    if i + 1 in skipper['races']:
                        for race in finalRaces:
                            if race['raceID'] == f"{regattaID}/{str(i + 1)}{skipper['div']}":
                                if race['Sailor'] == skipper['name'] and race['Div'] == skipper['div']:
                                    print("found illegal duplicate skipper:", skipper['name'], regattaID)
                                    finalRaces = [s for s in finalRaces if not s.equals(race)]
                                    skipper['races'].remove(i+1)

        # Crew for both A and B maybe shouldnt be removed? it is legal to do once
        for crew in crews:
        #     self = 0
        #     for other in crews: 
        #         if other['name'] == crew['name']:
        #             self +=1
        #     if self > 1 and crew in crews:
        #         crews.remove(crew)
        #         print('removed duplicate crew',crew['name'], regatta)
            
            for i, score in enumerate(teamScores[crew['div']]):
                    if i + 1 in crew['races']:
                        for race in finalRaces:
                            if race['raceID'] == f"{regattaID}/{str(i + 1)}{crew['div']}":
                                if race['Sailor'] == crew['name'] and race['Div'] == crew['div']:
                                    print("found illegal duplicate crew:", crew['name'], regattaID)
                                    finalRaces = [s for s in finalRaces if not s.equals(race)]
                                    crew['races'].remove(i+1)
        
        # update skippers and crews once all rows for a team are done.
        addRaces(finalRaces, teamScores, skippers, crews, 'Skipper', teamHome, host, regattaID, teamLink, scoring, boat_type, teamName, raceDate)
        addRaces(finalRaces, teamScores, crews, skippers, 'Crew', teamHome, host, regattaID, teamLink, scoring, boat_type,teamName, raceDate)
        skippers = []
        crews = []
        
    # Test for duplicate scores
    # for pos in ['Skipper', 'Crew']:
    #     for division in range(numDivisions):
    #         div = ['A', 'B', 'C'][division]
    #         for race in range(raceCount):
    #             seen = []
    #             for score in [r for r in finalRaces if r['raceID'].split("/")[2] == str(race) + div and r['Position'] == pos]:
    #                 if score['Score'] not in seen:
    #                     seen.append(score['Score'])
    #                 else:
    #                     # print('Duplicate score found in', regatta, race)
    #                     # print(score['Score'], score['Sailor'])
    #                     print(f"Duplicate score found in {regatta} race:{str(race)} score:{str(score['Score'])} {score['Sailor']}")
    
    return finalRaces

async def process_in_process(executor, result):
    loop = asyncio.get_event_loop()
    try:
        output = await loop.run_in_executor(executor, processData, result)
        return output
    except Exception as e:
        print("ERROR:", e)
        raise

async def getBatch(client, regattaKeys, regattaValues, semaphore, executor):
    tasks = []
    for i, regatta in enumerate(regattaValues):
        regattaID = regattaKeys[i]
        tasks.append(fetchData(client, semaphore, regattaID, regatta['link'], regatta['scoring'], regatta['date'], regatta['meta'], regatta['missing']))
    results = await asyncio.gather(*tasks)
    
    tasks = [process_in_process(executor, regatta) for regatta in results if regatta is not None and regatta['process']]
    print(f"{len(results)} regattas were checked and {len(tasks)} changed or need to be processed")
    rows = await asyncio.gather(*tasks)
    return rows

async def main(regattas):
    async with httpx.AsyncClient(timeout=httpx.Timeout(None,connect=15.0, read=10.0)) as client:
        allRows = []
        batchSize = 100
        semaphore = asyncio.Semaphore(batchSize)
        executor = ProcessPoolExecutor()  # Process pool for CPU-bound work
        
        for j in range (0, len(regattas.values()), batchSize):
            print(f"Processing batch {j // batchSize + 1}/{len(regattas.values()) // batchSize + 1}...")
            batchKeys = list(regattas.keys())[j:j + batchSize]
            batchRegattas = list(regattas.values())[j:j + batchSize]
            results = await getBatch(client,batchKeys, batchRegattas, semaphore, executor)
            allRows.extend(results)
            
        return allRows


def runFleetScrape(loadfile, outfile):
    print("----SCRAPING FLEET RACING----")
    start = time.time()
    seasons = [sub for s in [[f"f{i}",f"s{i}"] for i in range(10,27)] for sub in s]
    # seasons = ['s14', 'f14']
    
    # seasons = ['f25']

    df_races = pd.DataFrame()
    try:
        print("attempting to read from file")
        df_races = pd.read_parquet(loadfile) 
        print("read from file")
    except:
        df_races = pd.DataFrame(columns=["Score", "Div", "Sailor", "Link", "key", "GradYear", "Position", "Partner", "Venue", "Regatta", "Scoring", "raceID", "adjusted_raceID", "Date", "raceNum", "Team", "Teamlink", "Boat", "TeamBoatName"]) 

    racesRegattas = df_races['Regatta'].unique()
    
    try:
        with open("pages/scrape_state.json", "r") as f:
            scrape_state = json.load(f)
    except FileNotFoundError:
        scrape_state = {}
    
    # validScorings = ["3 Divisions", "2 Divisions", "Combined", "Team"]
    validScorings = ["3 Divisions", "2 Divisions", "Combined"]
    
    regattas = {}
    for season in seasons:
        print("getting all regattas in", season)
        url = f"https://scores.collegesailing.org/{season}/"
        page = requests.get(url)
        listSoup = BeautifulSoup(page.content, 'html.parser')
        
        try:
            tbody = listSoup.find('table', class_="season-summary").find('tbody')
        except Exception as e:
            print(e)
            continue
        
        for link in tbody.find_all("a", href=True):
            scoring = link.parent.next_sibling.next_sibling.next_sibling.text
            regatta_date = link.parent.next_sibling.next_sibling.next_sibling.next_sibling.text
                
            regatta_link = season + "/" + link['href']
            recent = season in seasons[-2:]
            missing = regatta_link not in racesRegattas
            
            if scoring in validScorings and (missing or recent):
                meta = scrape_state.setdefault(regatta_link, {'full-scores' : {}, 'sailors' : {}})
                regattas[regatta_link] = {"link": regatta_link, "scoring": scoring, 'missing': missing, 'date': regatta_date, 'meta': meta}

    # regattas = {'f25/oberg' : {'link':'f25/oberg','scoring':'3 Divisions', 'rescrape' : True, 'date': ''}}

    if len(regattas.values()) > 0:        
        totalRows = asyncio.run(main(regattas))
        totalRows = [sub for row in totalRows for sub in row]
        
        with open("pages/scrape_state.json", "w") as f:
            json.dump(scrape_state, f, indent=2)
            
        df_races = pd.concat([df_races, pd.DataFrame(totalRows)])
        df_races = df_races.drop_duplicates(subset=['raceID', 'Sailor'], keep='last').reset_index(drop=True)
        df_races["Date"] = pd.to_datetime(df_races["Date"], errors="raise")
        df_races["raceNum"] = pd.to_numeric(df_races["raceNum"], errors="raise")
        df_races.to_parquet(outfile, index=False)
        
    else:
        print("no new races to be scraped.")

    end = time.time()
    print(f"{int((end-start) // 60)}:{int((end-start) % 60)}")
    return df_races

if __name__ == "__main__":
    runFleetScrape("racesfrtest.parquet", "racesfrtest.parquet")