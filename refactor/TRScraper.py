from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np
from datetime import date, datetime
import os

def setup():
  # seasons = ['f24', 's24', 'f23', 's23', 'f22','s22']
  # seasons = ['f24', 's24', 'f23', 's23', 'f22','s22']
  seasons = [[f"f{i}",f"s{i}"] for i in range (16,25)]
  seasons = [sub for s in seasons for sub in s] + ['s25']
  # seasons = ['s24']

  df_old = pd.DataFrame()
  try:
    df_old = pd.read_json("racesTR.json")
  except:
      df_old = pd.DataFrame(columns=['raceID','regatta','raceNum', 'round','date','allSkipperKeys','allCrewKeys','teamAName','teamAUni','teamANick','teamALink','teamAID','teamABoats','teamAScore','teamAOutcome','teamBName','teamBUni','teamBNick','teamBLink','teamBID','teamBBoats','teamBScore','teamBOutcome'])

  regattas = {}

  for season in seasons:
    url = f"https://scores.collegesailing.org/{season}/"
    page = requests.get(url)
    listSoup = BeautifulSoup(page.content, 'html.parser')
    
    tbody = listSoup.find('table', class_="season-summary").find('tbody')
    
    for link in tbody.find_all("a", href=True):
      scoring = link.parent.next_sibling.next_sibling.next_sibling.text
      regatta_date = link.parent.next_sibling.next_sibling.next_sibling.next_sibling.text
      regatta_status = link.parent.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling.text
      
      rescrape = regatta_status != 'Official'
      if (datetime.today() - datetime.strptime(regatta_date, "%m/%d/%Y")).days > 14:
          rescrape = False
      if rescrape:
          print(link['href'], regatta_date)
          
      scrape = (season + "/" + link['href']) not in df_old['regatta'].unique() or rescrape
      if (scoring == "Team") and scrape:
          regattas[season + "/" + link['href']] = {"link":season + "/" + link['href'], "scoring": scoring, 'rescrape': rescrape}
  print(len(regattas))

  regattaSoups = {}

  for i, regatta in enumerate(list(regattas.values())):
      link = regatta['link']
      rescrape = regatta['rescrape']
      
      allRaces = sailors = report = None
      regattaID = list(regattas.keys())[i]
      
      if not os.path.exists(f"pagesTR/{link.split("/")[0]}"):
          os.makedirs(f"pagesTR/{link.split("/")[0]}")
      if os.path.exists(f"pagesTR/{link}-allraces.html") and os.path.exists(f"pagesTR/{link}-sailors.html") and os.path.exists(f"pagesTR/{link}-report.html") and not rescrape:
          if i % 50 == 0:
              print(f"({i + 1}/{len(list(regattas.values()))}) getting soup from file for {regattaID}")
          with open(f"pagesTR/{link}-allraces.html", "r") as f:
              allRaces = BeautifulSoup(f.read(), 'html.parser')
          with open(f"pagesTR/{link}-sailors.html", "r") as f:
              sailors = BeautifulSoup(f.read(), 'html.parser')
          with open(f"pagesTR/{link}-report.html", "r") as f:
              report = BeautifulSoup(f.read(), 'html.parser')
      else:
          print(f"({i + 1}/{len(list(regattas.values()))}) getting soup for {regattaID}")
          # all races
          url = f"https://scores.collegesailing.org/{regatta['link']}/all/"
          page = requests.get(url)
          with open(f"pagesTR/{link}-allraces.html", "w") as f:
                          f.write(str(page.content))
          allRaces = BeautifulSoup(page.content, 'html.parser')

          # sailors
          url = f"https://scores.collegesailing.org/{regatta['link']}/sailors/"
          page = requests.get(url)
          with open(f"pagesTR/{link}-sailors.html", "w") as f:
                          f.write(str(page.content))
          sailors = BeautifulSoup(page.content, 'html.parser')
          
          url = f"https://scores.collegesailing.org/{regatta['link']}/"
          page = requests.get(url)
          with open(f"pagesTR/{link}-report.html", "w") as f:
                          f.write(str(page.content))
          report = BeautifulSoup(page.content, 'html.parser')
      
      regattaSoups[regattaID] = {"allRaces": allRaces, "sailors": sailors, 'report': report, "scoring": regatta['scoring']}
      
  return regattaSoups, df_old, regattas
    
    
def makeRaceList(raceRows):
  raceList = []
  round = ''
  
  # loop through races
  for raceRow in raceRows: 
      if raceRow.has_attr('class') and raceRow.attrs['class'][0] == 'roundrow':
          round = raceRow.text
          continue
      
      raceData = [el.text for el in raceRow.find_all('td')]
      raceNum = raceData[0]
      
      teamAEl = raceRow.find_all('td', class_='team1')
      if(len(teamAEl) < 3):
          print("No score for this race?", raceNum)
          continue
      teamAName = list(teamAEl[1].stripped_strings)[0]
      teamANick = list(teamAEl[1].stripped_strings)[1]
      teamALink = teamAEl[1].find('a').attrs['href']
      teamAScore = teamAEl[2].text
      teamAWin = teamAEl[1].attrs['class'][0].split("-")[1]
      teamA = {'name': teamAName, 'nick': teamANick, 'link': teamALink.split("/")[2], 'outcome': teamAWin, 'score': teamAScore}
          
      teamBEl = raceRow.find_all('td', class_='team2')
      teamBName = list(teamBEl[1].stripped_strings)[0]
      teamBNick = list(teamBEl[1].stripped_strings)[1]
      teamBLink = teamBEl[1].find('a').attrs['href']
      teamBScore = teamBEl[0].text
      teamBWin = teamBEl[1].attrs['class'][0].split("-")[1]
      teamB = {'name': teamBName, 'nick': teamBNick, 'link': teamBLink.split("/")[2], 'outcome': teamBWin, 'score': teamBScore}
      # print(raceNum, winningTeam, winningScore, losingTeam, losingScore)
      
      raceList.append({'raceNum': int(raceNum),
                        'round': round,
                        'teamA': teamA,
                        'teamB': teamB,
                        })
  return raceList
  
  
def getSailorLinks(reportPage):
  sailorLinks = []
  
  mainTable = reportPage.find('table', class_='results').contents[1]
  allSailors = mainTable.find_all('td', class_='sailor')
  
  for sailor in allSailors:
      sailorName = sailor.text.split(" '")[0].strip()
      sailorLink = 'Unknown'
      if sailor.find('a') != None:
          sailorLink = sailor.find('a').attrs['href'].split("/")[2]
      if sailorName not in [p['name'] for p in sailorLinks]:
          sailorLinks.append({'name': sailorName, 'link':sailorLink})
  return sailorLinks

def getTeamReportInfo(reportPage):
  teamInfoList = []
  
  mainTable = reportPage.find('table', class_='results').contents[1]
  teamRows = mainTable.find_all('tr', class_='topborder')
  
  for row in teamRows:
      teamRank = row.contents[1].text
      teamID = row.attrs['class'][2].split("-")[1]
      teamNick = row.find('td', class_='teamname').text
      uniName = row.find('a').text
      teamLink = row.find('a').attrs['href']
      tiebreaker = {'char': row.find('td', class_='tiebreaker').text,'text':row.find('td', class_='tiebreaker').attrs['title']}
      
      teamInfoList.append({'rank': teamRank, 'uniName': uniName, 'teamNick': teamNick, 'teamID': teamID, 'teamLink': teamLink, 'tiebreaker': tiebreaker})
  return teamInfoList
  
  
def makeSailorList(sailorData, regatta):
  sailorRaceList = []
  
  ports = sailorData.find_all('div', class_='port')
  
  round = ''
  for port in ports:
      round = port.find('h3').text
      numTables = len(port.find_all('table'))
      if numTables == 0:
          return [{'name': 'no sailors!'}]
      tableRows = port.find('table').contents[0].contents
  
      opponentsRow = tableRows[0].find_all('th')
      teamNames = [team.text for team in opponentsRow]
      
      fullTeamNames = [el.find_all('th')[0].text for el in tableRows[1:]]
      teamIDs = [el.attrs['class'][1].split("-")[1] for el in tableRows[1:]]
      
      for tableRow, teamName, fullTeamName, teamID in zip(tableRows[1:], teamNames, fullTeamNames, teamIDs):
          sailorTables = tableRow.find_all("table", class_='tr-boats')
          tempNames = teamNames.copy()
          tempNames.remove(teamName)
          tempFullNames = fullTeamNames.copy()
          tempFullNames.remove(fullTeamName)
          
          tempTeamIDs = teamIDs.copy()
          tempTeamIDs.remove(teamID)
          
          for race, opponent, opponentFull, oppID in zip(sailorTables, tempNames, tempFullNames, tempTeamIDs):
              boats = [list(names.stripped_strings) for names in race.find_all('td')][:3]
              for boat in boats:
                  skipperName, skipperYear, crewName, crewYear = ['Unknown'] * 4
                  
                  if len(boat) == 2:
                      if boat[0] != 'No show':
                          skipperName = boat[0].split(" '")[0].strip()
                          skipperYear = boat[0].split(" '")[1].strip()
                      if boat[1] != 'No show':
                          crewName = boat[1].split(" '")[0].strip()
                          crewYear = boat[1].split(" '")[1].strip()
                      
                  sailorRaceList.append({'name': skipperName,
                                          'year': skipperYear,
                                          'pos': 'skipper',
                                          'round': round,
                                          'teamName': teamName, 
                                          'teamID': teamID,
                                          'fullName': fullTeamName, 
                                          'opponent': opponent,
                                          'opponentFull': opponentFull,
                                          'oppID': oppID,
                                          'partner': crewName,
                                          'regatta': regatta
                                          })
                  
                  sailorRaceList.append({'name': crewName,
                                          'year': crewYear,
                                          'pos': 'crew',
                                          'round': round,
                                          'teamName': teamName, 
                                          'teamID': teamID,
                                          'fullName': fullTeamName, 
                                          'opponent': opponent,
                                          'opponentFull': opponentFull,
                                          'oppID': oppID,
                                          'partner': skipperName,
                                          'regatta': regatta
                                          })
                    
  return sailorRaceList

def getTeamIDs(sailorData):
    data = []
    ports = sailorData.find_all('div', class_='port')
    
    for port in ports:
        tableRows = port.find('table').contents[0].contents
        opponentsRow = tableRows[0].find_all('th')
        
        for team, el in zip(opponentsRow, tableRows[1:]):
            teamName = team.text
            uniName = el.find('th').text
            teamID = el.attrs['class'][1].split("-")[1]
            if teamID not in [d['teamID'] for d in data]:
                data.append({'teamName': teamName, 'uniName': uniName, 'teamID': teamID})
    
    return data
  
def getData(regattaSoups, regattas):
  data = []
  totalSailors = {}
  df_totalSailors = pd.DataFrame()

  for i, regatta in enumerate(list(regattaSoups.keys())):
  # regatta = list(regattaSoups.keys())[0]
  # regatta = 's25/mass-maritime-team-race-invite'
      print(f"({i + 1}/{len(list(regattas.values()))}) analyzing {regatta}")

      allRaces = regattaSoups[regatta]['allRaces']
      sailors = regattaSoups[regatta]['sailors']
      reportPage = regattaSoups[regatta]['report']
      scoring = regattaSoups[regatta]['scoring']

      if len(allRaces.find_all('table', class_="teamscorelist")) == 0: 
          print(f"no scores entered for {regatta}, skipping")
          continue

      scoreData = allRaces.find_all('table', class_="teamscorelist")[
          0].contents

      host = allRaces.find("span", itemprop='location').text
      date = allRaces.find("time").attrs['datetime']
      date = date[:10]

      regattaType = allRaces.find("span", itemprop='description').text
      
      host = allRaces.find("span", itemprop='location').text

      raceRows = [row for table in scoreData if table for row in table.contents][1:]

      df_races = pd.DataFrame(makeRaceList(raceRows))

      df_sailors = pd.DataFrame(makeSailorList(sailors, regatta))
      if len(df_sailors['name'].unique()) < 2:
          print("No rp ented for this regatta", regatta)
          continue

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
          
          data.append({'raceID': f"{regatta}/{raceNum}",
                      'regatta': regatta,
                      'raceNum': raceNum, 'round': round,
                      'date': date,
                      'Venue': host,
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
    
def scrapeTR():    
  print("Scraping TR")
  regattaSoups, df_old, regattas = setup()
  data, totalSailors = getData(regattaSoups, regattas)
  
  df_cur = pd.DataFrame(data)
  df_combined = pd.concat([df_old, df_cur])

  df_final = df_combined.drop_duplicates(subset='raceID', keep='last').reset_index(drop=True)

  df_totalSailors2 = pd.DataFrame.from_dict(totalSailors,orient='index')
  df_totalSailors2 = df_totalSailors2.reset_index(drop=True)
  df_totalSailors2.to_json("trSailorInfoAll.json", index=False)

  from datetime import date
  df_final.to_json(f"TR-{date.today().strftime("%Y%m%d")}.json", index=False,date_format='iso')
  df_final.to_json(f"racesTR.json", index=False,date_format='iso')
  
  return df_final, df_totalSailors2