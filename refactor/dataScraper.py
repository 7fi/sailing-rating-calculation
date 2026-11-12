# %%
from bs4 import BeautifulSoup
import requests
import json
import httpx
import asyncio
import pandas as pd
import numpy as np
import time
import os
from concurrent.futures import ProcessPoolExecutor
from pageCache import readPage, writePage

def blankRow(link):
    """A placeholder for a sailor whose page could not be read.

    The sailor still needs a row, or the races they are in cannot be calculated. A blank
    name marks it, and runSailorData refetches blank rows on the next run.
    """
    return [link, link, "", "", "", "", "", "", "", "", ""]

def isRetryable(e):
    """Timeouts, dropped connections, rate limiting and server errors are worth another try."""
    if isinstance(e, (httpx.TimeoutException, httpx.ReadError, httpx.RemoteProtocolError)):
        return True
    return isinstance(e, httpx.HTTPStatusError) and (e.response.status_code == 429 or e.response.status_code >= 500)

async def fetchData(client, semaphore, link):
    retries = 10
    backoff = 1
    broken_links = ['catherine-lindsay', 'francis-humphrey', 'gabriel-gabe-morreale', 'henriette-smith', 'iona-deacon', 'jason-case-pittsburgh', 'jewel-min-min-kelly', 'race-nicolia', 'showtime', 'gabriel-gabe-morreale-2028']
    sailorPage = None
    if link not in broken_links and os.path.exists(f"sailorPages/{link}.html"):
        sailorPage = BeautifulSoup(readPage(f"sailorPages/{link}.html"), 'html.parser')
        # Error pages used to be cached too - a sailor linked from a regatta before their
        # profile page exists gets a 404 - and reading one back stored a blank sailor for
        # good. A cached page without the sailor data is refetched instead of trusted.
        if sailorPage.find('meta', attrs={'name': 'ts:data'}) is None:
            sailorPage = None

    if sailorPage is None:
        url = f"https://scores.collegesailing.org/sailors/{link}/"
        for attempt in range(retries):
            try:
                async with semaphore:  # Limit concurrent requests
                    page = await client.get(url)
                page.raise_for_status()  # Raise HTTPError for bad responses (4xx, 5xx)
                # Only cache a page that was actually served, never an error page.
                writePage(f"sailorPages/{link}.html", page)
                sailorPage = BeautifulSoup(page.text, 'html.parser')
                break
            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                if not isRetryable(e) or attempt == retries - 1:
                    print(f"{type(e).__name__} when fetching {url}: {e}. Skipping...")
                    return blankRow(link)
                print(f"{type(e).__name__} when fetching {url}. Retrying... ({attempt + 1}/{retries})")
                await asyncio.sleep(backoff)  # Wait before retrying, outside the semaphore
                backoff *= 2  # Exponential backoff

    sailorData = sailorPage.find('meta', attrs={'name': 'ts:data'})
    school = ""
    try:
        school = sailorPage.find('ul',id='menu').contents[3].text
        # print(sailorPage.find('ul',id='menu').contents)
        # print(school)
    except:
        print(link)
    if sailorData is None: 
        # these goofballs have fucked up pages
        if link == 'henriette-smith':
            sailorData = {"name": 'Henriette Smith', 'year': '2016', 'school': '/schools/oregon/', 'gender': 'F','first_name': '', 'last_name': '', 'id': '', 'external_id': ''}
        elif link == 'iona-deacon':
            sailorData = {"name": 'Iona Deacon', 'year': '2016', 'school': '/schools/british-columbia/', 'gender': 'F', 'first_name': '', 'last_name': '', 'id': '', 'external_id': ''}
        elif link == 'andrew-smith-2028':
            sailorData = {"name": 'Andrew Smith', 'year': '2028', 'school': '/schools/michigan-state/', 'gender': 'M', 'first_name': '', 'last_name': '', 'id': '', 'external_id': ''}
        elif link == 'patrick-freese': 
            sailorData = {"name": 'Patrick Freese', 'year': '2016', 'school': '/schools/marquette/', 'gender': "M", 'first_name': '', 'last_name': '', 'id': '', 'external_id': ''}
        elif link == 'giuditta-di-laghi': 
            sailorData = {"name": 'Giuditta Di Laghi', 'year': '2016', 'school': '/schools/hampton/', 'gender': "F", 'first_name': '', 'last_name': '', 'id': '', 'external_id': ''}
        elif link == 'erin-kemp': 
            sailorData = {"name": 'Erin Kemp', 'year': '2016', 'school': '/schools/maryland/', 'gender': "F", 'first_name': '', 'last_name': '', 'id': '', 'external_id': ''}
        # elif link == 'catherine-lindsay':
        #     sailorData = {'gender': "F", 'name': 'Catherine "B" Lindsay', 'first_name':'Catherine "B"', 'last_name':'Lindsay', 'year': '2023', 'school': 'url:/schools/yale/', 'id': '3124222', 'external_id': None}
            
    else:
        try:
            sailorData = json.loads(sailorData['content'])
        except:
            print("json error for", link, sailorData['content'])
            
        
    if sailorData is not None:
        try:
            return [link, link, sailorData['name'], sailorData['first_name'], sailorData['last_name'],sailorData['gender'],sailorData['year'], sailorData['school'].split("/")[2][:-1], school, sailorData['id'], sailorData['external_id']]
        except:
            print("no sailor data forr", link)
            return blankRow(link)
    else:
        print("no sailor data for", link)
        return blankRow(link)

async def getBatch(client, links, semaphore):
    tasks = []
    for link in links:
        tasks.append(fetchData(client, semaphore, link))
    results = await asyncio.gather(*tasks)
    # tasks = [process_in_process(executor, regatta) for regatta in results]
    # rows = await asyncio.gather(*tasks)
    return results

async def main(links):
  async with httpx.AsyncClient(timeout=httpx.Timeout(None,connect=15.0, read=10.0)) as client:
        allRows = []
        batchSize = 200
        semaphore = asyncio.Semaphore(batchSize)
        
        for j in range (0, len(links), batchSize):
            print(f"Processing batch {j // batchSize + 1}/{len(links) // batchSize + 1}...")
            batchLinks = list(links)[j:j + batchSize]
            results = await getBatch(client, batchLinks, semaphore)
            allRows.extend(results)
        return allRows
# %%
def runSailorData(frfile, trfile, oldDataFile, outfile):
    # %%
    # rootDir = "./../"
    # frfile, trfile, oldDataFile, outfile = rootDir + "racesfrtest.parquet", rootDir + "trSailorInfoAll.json", rootDir + "sailor_data2.parquet", rootDir + "sailor_data2.parquet"
    print("----- Scraping sailor data ------")
    df_races = pd.read_parquet(frfile)
    trPeople = pd.read_json(trfile)
    
    try:
        old = pd.read_parquet(oldDataFile)
    except FileNotFoundError:
        old = pd.DataFrame()

    if len(old) > 0:
        # Blank rows are sailors whose page could not be read last time. Leaving them out
        # of old means their links count as new below, so they are fetched again.
        retry = old['link'].notna() & (old['name'] == '')
        if retry.any():
            print(f"Refetching {int(retry.sum())} sailors whose pages could not be read before")
        old = old.loc[~retry]

    # %%
    df_races['Link'] = df_races['Link'].fillna('Unknown') # fill empty links
    links = df_races['Link'].dropna().unique()
    if 'link' in trPeople.columns:
        links = np.append(links, trPeople['link'].dropna().unique())
    links = links[links != 'Unknown']
    print(len(links))
    if len(old) > 0:
        old_links_set = set(old['link'])
        links = [l for l in links if l not in old_links_set]
    print(len(links))
    links = np.unique(links)
    
    # %%
    totalRows = asyncio.run(main(links))
    # print(totalRows.shape())
    # if sum([len(x) for x in totalRows]) == 0:
    #     old.to_parquet(outfile, index=False)
    #     return old
    
    df_people = pd.DataFrame(totalRows, columns=['key', 'link', 'name', 'first_name', 'last_name', 'gender', 'year', 'teamLink','team','id', 'external_id'])
    
    # %%
    # Filter people without a link in df_races
    df_races_no_link = df_races[df_races['Link'] == 'Unknown']

    # Filter people without a link in trPeople
    if 'link' in trPeople.columns:
        trPeople_no_link = trPeople[trPeople['link'] == 'Unknown']

        # Combine the data from both sources (people without a link)
        df_no_link = pd.concat([
            df_races_no_link[['Sailor', 'GradYear', 'Team']].rename(columns={'Sailor': 'name', 'GradYear': 'year', 'Team': 'team'}),
            trPeople_no_link[['key', 'name', 'year', 'team']]
        ])
    else:
        df_no_link = df_races_no_link[['Sailor', 'GradYear', 'Team']].rename(columns={'Sailor': 'name', 'GradYear': 'year', 'Team': 'team'})
        df_no_link['key'] = pd.Series()

    # Add default values for missing columns to match the structure of df_people
    df_no_link['key'] = df_no_link['key'].fillna(df_no_link['name'] + '-' + df_no_link['team'])
    
    df_no_link['link'] = np.nan
    df_no_link['first_name'] = np.nan
    df_no_link['last_name'] = np.nan
    df_no_link['gender'] = np.nan
    df_no_link['id'] = np.nan
    df_no_link['external_id'] = np.nan

    # Concatenate the two DataFrames (with and without a link)
    if len(old) > 0:
        df_people = pd.concat([df_people, old], ignore_index=True)
    if len(df_no_link) > 0: 
        df_people = pd.concat([df_people, df_no_link], ignore_index=True)
    
    df_people['id'] = pd.to_numeric(df_people['id'], errors='raise')

    # Sailors without a link are rebuilt from the races every run and old already holds
    # them, so without this the file gained a full copy of them each time. setupPeople
    # keeps the last row for a key, so keeping the last one here changes nothing there.
    df_people = df_people.drop_duplicates(subset='key', keep='last').reset_index(drop=True)

    # %%
    print("outputting to parquet")
    df_people.to_parquet(outfile, index=False)
    return df_people

if __name__ == "__main__":
    runSailorData("racesfr.parquet", "trSailorInfoAll.json", "sailor_data2.parquet", "sailor_data2.parquet")
    # runSailorData("racesfrtest.json", "trSailorInfoAll.json", "sailor_data2.json", "sailor_data2.parquet")
