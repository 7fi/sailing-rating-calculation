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

async def cleanup_semaphore(semaphore):
    if semaphore.locked():
        # Forcefully release if it was accidentally left locked
        semaphore.release()
        print("Semaphore was released manually.")

async def fetchData(client, semaphore, link):
    retries = 10
    backoff = 1
    broken_links = ['catherine-lindsay', 'francis-humphrey', 'gabriel-gabe-morreale', 'henriette-smith', 'iona-deacon', 'jason-case-pittsburgh', 'jewel-min-min-kelly', 'race-nicolia', 'showtime']
    if link not in broken_links and os.path.exists(f"sailorPages/{link}.html"):
        with open(f"sailorPages/{link}.html", "r") as f:
            sailorPage = BeautifulSoup(f.read(), 'html.parser')
    else:
        for attempt in range(retries):
            try:
                async with semaphore:  # Limit concurrent requests
                    url = f"https://scores.collegesailing.org/sailors/{link}/"
                    page = await client.get(url)
                    with open(f"sailorPages/{link}.html", "w") as f:
                        f.write(str(page.content))
                    sailorPage = BeautifulSoup(page.content, 'html.parser')
                    page.raise_for_status()  # Raise HTTPError for bad responses (4xx, 5xx)

            except httpx.ConnectTimeout as e:
                print(f"Connection timeout when fetching {url}. Retrying... ({attempt + 1}/{retries})")
                await asyncio.sleep(backoff)  # Wait before retrying
                backoff *= 2  # Exponential backoff
            except httpx.ReadError as e:
                print(f"Read error when fetching {url}. Retrying... ({attempt + 1}/{retries})")
                await asyncio.sleep(backoff)  # Wait before retrying
                backoff *= 2  # Exponential backoff
            except httpx.HTTPStatusError as e:
                print(f"HTTP error {e.response.status_code} when fetching {url}. Skipping...")
                await cleanup_semaphore(semaphore)
                return None
            except httpx.RequestError as e:
                print(f"Request error {e}. Skipping...")
                await cleanup_semaphore(semaphore)
                return None
        await cleanup_semaphore(semaphore)
        
    sailorData = sailorPage.find('meta', attrs={'name': 'ts:data'})
    try:
        school = sailorPage.find('ul',id='menu').contents[3].text
    except:
        print(link)
    if sailorData is None: 
        # these goofballs have fucked up pages
        if link == 'henriette-smith':
            sailorData = {"name": 'Henriette Smith', 'year': '2016', 'school': '/schools/oregon/', 'gender': 'F','first_name': '', 'last_name': '', 'id': '', 'external_id': ''}
        elif link == 'iona-deacon':
            sailorData = {"name": 'Iona Deacon', 'year': '2016', 'school': '/schools/british-columbia/', 'gender': 'F', 'first_name': '', 'last_name': '', 'id': '', 'external_id': ''}
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
            return []
    else:
        print("no sailor data for", link)
        return []

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

if __name__ == "__main__":
    df_races = pd.read_json("racesfr.json")
    trPeople = pd.read_json("trSailorInfoAll.json")
    old = pd.read_json("sailor_data2.json")
    # old = old.sample(frac=0.8, random_state=42) # for testing
    
    df_races['Link'] = df_races['Link'].fillna('Unknown') # fill empty links
    links = df_races['Link'].dropna().unique()
    links = np.append(links, trPeople['link'].dropna().unique())
    links = links[links != 'Unknown']
    print(len(links))
    old_links_set = set(old['link'])
    links = [l for l in links if l not in old_links_set]
    print(len(links))
    links = np.unique(links)
  
    totalRows = asyncio.run(main(links))
    df_people = pd.DataFrame(totalRows, columns=['key', 'link', 'name', 'first_name', 'last_name', 'gender', 'year', 'teamLink','team','id', 'external_id'])
    
    # Filter people without a link in df_races
    df_races_no_link = df_races[df_races['Link'] == 'Unknown']

    # Filter people without a link in trPeople
    trPeople_no_link = trPeople[trPeople['link'] == 'Unknown']

    # Combine the data from both sources (people without a link)
    df_no_link = pd.concat([
        df_races_no_link[['Sailor', 'GradYear', 'Team']].rename(columns={'Sailor': 'name', 'GradYear': 'year', 'Team': 'team'}),
        trPeople_no_link[['key', 'name', 'year', 'team']]
    ])

    # Add default values for missing columns to match the structure of df_people
    df_no_link['key'] = df_no_link['key'].fillna(df_no_link['name'] + '-' + df_no_link['team'])
    
    df_no_link['link'] = np.nan
    df_no_link['first_name'] = np.nan
    df_no_link['last_name'] = np.nan
    df_no_link['gender'] = np.nan
    df_no_link['id'] = np.nan
    df_no_link['external_id'] = np.nan

    # Concatenate the two DataFrames (with and without a link)
    df_people_final = pd.concat([old, df_people, df_no_link], ignore_index=True)
    
    df_people_final.to_json("sailor_data2.json", index=False)
