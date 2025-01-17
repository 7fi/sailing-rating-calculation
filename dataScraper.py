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
    if os.path.exists(f"sailorPages/{link}.html"):
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
        
    sailorData = sailorPage.find('meta',attrs={'name': 'ts:data'})
    if sailorData is not None:
        sailorData = json.loads(sailorData['content'])
        return [link, sailorData['name'],sailorData['first_name'],sailorData['last_name'],sailorData['gender'],sailorData['year'],sailorData['school'].split("/")[2], sailorData['id'], sailorData['external_id']]
    else:
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
        batchSize = 100
        semaphore = asyncio.Semaphore(batchSize)
        
        for j in range (0, len(links), batchSize):
            print(f"Processing batch {j // batchSize + 1}/{len(links) // batchSize + 1}...")
            batchLinks = list(links)[j:j + batchSize]
            results = await getBatch(client, batchLinks, semaphore)
            allRows.extend(results)
        return allRows

if __name__ == "__main__":
    df_races = pd.read_json("races_new_test.json")
    links = df_races['Link'].dropna().unique()
  
    totalRows = asyncio.run(main(links[70 * 100:]))
    # totalRows = [sub for row in totalRows if row is not None for sub in row]
    df_races_new = pd.DataFrame(totalRows, columns=['link', 'name', 'first_name', 'last_name', 'gender', 'year', 'school','id', 'external_id'])
    df_races_new.to_json("sailor_data.json", index=False)