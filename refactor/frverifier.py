import mysql.connector
from dotenv import load_dotenv
import os
import polars as pl

load_dotenv()
connection = mysql.connector.connect(
        host=os.getenv('DB_HOST'),
        port=os.getenv('DB_PORT'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASS'),
        database=os.getenv('DB_NAME')
    )

df_races =  pl.read_parquet("racesfrtest.parquet")
df_races = df_races.rename({"key": "sailorID"})
df_races = df_races.filter(pl.col("Score").is_not_null())

race_counts = df_races.group_by("raceID").agg(pl.count().alias("count"))
valid_race_ids = race_counts.filter(pl.col("count") > 6).select("raceID")
df_races = df_races.join(valid_race_ids, on="raceID", how="inner")

with connection.cursor() as cursor:
    cursor.execute("SELECT season, regatta, raceNumber, division, sailorID FROM FleetScores")
    rows = [(f"{row[0]}/{row[1]}/{row[2]}{row[3]}", row[4]) for row in cursor.fetchall()]
    print(len(rows), len(df_races), len(df_races) - len(rows))
    
    df_db = pl.DataFrame(rows, schema=["raceID", "sailorID"], orient='row')
    df_filtered = df_races.join(df_db, on=["raceID", "sailorID"], how='anti')

    print(df_filtered)
    print(df_filtered[5, 'raceID'], df_filtered[5, 'sailorID'])

connection.close()