import numpy as np
import pandas as pd
from collections import defaultdict
import time
from config import Config

def buildRivals(dfr: pd.DataFrame, config: Config):
    print("Calculating Rivals")

    # Split fleet and team races
    dfr["season"] = dfr["adjusted_raceID"].str.split("/").str[0]
    fleet_df = dfr[dfr["Scoring"] != "fleet"].copy()
    team_df = dfr[dfr["Scoring"] == "team"].copy()

    # Build sailor ID map from ALL races
    fleet_keys = set(fleet_df["key"].dropna().unique())

    team_keys = set()
    for _, row in team_df.iterrows():
        for boat in row.get("teamABoats", []):
            if boat.get("skipperKey"):
                team_keys.add(boat["skipperKey"])
            if boat.get("crewKey"):
                team_keys.add(boat["crewKey"])
        for boat in row.get("teamBBoats", []):
            if boat.get("skipperKey"):
                team_keys.add(boat["skipperKey"])
            if boat.get("crewKey"):
                team_keys.add(boat["crewKey"])

    all_keys = fleet_keys.union(team_keys)
    sailor_ids = {k: i for i, k in enumerate(all_keys)}

    # Season + Position IDs (fleet defines universe)
    season_ids = {s: i for i, s in enumerate(dfr["season"].dropna().unique())}
    pos_ids = {p: i for i, p in enumerate(['Skipper', 'Crew'])}

    # Map fleet dataframe to integer IDs
    fleet_df["sailor_id"] = fleet_df["key"].map(sailor_ids)
    fleet_df["season_id"] = fleet_df["season"].map(season_ids)
    fleet_df["pos_id"] = fleet_df["Position"].map(pos_ids)

    sailor_arr = fleet_df["sailor_id"].to_numpy()
    score_arr = fleet_df["Score"].to_numpy()

    # Sailor metadata (for export later)
    sailor_meta = (
        fleet_df[["sailor_id", "Sailor", "Team"]]
        .drop_duplicates("sailor_id")
        .set_index("sailor_id")
        .to_dict("index")
    )

    # Group fleet races (FAST)
    grouped = fleet_df.groupby(
        ["adjusted_raceID", "season_id", "pos_id"]
    ).indices

    rivals = defaultdict(lambda: [0, 0])  # [wins, total]

    # Fleet race processing
    for (race_id, season_id, pos_id), idx in grouped.items():

        race_sailors = sailor_arr[idx]
        race_scores = score_arr[idx]

        order = race_scores.argsort()
        race_sailors = race_sailors[order]

        n = len(race_sailors)

        for i in range(n):
            a = race_sailors[i]

            for j in range(i + 1, n):
                b = race_sailors[j]

                key_win = (a, pos_id, b, season_id)
                key_loss = (b, pos_id, a, season_id)

                rivals[key_win][0] += 1
                rivals[key_win][1] += 1
                rivals[key_loss][1] += 1

    # Team race processing
    for _, row in team_df.iterrows():

        season = row.get("season")
        if season not in season_ids:
            continue

        season_id = season_ids[season]

        teamA_won = row.get("teamAOutcome") == "win"
        teamB_won = row.get("teamBOutcome") == "win"

        if not teamA_won and not teamB_won:
            continue  # skip ties

        for pos in ["Skipper", "Crew"]:

            pos_id = pos_ids[pos]

            teamA_ids = []
            teamB_ids = []

            for boat in row.get("teamABoats", []):
                key = boat.get(pos.lower() + "Key")
                if key in sailor_ids:
                    teamA_ids.append(sailor_ids[key])

            for boat in row.get("teamBBoats", []):
                key = boat.get(pos.lower() + "Key")
                if key in sailor_ids:
                    teamB_ids.append(sailor_ids[key])

            # if either list is empty
            if not teamA_ids or not teamB_ids:
                continue

            if teamA_won:
                winners = teamA_ids
                losers = teamB_ids
            else:
                winners = teamB_ids
                losers = teamA_ids

            for a in winners:
                for b in losers:
                    key_ab = (a, pos_id, b, season_id)
                    key_ba = (b, pos_id, a, season_id)

                    # add win
                    rivals[key_ab][0] += 1
                    
                    # add totals
                    rivals[key_ab][1] += 1
                    rivals[key_ba][1] += 1

    # Flatten for export
    rows = []

    for (a, pos_id, b, season_id), (wins, total) in rivals.items():

        rows.append({
            "sailor_id": a,
            "pos_id": pos_id,
            "opponent_id": b,
            "season_id": season_id,
            "wins": wins,
            "total": total,
        })

    df_out = pd.DataFrame(rows)
    
    # Convert ID → original string keys
    sailor_lookup = {v: k for k, v in sailor_ids.items()}

    df_out["sailor_key"] = df_out["sailor_id"].map(sailor_lookup)
    df_out["opponent_key"] = df_out["opponent_id"].map(sailor_lookup)
    
    df_out["sailor_key"] = df_out["sailor_key"].replace(config.merges)
    df_out["opponent_key"] = df_out["opponent_key"].replace(config.merges)

    # Add opponent metadata (no merge needed)
    df_out["opponent_fullname"] = df_out["opponent_id"].map(
        lambda x: sailor_meta.get(x, {}).get("Sailor")
    )

    df_out["opponent_team"] = df_out["opponent_id"].map(
        lambda x: sailor_meta.get(x, {}).get("Team")
    )

    # Convert season / position IDs back to labels
    season_lookup = {v: k for k, v in season_ids.items()}
    pos_lookup = {v: k for k, v in pos_ids.items()}

    df_out["season"] = df_out["season_id"].map(season_lookup)
    df_out["position"] = df_out["pos_id"].map(pos_lookup)

    # Keep only final columns in correct order
    df_out = df_out[
        [
            "sailor_key",
            "opponent_key",
            "opponent_fullname",
            "opponent_team",
            "position",
            "season",
            "total",
            "wins",
        ]
    ]
    
    return df_out

def uploadRivals(df_rivals, connection, batch_size=10_000):
    print("Uploading Rivals")
    rival_rows = list(df_rivals.itertuples(index=False, name=None))
    
    with connection.cursor() as cursor:
        for i in range(0, len(rival_rows), batch_size):
            batch = rival_rows[i:i + batch_size]
            cursor.executemany("""
                INSERT INTO SailorRivals (
                    sailorID, rivalID, rivalName, rivalTeam, position, season, raceCount, winCount
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    raceCount = VALUES(raceCount),
                    winCount = VALUES(winCount)
            """, batch)
    connection.commit()

if __name__ == "__main__":
    df_races_fr = pd.read_parquet("racesfrtest.parquet")
    df_races_tr = pd.read_parquet("racesTR.parquet")
        
    df_races_full = pd.concat([df_races_fr, df_races_tr])
    
    # clean up memory
    del df_races_fr, df_races_tr
    
    dfr = df_races_full.sort_values(['Date', 'raceNum', 'Div']).reset_index(drop=True)
    
    config = Config()
    
    start = time.time()
    df_out = buildRivals(dfr, config)
    print(time.time()-start)
    
    df_out.to_parquet('rivalstesting.parquet')