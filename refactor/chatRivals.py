import numpy as np
import pandas as pd
from collections import defaultdict
import time

def buildRivals(dfr: pd.DataFrame):

    # --------------------------------------------------
    # 1️⃣  Precompute season column ONCE
    # --------------------------------------------------
    dfr = dfr.copy()
    dfr["season"] = dfr["adjusted_raceID"].str.split("/").str[0]

    # --------------------------------------------------
    # 2️⃣  Convert sailors, seasons, positions to int IDs
    # --------------------------------------------------
    sailor_ids = {k: i for i, k in enumerate(dfr["key"].unique())}
    season_ids = {s: i for i, s in enumerate(dfr["season"].unique())}
    pos_ids = {p: i for i, p in enumerate(dfr["Position"].unique())}

    dfr["sailor_id"] = dfr["key"].map(sailor_ids)
    dfr["season_id"] = dfr["season"].map(season_ids)
    dfr["pos_id"] = dfr["Position"].map(pos_ids)

    # --------------------------------------------------
    # 3️⃣  Extract numpy arrays ONCE
    # --------------------------------------------------
    sailor_arr = dfr["sailor_id"].to_numpy()
    score_arr = dfr["Score"].to_numpy()

    # Store metadata separately (only once per sailor)
    sailor_meta = (
        dfr[["sailor_id", "Sailor", "Team"]]
        .drop_duplicates("sailor_id")
        .set_index("sailor_id")
        .to_dict("index")
    )

    # --------------------------------------------------
    # 4️⃣  Group by race using indices (FAST)
    # --------------------------------------------------
    grouped = dfr.groupby(
        ["adjusted_raceID", "season_id", "pos_id"]
    ).indices

    # Flat rival dictionary
    # key: (a_id, pos_id, b_id, season_id)
    # value: [wins, total]
    rivals = defaultdict(lambda: [0, 0])

    # --------------------------------------------------
    # 5️⃣  Main computation loop
    # --------------------------------------------------
    for (race_id, season_id, pos_id), idx in grouped.items():

        race_sailors = sailor_arr[idx]
        race_scores = score_arr[idx]

        # Sort by score (lower score = better finish)
        order = race_scores.argsort()
        race_sailors = race_sailors[order]

        n = len(race_sailors)

        # Pairwise comparisons
        for i in range(n):
            a = race_sailors[i]

            for j in range(i + 1, n):
                b = race_sailors[j]

                # a beat b
                key_win = (a, pos_id, b, season_id)
                key_loss = (b, pos_id, a, season_id)

                rivals[key_win][0] += 1
                rivals[key_win][1] += 1
                rivals[key_loss][1] += 1

    # --------------------------------------------------
    # 6️⃣  Flatten for export
    # --------------------------------------------------
    rows = []

    for (a, pos_id, b, season_id), (wins, total) in rivals.items():

        rows.append({
            "sailor_id": a,
            "position_id": pos_id,
            "opponent_id": b,
            "season_id": season_id,
            "wins": wins,
            "total": total,
        })

    df_out = pd.DataFrame(rows)
    
    sailor_lookup = {v: k for k, v in sailor_ids.items()}
    df_out["sailor_key"] = df_out["sailor_id"].map(sailor_lookup)
    df_out["opponent_key"] = df_out["opponent_id"].map(sailor_lookup)
    
    sailor_meta = (
    dfr[["key", "sailor_id", "Sailor", "Team"]]
        .drop_duplicates("sailor_id")
        .copy()
    )

    df_out = df_out.merge(
    sailor_meta.rename(columns={
        "sailor_id": "opponent_id",
        "Sailor": "opponent_fullname",
        "Team": "opponent_team",
        "key": "opponent_key_check"
    }),
    on="opponent_id",
    how="left"
)

    season_lookup = {v: k for k, v in season_ids.items()}
    pos_lookup = {v: k for k, v in pos_ids.items()}

    # Convert IDs back to readable labels
    df_out["season"] = df_out["season_id"].map(season_lookup)
    df_out["position"] = df_out["position_id"].map(pos_lookup)

    # Drop internal ID columns
    df_out = df_out.drop(
        columns=[
            "sailor_id",
            "opponent_id",
            "season_id",
            "position_id",
            "opponent_key_check"  # optional
        ]
    )
    
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
    dfr = pd.read_parquet('racesfrtest.parquet')
    start = time.time()
    df_out = buildRivals(dfr)
    print(time.time()-start)
    df_out.to_parquet('rivalstesting.parquet')