from Sailors import Sailor
import csv
import io
import tempfile
import pandas as pd

def updateHomepageStats(connection):
    # Update homepage stats
    with connection.cursor() as cursor:
        cursor.execute("""
            UPDATE HomePageStats SET 
                numSailors = (SELECT COUNT(*) FROM Sailors),
                numScores = (SELECT COUNT(*) FROM FleetScores) + (SELECT COUNT(*) * 6 FROM TRScores),
                numTeams = (SELECT COUNT(*) FROM Teams)
            WHERE id = 1;
        """)
        
    connection.commit()

# UNUSED
def uploadScoresBySailor(people : dict[str,Sailor], connection, batch_size=10000):
    fleet_rows = []
    team_rows = []

    for index, (key, sailor) in enumerate(people.items()):
        if index % 1000 == 0:
            print(f"Processing sailor {sailor.name} {index}/{len(people)}")
        
        races = sailor.races
        
        for race in races:
            raceID_parts = race['raceID'].split("/")
            if race['type'] == 'fleet':
                fleet_rows.append([
                    raceID_parts[0],               # season
                    raceID_parts[1],               # regatta
                    raceID_parts[2][:-1],          # raceNumber
                    raceID_parts[2][-1],           # division
                    key,
                    race['partner']['key'],
                    race['partner']['name'],
                    race['score'],
                    race['predicted'],
                    race['ratio'],
                    '',                             # penalty
                    race['pos'],
                    race['date'],
                    race['scoring'],
                    race['venue'],
                    '',                             # boat
                    race['boatName'],
                    race['ratingType'],
                    race['oldRating'],
                    race['newRating'],
                    race['regAvg']
                ])
            elif race['type'] == 'team':
                team_rows.append([
                    raceID_parts[0],               # season
                    raceID_parts[1],               # regatta
                    raceID_parts[2],               # raceNumber
                    race['round'],
                    key,
                    race['partner']['key'],
                    race['partner']['name'],
                    race['opponentTeam'],
                    race['opponentNick'],
                    race['score'],
                    race['outcome'],
                    race['predicted'],
                    '',                             # penalty
                    race['pos'],
                    race['date'],
                    race['venue'],
                    '',                             # boat
                    '',                             # boatName
                    race['ratingType'],
                    race['oldRating'],
                    race['newRating'],
                    race['regAvg']
                ])
    
    def batch_insert(table_name, columns, data):
        for start in range(0, len(data), batch_size):
            print("Inserting", start, "/", len(data))
            batch = data[start:start + batch_size]
            placeholders = ",".join(["%s"] * len(columns))
            updates = ",".join([f"{col} = VALUES({col})" for col in columns])
            sql = f"""INSERT INTO {table_name} ({','.join(columns)}) VALUES ({placeholders})
                        ON DUPLICATE KEY UPDATE
                            {updates}"""
            with connection.cursor() as cursor:
                cursor.executemany(sql, batch)
            connection.commit()
            
    
    fleet_columns = [
        'season', 'regatta', 'raceNumber', 'division', 'sailorID', 'partnerID', 'partnerName',
        'score', 'predicted', 'ratio', 'penalty', 'position', 'date', 'scoring', 'venue',
        'boat','boatName', 'ratingType', 'oldRating', 'newRating', 'regAvg'
    ]
    team_columns = [
        'season', 'regatta', 'raceNumber', 'round', 'sailorID', 'partnerID', 'partnerName',
        'opponentTeam', 'opponentNick', 'score', 'outcome', 'predicted', 'penalty', 'position',
        'date', 'venue', 'boat', 'boatName', 'ratingType', 'oldRating', 'newRating', 'regAvg'
    ]
    
    print("Inserting FleetScores...")
    batch_insert("FleetScores", fleet_columns, fleet_rows)
    
    print("Inserting TRScores...")
    batch_insert("TRScores", team_columns, team_rows)
    
    updateHomepageStats(connection)
    # cursor.close()
    print("Upload complete.")

def batch_insert(table_name, columns, data, connection, batch_size=10_000):
    for start in range(0, len(data), batch_size):
        print("Inserting", start, "/", len(data))
        batch = data[start:start + batch_size]
        placeholders = ",".join(["%s"] * len(columns))
        updates = ",".join([f"{col} = VALUES({col})" for col in columns])
        sql = f"""INSERT INTO {table_name} 
                  ({','.join(columns)}) 
                  VALUES ({placeholders})
                  ON DUPLICATE KEY UPDATE {updates}
                  WHERE lastUpdated < VALUES(calculatedAt)"""
        print(sql)
        with connection.cursor() as cursor:
            cursor.executemany(sql, batch)
    connection.commit()

def loadInfileSql(path, table, cols):
    return f"""
            LOAD DATA LOCAL INFILE '{path}'
            REPLACE INTO TABLE {table}
            FIELDS TERMINATED BY '\t'
            LINES TERMINATED BY '\n'
            ({','.join(cols)})
            """


def deleteStaleScores(table, keyCols, cols, csvPath, upload_df, connection, batch_size=200):
    """Delete rows of the just-uploaded regattas that this upload no longer accounts for.

    LOAD DATA ... REPLACE only overwrites a row whose key an incoming row collides with.
    When a team fixes the wrong sailors they had entered for a race, the corrected scrape
    contains no row at all for the people who were taken off, so nothing ever collides
    with their rows and they stay in the table for good. Reload the upload into a
    temporary copy of the table and delete whatever the real table holds beyond it.

    Scoped to the (season, regatta) pairs actually in the upload, so a run covering only
    some regattas reconciles just those instead of deleting every regatta it was never
    asked about. The temp table is referenced once per statement - MySQL cannot open a
    temporary table twice in one query.
    """
    temp = f"tmp_{table}"
    keyMatch = " AND ".join(f"s.{c} = t.{c}" for c in keyCols)
    scope = list(upload_df[['season', 'regatta']].drop_duplicates().itertuples(index=False, name=None))

    removed = 0
    with connection.cursor() as cursor:
        cursor.execute(f"DROP TEMPORARY TABLE IF EXISTS {temp}")
        # LIKE carries over the primary key, so the NOT EXISTS probe below is an index lookup.
        cursor.execute(f"CREATE TEMPORARY TABLE {temp} LIKE {table}")
        cursor.execute(loadInfileSql(csvPath, temp, cols))

        for start in range(0, len(scope), batch_size):
            batch = scope[start:start + batch_size]
            pairs = ",".join(["(%s,%s)"] * len(batch))
            cursor.execute(f"""
                DELETE t FROM {table} t
                WHERE (t.season, t.regatta) IN ({pairs})
                  AND NOT EXISTS (SELECT 1 FROM {temp} s WHERE {keyMatch})
            """, [v for pair in batch for v in pair])
            removed += cursor.rowcount

        cursor.execute(f"DROP TEMPORARY TABLE IF EXISTS {temp}")
    connection.commit()
    print(f"Removed {removed} stale rows from {table} across {len(scope)} regattas")


def uploadAllScores(allFrRows, allTrRows, connection, batch_size=10_000):

    fleet_columns = [
        'season', 'regatta', 'raceNumber', 'division', 'sailorID', 'partnerID', 'partnerName',
        'score', 'predicted', 'ratio', 'penalty', 'position', 'date', 'scoring', 'venue',
        'boat','boatName', 'ratingType', 'oldRating', 'newRating', 'regAvg'
    ]
    
    team_columns = [
        'season', 'regatta', 'raceNumber', 'round', 'sailorID', 'partnerID', 'partnerName',
        'opponentTeam', 'opponentNick', 'score', 'outcome', 'predicted', 'penalty', 'position',
        'date', 'venue', 'boat', 'boatName', 'ratingType', 'oldRating', 'newRating', 'regAvg'
    ]
    
    # The primary key of each table, and so what identifies a row when reconciling below.
    # TRScores is left alone for now - fleet racing first.
    fleet_key = ['season', 'regatta', 'raceNumber', 'division', 'sailorID']
    team_key = None

    print(allFrRows.columns)
    for upload_df, table, cols, keyCols in zip([allFrRows, allTrRows],
                                      ['FleetScores', 'TRScores'], [fleet_columns, team_columns],
                                      [fleet_key, team_key]):
        upload_df = upload_df.reindex(columns=cols)
        upload_df['date'] = pd.to_datetime(upload_df['date'], unit='s')

        # na_rep below writes \N, which LOAD DATA only reads as NULL while it is left
        # unescaped. QUOTE_NONE escaping would turn it into the literal string "\N", so
        # refuse rather than quietly writing that.
        nullCols = [c for c in cols if upload_df[c].isna().any()]
        if nullCols:
            raise ValueError(f"Null values in {table} columns {nullCols}; LOAD DATA cannot represent them here")

        with tempfile.NamedTemporaryFile(mode='w+', suffix='.csv', delete=True) as temp_file:
            # Tab separated, and escaped the way LOAD DATA reads it by default: backslash
            # escapes, no quoting. pandas' default QUOTE_MINIMAL disagrees on both counts -
            # it wraps a field containing a quote in quotes that LOAD DATA does not strip
            # (so Tyler "TMAC" Macdonald landed in the DB as "Tyler ""TMAC"" Macdonald"),
            # and it leaves backslashes unescaped for LOAD DATA to swallow (so O\'Connell
            # became O'Connell). QUOTE_NONE with escapechar makes the two sides agree.
            upload_df.to_csv(temp_file.name, index=False, header=False, sep='\t', na_rep='\\N',
                             encoding='utf-8', quoting=csv.QUOTE_NONE, escapechar='\\')
            temp_file.flush() # Ensure all data is written to disk

            # 3. The SQL Command
            # Use REPLACE to handle the "Update" logic you had before
            with connection.cursor() as cursor:
                cursor.execute(loadInfileSql(temp_file.name, table, cols))
            connection.commit()

            if keyCols:
                deleteStaleScores(table, keyCols, cols, temp_file.name, upload_df, connection)

    # batch_insert("FleetScores", fleet_columns, allFrRows, connection, batch_size)
    # batch_insert("TRScores", team_columns, allTrRows, connection, batch_size)