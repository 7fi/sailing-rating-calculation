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

def uploadAllScores(allFrRows, allTrRows, connection, batch_size=10_000):
    
    fleet_columns = [
        'season', 'regatta', 'raceNumber', 'division', 'sailorID', 'partnerID', 'partnerName',
        'score', 'predicted', 'ratio', 'penalty', 'position', 'date', 'scoring', 'venue',
        'boat','boatName', 'ratingType', 'oldRating', 'newRating', 'regAvg', 'credit'
    ]
    
    team_columns = [
        'season', 'regatta', 'raceNumber', 'round', 'sailorID', 'partnerID', 'partnerName',
        'opponentTeam', 'opponentNick', 'score', 'outcome', 'predicted', 'penalty', 'position',
        'date', 'venue', 'boat', 'boatName', 'ratingType', 'oldRating', 'newRating', 'regAvg'
    ]
    
    print(allFrRows.columns)
    for upload_df, table, cols in zip([allFrRows, allTrRows],
                                      ['FleetScores', 'TRScores'], [fleet_columns, team_columns]):
        upload_df = upload_df.reindex(columns=cols)
        upload_df['date'] = pd.to_datetime(upload_df['date'], unit='s')

        # NULLs cannot be written as \N here: QUOTE_NONE with escapechar='\\' escapes
        # the backslash, so \N reaches the file as \\N and LOAD DATA reads it as the
        # literal two-character string, which a FLOAT column rejects with
        # "Incorrect FLOAT value: '\\N'". Nullable columns are therefore written as the
        # empty string (which needs no escaping) and mapped back to NULL through a user
        # variable and NULLIF in the LOAD DATA statement below.
        #
        # `credit` is legitimately absent - team races have no joint fit, and neither
        # does a fleet row the fit skipped. Every other column must be populated, since
        # an empty string there would be silently coerced rather than rejected.
        nullable = [c for c in ('credit',) if c in cols]
        nullCols = [c for c in cols if c not in nullable and upload_df[c].isna().any()]
        if nullCols:
            raise ValueError(f"Null values in {table} columns {nullCols}; LOAD DATA cannot represent them here")

        with tempfile.NamedTemporaryFile(mode='w+', suffix='.csv', delete=True) as temp_file:
            # Tab separated, and escaped the way LOAD DATA reads it by default: backslash
            # escapes, no quoting. pandas' default QUOTE_MINIMAL disagrees on both counts -
            # it wraps a field containing a quote in quotes that LOAD DATA does not strip
            # (so Tyler "TMAC" Macdonald landed in the DB as "Tyler ""TMAC"" Macdonald"),
            # and it leaves backslashes unescaped for LOAD DATA to swallow (so O\'Connell
            # became O'Connell). QUOTE_NONE with escapechar makes the two sides agree.
            upload_df.to_csv(temp_file.name, index=False, header=False, sep='\t', na_rep='',
                             encoding='utf-8', quoting=csv.QUOTE_NONE, escapechar='\\')
            temp_file.flush() # Ensure all data is written to disk

            # Nullable columns are read into a user variable so an empty field becomes
            # a real NULL instead of being coerced to 0.
            loadCols = [f'@{c}' if c in nullable else c for c in cols]
            setClause = ', '.join(f'{c} = NULLIF(@{c}, \'\')' for c in nullable)

            # 3. The SQL Command
            # Use REPLACE to handle the "Update" logic you had before
            sql = f"""
            LOAD DATA LOCAL INFILE '{temp_file.name}'
            REPLACE INTO TABLE {table}
            FIELDS TERMINATED BY '\t'
            LINES TERMINATED BY '\n'
            ({','.join(loadCols)})
            {'SET ' + setClause if setClause else ''}
            """
        
            with connection.cursor() as cursor:
                cursor.execute(sql)
            connection.commit()
    
    # batch_insert("FleetScores", fleet_columns, allFrRows, connection, batch_size)
    # batch_insert("TRScores", team_columns, allTrRows, connection, batch_size)