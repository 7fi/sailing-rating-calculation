from Sailors import Sailor
import io
import tempfile

def updateHomepageStats(connection):
    # Update homepage stats
    with connection.cursor() as cursor:
        cursor.execute("""
            UPDATE HomePageStats SET 
                numSailors = (SELECT COUNT(*) FROM Sailors),
                numScores = (SELECT COUNT(*) FROM FleetScores) + (SELECT COUNT(*) FROM TRScores),
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
        'boat','boatName', 'ratingType', 'oldRating', 'newRating', 'regAvg'
    ]
    team_columns = [
        'season', 'regatta', 'raceNumber', 'round', 'sailorID', 'partnerID', 'partnerName',
        'opponentTeam', 'opponentNick', 'score', 'outcome', 'predicted', 'penalty', 'position',
        'date', 'venue', 'boat', 'boatName', 'ratingType', 'oldRating', 'newRating', 'regAvg'
    ]
    
    print(allFrRows.columns)
    for df, table, cols in zip([allFrRows, allTrRows],['FleetScores', 'TRScores'], [fleet_columns, team_columns]):
        upload_df = df[cols]
        
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.csv', delete=True) as temp_file:
            # Ensure you use a tab separator to avoid comma-conflicts in names
            upload_df.to_csv(temp_file.name, index=False, header=False, sep='\t', na_rep='\\N', encoding='utf-8')
            temp_file.flush() # Ensure all data is written to disk

            # 3. The SQL Command
            # Use REPLACE to handle the "Update" logic you had before
            sql = f"""
            LOAD DATA LOCAL INFILE '{temp_file.name}'
            REPLACE INTO TABLE {table}
            FIELDS TERMINATED BY '\t'
            LINES TERMINATED BY '\n'
            ({','.join(cols)})
            """
        
            with connection.cursor() as cursor:
                cursor.execute(sql)
            connection.commit()
    
    # batch_insert("FleetScores", fleet_columns, allFrRows, connection, batch_size)
    # batch_insert("TRScores", team_columns, allTrRows, connection, batch_size)
        
    updateHomepageStats(connection)