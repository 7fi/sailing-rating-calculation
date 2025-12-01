from Sailors import Sailor
def uploadScoresBySailor(people : dict[str,Sailor], cursor, connection, batch_size=3000):
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
    
    # Function to insert in batches
    def batch_insert(cursor, table_name, columns, data):
        for start in range(0, len(data), batch_size):
            print("Inserting", start, "/", len(data))
            batch = data[start:start + batch_size]
            placeholders = ",".join(["%s"] * len(columns))
            sql = f"INSERT IGNORE INTO {table_name} ({','.join(columns)}) VALUES ({placeholders})"
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
    batch_insert(cursor, "FleetScores", fleet_columns, fleet_rows)
    
    # print("Inserting TRScores...")
    # batch_insert(cursor, "TRScores", team_columns, team_rows)
    
    # Update homepage stats
    cursor.execute("""
        UPDATE HomePageStats SET 
            numSailors = (SELECT COUNT(*) FROM Sailors),
            numScores = (SELECT COUNT(*) FROM FleetScores) + (SELECT COUNT(*) FROM TRScores),
            numTeams = (SELECT COUNT(*) FROM Teams)
        WHERE id = 1;
    """)
    
    connection.commit()
    # cursor.close()
    print("Upload complete.")
