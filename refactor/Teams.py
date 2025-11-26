from fullfile import validPerson
from config import Config
from regions import teamRegions
import datetime
import pandas as pd
import mysql


def validPerson(p, type, config: Config):
    return (p.cross > 20
            and p.outLinks > 70
            and not p.hasconfig.TargetSeasons(config.config.targetSeasons, type)
            # and (2000 + int(p.year.split()[0]) > gradCutoff if isinstance(p.year, str) and len(p.year.split()) > 1 else int(p.year) > gradCutoff)
            # and sum([p['raceCount'][seas] for seas in config.targetSeasons if seas in p['raceCount'].keys()]) > 5
            )

def uploadTeams(df_sailors, df_races_full, people, cursor, connection, config: Config):

    def getCounts(races):
        # season_counts = defaultdict(int)
        season_counts = {}

        for race in races:
            season = race["raceID"].split("/")[0]
            if season not in season_counts.keys():
                season_counts[season] = {}
            if race['pos'] not in season_counts[season].keys():
                season_counts[season][race['pos']] = 0
            season_counts[season][race['pos']] += 1

        return dict(season_counts)

    # batch = db.batch()
    # col = db.collection('eloTeams')
    teams = []
    predteams = []
    scrape = False
    teamNames = teamRegions.keys()
    lenteams = len(teamNames)

    season_mask = df_sailors['Seasons'].apply(lambda x: not set(x['skipper']).isdisjoint(
        config.config.targetSeasons) or not set(x['crew']).isdisjoint(config.config.targetSeasons))

    # Explode the Teams column to enable grouping
    df_exploded = df_sailors.loc[season_mask].explode('Teams')

    # Group by team and compute necessary aggregates
    team_stats = df_exploded.groupby('Teams').agg(
        numCurMembers=('Teams', 'count'),
        avgSkipperOrdinal=('srOrd', 'mean'),
        avgCrewOrdinal=('crOrd', 'mean'),
        avgSkipperRatio=('skipperAvgRatio', 'mean'),
        avgCrewRatio=('crewAvgRatio', 'mean')
    )

    # Calculate the average values as in the original code
    team_stats['avg'] = (team_stats['avgSkipperOrdinal'] +
                         team_stats['avgCrewOrdinal']) / 2
    team_stats['avgRatio'] = (
        team_stats['avgSkipperRatio'] + team_stats['avgCrewRatio']) / 2

    team_stats = team_stats.reindex(teamNames, fill_value=0)
    today = datetime.today()

    # team_link_map = df_races_full.dropna().drop_duplicates('Team').set_index('Team')['Teamlink'].to_dict()

    df_cleaned = df_races_full.dropna(
        subset=['Team', 'Teamlink']).drop_duplicates(subset='Team', keep='first')

    # Create a dictionary with 'Team' as the key and 'TeamLink' as the value
    team_link_map = pd.Series(
        df_cleaned.Teamlink.values, index=df_cleaned.Team).to_dict()


    for i, (team, row) in enumerate(team_stats.iterrows()):
        # if team != "MIT":
        #     continue
        print(f"{i}/{len(team_stats)} {team}")
        avg = row['avg']
        avgRatio = row['avgRatio']
        numCurMembers = row['numCurMembers']

        region = teamRegions[team]
        # teamLink = df_races.loc[df_races['Team'] == team, 'Teamlink'].iloc[0]
        # Default to '' if team not found
        teamLink = team_link_map.get(team, '')
        url = f"https://scores.collegesailing.org/schools/{teamLink}"

        filtered_people = [p for p in people.values() if team in p.teams]

        members = [{"name": str(p.name),
                    "key": p.key,
                    "gender": p.gender,
                    "year": str(p.year),
                    'teams': list(p.teams),
                    'skipperRating': int(p.sr.ordinal(target=config.targetElo, alpha=config.alpha)),
                    # 'sr': {'mu': p.sr.mu, 'sigma': p.sr.sigma},
                    'crewRating': int(p.cr.ordinal(target=config.targetElo, alpha=config.alpha)),
                    'womenSkipperRating': int(p.wsr.ordinal(target=config.targetElo, alpha=config.alpha)),
                    'womenCrewRating': int(p.wcr.ordinal(target=config.targetElo, alpha=config.alpha)),
                    'tsr': p.tsr.ordinal(target=config.targetElo, alpha=config.alpha),
                    'wtsr': p.wtsr.ordinal(target=config.targetElo, alpha=config.alpha),
                    'tcr': p.tcr.ordinal(target=config.targetElo, alpha=config.alpha),
                    'wtcr': p.wtcr.ordinal(target=config.targetElo, alpha=config.alpha),
                    'avgSkipperRatio': float(p.avgSkipperRatio),
                    'avgCrewRatio': float(p.avgCrewRatio),
                    # season -> position -> count (normalized position names)
                    'raceCount': (lambda rc: {s: {pos.title(): cnt for pos, cnt in posd.items()} for s, posd in rc.items()})(getCounts(p.races)),
                    # position -> season -> count (used later when inserting SailorTeams)
                    'raceCounts': (lambda rc_norm, ps: {
                        'skipper': {season: rc_norm.get(season, {}).get('Skipper', 0) for season in list(ps['skipper'])},
                        'crew': {season: rc_norm.get(season, {}).get('Crew', 0) for season in list(ps['crew'])}
                    })( (lambda rc: {s: {pos.title(): cnt for pos, cnt in posd.items()} for s, posd in rc.items()})(getCounts(p.races)), p.seasons ),
                    'seasons': p.seasons,
                    'cross': sum([race.get('cross', 0) for race in p.races]),
                    'outLinks': sum([race['outLinks'] for race in p.races if 'outLinks' in race.keys()]),
                    'skipperRank': int(p.skipperRank),
                    'crewRank': int(p.crewRank),
                    'womenSkipperRank': int(p.womenSkipperRank),
                    'womenCrewRank': int(p.womenCrewRank)
                    } for p in filtered_people]
        
        topRating = 0
        topWomenRating = 0
        topRatingTR = 0
        topWomenRatingTR = 0
        
        topRating = 0
        topSkipperSum = 0
        topCrewsSum = 0
        topWomenRating = 0
        topRatingTR = 0
        topWomenRatingTR = 0

        numTops = 3
        if numCurMembers > 0:
            topSkippers = sorted([p for p in filtered_people
                                  if validPerson(p, 'skipper')
                                  #   if sum([race['cross'] for race in p.races if 'cross' in race.keys()]) > 20
                                  #   and sum([race['outLinks'] for race in p.races if 'outLinks' in race.keys()]) > 70
                                  #   and not set(p.seasons['skipper']).isdisjoint(config.targetSeasons)
                                  # and sum([p['raceCount'][seas] for seas in config.targetSeasons if seas in p['raceCount'].keys()]) > 5
                                  ],
                                 key=lambda x: x.sr.ordinal(
                                     target=config.targetElo, alpha=config.alpha),
                                 reverse=True)
            topSkipperSum = sum([p.sr.ordinal(target=config.targetElo, alpha=config.alpha)
                                 for p in topSkippers[:numTops]])
            topSkippers = [{'name': p.name, 'key': p.key,
                            'mu': p.sr.mu, 'sigma': p.sr.sigma} for p in topSkippers]

            topCrews = sorted([p for p in filtered_people if validPerson(p, 'crew')
                               #   if sum([race['cross'] for race in p.races if 'cross' in race.keys()]) > 20
                               #   and sum([race['outLinks'] for race in p.races if 'outLinks' in race.keys()]) > 70
                               #   and not set(p.seasons['crew']).isdisjoint(config.targetSeasons)
                               ],
                              key=lambda x: x.cr.ordinal(
                                  target=config.targetElo, alpha=config.alpha),
                              reverse=True)
            topCrewsSum = sum([p.cr.ordinal(target=config.targetElo, alpha=config.alpha)
                               for p in topCrews[:numTops]])
            topCrews = [{'name': p.name, 'key': p.key,
                         'mu': p.cr.mu, 'sigma': p.cr.sigma} for p in topCrews]

            topRating = (topSkipperSum + topCrewsSum) / (numTops * 2)
            # print(topSkippers, topCrews, topRating)

            # Women's
            numTops = 2
            topWomenSkippers = sorted([p['womenSkipperRating'] for p in members
                                       #   if p['skipperRank'] != 0
                                      if p['cross'] > 20
                                      and p['gender'] == 'F'
                                      and not set(p['raceCount'].keys()).isdisjoint(config.targetSeasons)
                                      and all(['Skipper' in p['raceCount'][seas].keys() for seas in config.targetSeasons if seas in p['raceCount'].keys()])
                                      and sum([p['raceCount'][seas]['Skipper'] for seas in config.targetSeasons if seas in p['raceCount'].keys()]) > 5
                                       ], reverse=True)[:numTops]
            topWomenSkipperSum = sum(topWomenSkippers)

            topWomenCrews = sorted([p['womenCrewRating'] for p in members
                                    #    if p['crewRank'] != 0
                                    if p['cross'] > 20
                                    and p['gender'] == 'F'
                                    and not set(p['raceCount'].keys()).isdisjoint(config.targetSeasons)
                                    and all(['Crew' in p['raceCount'][seas].keys() for seas in config.targetSeasons if seas in p['raceCount'].keys()])
                                    and sum([p['raceCount'][seas]['Crew'] for seas in config.targetSeasons if seas in p['raceCount'].keys()]) > 5
                                    ], reverse=True)[:numTops]
            topWomenCrewsSum = sum(topWomenCrews)

            topWomenRating = (topWomenSkipperSum +
                              topWomenCrewsSum) / (numTops * 2)

            # Team racing:

            numTops = 3

            topSkippersTR = sorted([p for p in filtered_people
                                    if p.tsr.mu != config.model.mu
                                    if not set(p.seasons['skipper']).isdisjoint(config.targetTRSeasons)],
                                   key=lambda x: x.tsr.ordinal(
                                       target=config.targetElo, alpha=config.alpha),
                                   reverse=True)
            topSkipperTRSum = sum([p.tsr.ordinal(target=config.targetElo, alpha=config.alpha)
                                   for p in topSkippersTR[:numTops]])
            topSkippersTR = [{'name': p.name, 'key': p.key,
                              'mu': p.tsr.mu, 'sigma': p.tsr.sigma} for p in topSkippersTR]

            topCrewsTR = sorted([p for p in filtered_people
                                 if p.tcr.mu != config.model.mu
                                 if not set(p.seasons['crew']).isdisjoint(config.targetTRSeasons)],
                                key=lambda x: x.tcr.ordinal(
                                    target=config.targetElo, alpha=config.alpha),
                                reverse=True)
            topCrewTRSum = sum([p.tcr.ordinal(target=config.targetElo, alpha=config.alpha)
                                for p in topCrewsTR[:numTops]])
            topCrewsTR = [{'name': p.name, 'key': p.key,
                           'mu': p.tcr.mu, 'sigma': p.tcr.sigma} for p in topCrewsTR]
            topRatingTR = (topSkipperTRSum + topCrewTRSum) / (numTops * 2)

            # WOMEN's Team Racing:

            topWomenSkippersTR = sorted([p for p in filtered_people
                                         if p.wtsr.mu != config.model.mu
                                         and p.gender == 'F'
                                         and not set(p.seasons['skipper']).isdisjoint(config.targetTRSeasons)
                                         ],
                                        key=lambda x: x.wtsr.ordinal(
                                            target=config.targetElo, alpha=config.alpha),
                                        reverse=True)
            topWomenSkipperTRSum = sum([p.wtsr.ordinal(target=config.targetElo, alpha=config.alpha)
                                        for p in topWomenSkippersTR[:numTops]])
            topWomenSkippersTR = [{'name': p.name, 'key': p.key, 'mu': p.wtsr.mu,
                                   'sigma': p.wtsr.sigma} for p in topWomenSkippersTR]

            topWomenCrewsTR = sorted([p for p in filtered_people
                                      if p.wtcr.mu != config.model.mu
                                      and p.gender == 'F'
                                      and not set(p.seasons['crew']).isdisjoint(config.targetTRSeasons)
                                      ],
                                     key=lambda x: x.wtcr.ordinal(
                                         target=config.targetElo, alpha=config.alpha),
                                     reverse=True)
            topWomenCrewTRSum = sum([p.wtcr.ordinal(target=config.targetElo, alpha=config.alpha)
                                     for p in topWomenCrewsTR[:numTops]])
            topWomenCrewsTR = [{'name': p.name, 'key': p.key, 'mu': p.wtcr.mu,
                                'sigma': p.wtcr.sigma} for p in topWomenCrewsTR]
            topWomenRatingTR = (topWomenSkipperTRSum +
                                topWomenCrewTRSum) / (numTops * 2)


        cursor.execute("""
                    INSERT INTO Teams (teamID, teamName, topFleetRating, topWomenRating, topTeamRating, topWomenTeamRating, avgRating, avgRatio, region, link)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,   (team, team, topRating, topWomenRating, topRatingTR, topWomenRatingTR, avg, avgRatio, region, url))            
        
        print("inserted team", team)
        
        batch_size = 200  # adjust for your DB/network limits
        rows_to_insert = []

        for member in members:
            if member['key'] is None:
                print("No key for", member['name'])
                continue

            for position in ['skipper', 'crew']:
                for season in set(member['seasons'][position]):
                    rows_to_insert.append((
                        member['key'].replace("/", "-"),
                        team,
                        season,
                        position,
                        member['raceCounts'][position][season]
                    ))

        # # Insert in batches
        for start in range(0, len(rows_to_insert), batch_size):
            batch = rows_to_insert[start:start + batch_size]
            try:
                cursor.executemany("""
                    INSERT IGNORE INTO SailorTeams(sailorID, teamID, season, position, raceCount)
                    VALUES(%s,%s,%s,%s,%s)
                """, batch)
                connection.commit()
            except mysql.connector.errors.IntegrityError as e:
                print("Batch insert failed:", e)
                raise e

        connection.commit()
        
    # newTeams = sorted(teams, key=lambda x: x['topRating'], reverse=True)
    # return newTeams