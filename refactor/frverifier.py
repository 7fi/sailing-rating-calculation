import os

import mysql.connector
import polars as pl
from dotenv import load_dotenv

from config import Config

RACES_FILE = "racesfr.parquet"
OUTPUT_DIR = "verifier_output"
APPLY_MERGES = True      # apply Config.merges to the parquet keys, like the calc pipeline does
DROP_SKIPPED = True      # drop races calculateFR deliberately bails on - they never reach the DB
PREVIEW_ROWS = 15        # how many example rows to print for each mismatch set
TOP_REGATTAS = 15        # how many regattas to list in the per-regatta breakdown

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (season, regatta, raceNumber, division, sailorID) is the FleetScores primary key,
# so it is also what identifies a row on both sides of this comparison.
KEY_COLS = ["season", "regatta", "raceNumber", "division", "sailorID"]

# calculateFR treats these as not having sailed, so they don't count toward its
# "at least two racers to rate" check.
EXCUSED_PENALTIES = ["DNS", "BKD", "RDG", "BYE"]


def resolvePath(filename):
    """Files live in the repo root, but this script may be run from either directory."""
    return filename if os.path.exists(filename) else os.path.join(ROOT_DIR, filename)


def splitRaceID(df):
    """s26/some-regatta/3A -> season s26, regatta some-regatta, raceNumber 3, division A"""
    parts = pl.col("raceID").str.split("/")
    return df.with_columns(
        parts.list.get(0).alias("season"),
        parts.list.get(1).alias("regatta"),
        parts.list.get(2).str.head(-1).cast(pl.Int64).alias("raceNumber"),
        parts.list.get(2).str.tail(1).alias("division"),
    )


def splitSkippedRaces(df):
    """Split off the rows calculateFR returns early on, so they aren't reported as missing.

    main.py hands calculateFR one adjusted_raceID x Position group at a time, and it
    bails on a group with fewer than two sailors, a null leading score, or fewer than
    two sailors left once excused penalties are taken out. None of those groups ever
    produce a FleetScores row.
    """
    group = ["adjusted_raceID", "Position"]
    skipped = (
        (pl.len().over(group) < 2)
        | pl.col("Score").first().over(group).is_null()
        | ((~pl.col("penalty").is_in(EXCUSED_PENALTIES)).sum().over(group) < 2)
    )
    df = df.with_columns(skipped.alias("_skipped"))
    return df.filter(~pl.col("_skipped")).drop("_skipped"), df.filter("_skipped").drop("_skipped")


def loadRaces(path):
    df = pl.read_parquet(path)
    df = df.rename({"key": "sailorID"})

    if APPLY_MERGES:
        merged = pl.col("sailorID").replace(Config.merges)
        changed = df.select((merged != pl.col("sailorID")).sum()).item()
        df = df.with_columns(merged)
        print(f"Parquet: remapped {changed} sailorIDs through Config.merges")

    skipped = df.clear()
    if DROP_SKIPPED:
        df, skipped = splitSkippedRaces(df)
        print(f"Parquet: set aside {len(skipped)} rows in races calculateFR skips")

    # A null score only ever happens for a whole group, so the skip check above already
    # covers it. Guard anyway, since int(score) would be what breaks downstream.
    nulls = df["Score"].is_null().sum()
    if nulls:
        df = df.filter(pl.col("Score").is_not_null())
        print(f"Parquet: dropped {nulls} further rows with a null Score")

    df = splitRaceID(df)

    deduped = df.unique(subset=KEY_COLS, keep="first")
    if len(deduped) != len(df):
        print(f"Parquet: {len(df) - len(deduped)} duplicate rows on {KEY_COLS} (kept one of each)")
    return deduped, splitRaceID(skipped)


def loadDB(connection):
    with connection.cursor() as cursor:
        cursor.execute(
            """SELECT season, regatta, raceNumber, division, sailorID, score, scoring, date
               FROM FleetScores"""
        )
        rows = cursor.fetchall()

    return pl.DataFrame(
        rows,
        schema={
            "season": pl.String,
            "regatta": pl.String,
            "raceNumber": pl.Int64,
            "division": pl.String,
            "sailorID": pl.String,
            "score": pl.Int64,
            "scoring": pl.String,
            "date": pl.Datetime,
        },
        orient="row",
    )


def report(label, df, filename):
    print(f"\n{'=' * 70}\n{label}: {len(df)} rows\n{'=' * 70}")
    if len(df) == 0:
        return

    # A whole scoring type or division showing up here points at a bug in how
    # raceIDs are built, rather than at genuinely absent races.
    print("\nBy scoring type:")
    print(df.group_by("scoring").agg(pl.len().alias("rows")).sort("rows", descending=True))
    print("By division:")
    print(df.group_by("division").agg(pl.len().alias("rows")).sort("division"))

    byRegatta = (
        df.group_by("season", "regatta")
        .agg(pl.len().alias("rows"), pl.col("sailorID").n_unique().alias("sailors"))
        .sort("rows", descending=True)
    )
    print(f"\nAcross {len(byRegatta)} regattas, worst first:")
    with pl.Config(tbl_rows=TOP_REGATTAS, fmt_str_lengths=60):
        print(byRegatta.head(TOP_REGATTAS))

    print("\nExample rows:")
    with pl.Config(tbl_rows=PREVIEW_ROWS, fmt_str_lengths=60, tbl_cols=-1):
        print(df.sort(KEY_COLS).head(PREVIEW_ROWS))

    path = os.path.join(OUTPUT_DIR, filename)
    df.sort(KEY_COLS).write_csv(path)
    print(f"\nWrote all {len(df)} rows to {path}")


def main():
    load_dotenv()
    connection = mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASS"),
        database=os.getenv("DB_NAME"),
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df_races, df_skipped = loadRaces(resolvePath(RACES_FILE))
    df_db = loadDB(connection)
    connection.close()

    print(f"\nExpected in DB: {len(df_races)}   Actually in DB: {len(df_db)}   difference: {len(df_races) - len(df_db)}")

    # In the parquet but not the DB - these should have been uploaded and were not.
    missing = df_races.join(df_db, on=KEY_COLS, how="anti").select(
        *KEY_COLS,
        pl.col("Score").alias("score"),
        pl.col("Scoring").alias("scoring"),
        pl.col("Sailor").alias("sailorName"),
        pl.col("Partner").alias("partnerName"),
        pl.col("Date").alias("date"),
        pl.col("Venue").alias("venue"),
    )

    # In the DB but not the parquet - stale rows that should not be there.
    extra = df_db.join(df_races, on=KEY_COLS, how="anti")

    report("MISSING FROM DB (should have been uploaded, was not)", missing, "missing_from_db.csv")
    report("EXTRA IN DB (uploaded, should not be there)", extra, "extra_in_db.csv")

    # Rows for a skipped race sitting in the DB are stale - the pipeline would not write
    # them today. They are already counted in `extra`; call them out so it is clear why.
    if len(df_skipped):
        fromSkipped = extra.join(df_skipped, on=KEY_COLS, how="semi")
        print(f"\nOf the extra rows, {len(fromSkipped)} belong to races calculateFR skips.")
        df_skipped.sort(KEY_COLS).write_csv(os.path.join(OUTPUT_DIR, "skipped_races.csv"))
        print(f"Wrote the {len(df_skipped)} set-aside skipped rows to {OUTPUT_DIR}/skipped_races.csv")

    if len(missing) == 0 and len(extra) == 0:
        print("\nEverything matches.")


if __name__ == "__main__":
    main()
