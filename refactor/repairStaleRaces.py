"""One-time repair for stale fleet rows left in the races file by older scrapes.

Until dropRescrapedRegattas existed, re-scraping a regatta merged on (raceID, Sailor).
When a team fixed the wrong sailors it had entered, the corrected page carried no row at
all for the people who were taken off, so nothing overwrote their rows and they stayed in
the file. The scraper no longer does this, but it will not clean up after itself either:
an unchanged regatta page answers 304, is never reprocessed, and so never gets replaced.

Every row a single scrape of a regatta produces is stamped with essentially the same
updatedAt, and successive scrapes are days apart. So rows sitting well behind their own
regatta's newest stamp are leftovers from a run that has since been superseded.

    python repairStaleRaces.py           # report only
    python repairStaleRaces.py --apply   # rewrite the file, keeping a .bak
"""

import argparse
import os
import shutil

import pandas as pd

RACES_FILE = "racesfr.parquet"

# A regatta is processed in one call, so its rows land within a second of each other.
# An hour is far inside the gap between two pipeline runs.
RUN_TOLERANCE = 3600

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolvePath(filename):
    """Files live in the repo root, but this script may be run from either directory."""
    return filename if os.path.exists(filename) else os.path.join(ROOT_DIR, filename)


def findStaleRows(df_races):
    """Mask of rows belonging to a scrape run that a later scrape of that regatta replaced."""
    newestRun = df_races.groupby('Regatta')['updatedAt'].transform('max')
    return df_races['updatedAt'] < newestRun - RUN_TOLERANCE


def report(df_races, stale):
    print(f"{len(df_races)} rows, {int(stale.sum())} stale")
    if not stale.any():
        print("Nothing to repair.")
        return

    byRegatta = (
        df_races.loc[stale]
        .groupby('Regatta')
        .agg(staleRows=('raceID', 'size'), sailors=('Sailor', 'nunique'))
        .sort_values('staleRows', ascending=False)
    )
    print(f"\nAcross {len(byRegatta)} regattas:\n{byRegatta}")

    print("\nSailors being dropped:")
    perSailor = (
        df_races.loc[stale]
        .groupby(['Regatta', 'Sailor', 'Team'])
        .agg(rows=('raceID', 'size'))
        .sort_values('rows', ascending=False)
    )
    with pd.option_context('display.max_rows', 40, 'display.width', 140):
        print(perSailor)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true',
                        help="rewrite the races file instead of only reporting")
    parser.add_argument('--file', default=RACES_FILE)
    args = parser.parse_args()

    path = resolvePath(args.file)
    df_races = pd.read_parquet(path)

    stale = findStaleRows(df_races)
    report(df_races, stale)

    if not args.apply or not stale.any():
        if not args.apply and stale.any():
            print("\nRe-run with --apply to remove these rows.")
        return

    backup = path + ".bak"
    shutil.copy2(path, backup)
    print(f"\nBacked up to {backup}")

    df_races.loc[~stale].reset_index(drop=True).to_parquet(path, index=False)
    print(f"Wrote {int((~stale).sum())} rows to {path}")


if __name__ == "__main__":
    main()
