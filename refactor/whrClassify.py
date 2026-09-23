"""Derive each regatta's rating type without running the openskill pass.

`main.calculateAllRegattaInfo` decides whether a regatta is a women's event from the
genders of everyone entered, then labels it fr/wfr/tr/wtr. The joint fit needs the
same labels, and used to read them out of postcalcFRraces.parquet - which meant it
depended on a previous openskill run and, worse, on that run being current. This
reproduces the classification straight from the scraped data.
"""

import numpy as np
import pandas as pd


def _isWomens(genders):
    """Same rule as main.getWomensFR / getWomensTR: no men entered, and 4+ women.

    `genders` must be one entry per DISTINCT sailor, not per race row - the openskill
    pass takes `unique()` keys first, so counting rows would let one woman sailing
    sixteen races clear the "4+ women" bar on her own.
    """
    return ("M" not in genders) and (sum(1 for g in genders if g == "F") >= 4)


def genderMap(rootDir="", sailorInfoFile="sailor_data2.parquet", merges=None):
    info = pd.read_parquet(rootDir + sailorInfoFile, columns=["key", "gender"])
    m = dict(zip(info["key"], info["gender"]))
    if merges:
        # a merged-away key should resolve to the surviving sailor's gender
        for old, new in merges.items():
            if new in m:
                m.setdefault(old, m[new])
    return m


def fleetRatingTypes(df_races_fr, genders):
    """Per-row rating type ('sr'/'cr'/'wsr'/'wcr') for fleet races.

    Returns a Series aligned to df_races_fr. The womens flag is decided per REGATTA,
    exactly as the openskill pass does, then combined with each row's position.
    """
    pairs = df_races_fr[["Regatta", "key"]].drop_duplicates()
    pairs = pairs.assign(g=pairs["key"].map(genders))
    womensByRegatta = pairs.groupby("Regatta")["g"].apply(lambda s: _isWomens(list(s)))
    isW = df_races_fr["Regatta"].map(womensByRegatta).fillna(False)
    part = np.where(df_races_fr["Position"].str.lower() == "skipper", "s", "c")
    return pd.Series(np.where(isW, "w", "") + part + "r", index=df_races_fr.index)


def trWomensByRegatta(df_races_tr, genders):
    """Per-regatta womens flag for team races, from every sailor entered."""
    rows = []
    for reg, g in df_races_tr.groupby("Regatta"):
        keys = set()
        for col in ("allSkipperKeys", "allCrewKeys"):
            for kl in g[col]:
                if kl is not None:
                    keys.update(kl)
        rows.append((reg, _isWomens([genders.get(k) for k in keys])))
    return pd.Series(dict(rows))
