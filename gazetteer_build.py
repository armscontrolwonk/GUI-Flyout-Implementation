"""Bake the bundled gazetteer packs from official gazetteer raw files.

Derive, don't invent: every record in `data/gazetteer/*.txt.gz` comes
from an official U.S. gazetteer product, carries that product's own
feature ID, and the bake is reproducible from this script + the raw
files (which live OUTSIDE the repo — USGS staged products / the Drive
"Thrusty NGA" folder — and are recorded in data/gazetteer/MANIFEST.md
with retrieval dates).

Pack format (one line per FEATURE, pipe-delimited, gzip):
    ext_id|primary_name|variants|lat|lon|admin|feature_class|source
  - variants: ';'-joined alternate names (may be empty).  VARIANTS ARE
    LOAD-BEARING (user requirement 2026-08-17): romanizations differ
    across systems, so every variant a source carries is kept and
    searchable.
  - '|' and ';' inside names are replaced with '/'.
  - lat/lon: WGS84 decimal degrees.

Sources handled:
  - GNIS DomesticNames AllStates (USGS, public domain) — all ~974k
    named U.S. features, every class kept (streams and summits are what
    remote impact points land near).
  - BGN Antarctic gazetteer GPKG (USGS/ACAN) — official names + the
    AllNames variant table.
  - NGA GNS country/class files (Phase 2; schema hook below) — the
    worldwide half, baked the same way once the raw files reach a
    session (they are proxy-blocked here; the user's Drive holds them).

Run:  python3 gazetteer_build.py <raw_dir> [<out_dir>]
"""

import glob
import gzip
import os
import sqlite3
import sys
import zipfile

OUT_DIR_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "data", "gazetteer")


def _clean(name):
    return (name or "").replace("|", "/").replace(";", "/").strip()


def _fmt(ext_id, primary, variants, lat, lon, admin, fclass, source):
    vs = ";".join(dict.fromkeys(_clean(v) for v in variants
                                if _clean(v) and _clean(v) != _clean(primary)))
    return (f"{ext_id}|{_clean(primary)}|{vs}|{float(lat):.5f}|"
            f"{float(lon):.5f}|{_clean(admin)}|{_clean(fclass)}|{source}")


def build_gnis(allstates_zip, out_path):
    """GNIS DomesticNames_AllStates_Text.zip → gnis_us pack."""
    n = 0
    z = zipfile.ZipFile(allstates_zip)
    with gzip.open(out_path, "wt", encoding="utf-8") as out:
        for member in sorted(z.namelist()):
            if not member.lower().endswith(".txt"):
                continue
            with z.open(member) as f:
                header = f.readline().decode("utf-8-sig").rstrip("\r\n")
                cols = header.split("|")
                i_id = cols.index("feature_id")
                i_nm = cols.index("feature_name")
                i_fc = cols.index("feature_class")
                i_st = cols.index("state_name")
                i_la = cols.index("prim_lat_dec")
                i_lo = cols.index("prim_long_dec")
                for raw in f:
                    p = raw.decode("utf-8", "replace").rstrip("\r\n").split("|")
                    if len(p) <= i_lo:
                        continue
                    try:
                        lat, lon = float(p[i_la]), float(p[i_lo])
                    except ValueError:
                        continue
                    if lat == 0.0 and lon == 0.0:
                        continue          # GNIS's "unknown" sentinel
                    out.write(_fmt(f"GNIS:{p[i_id]}", p[i_nm], [], lat, lon,
                                   p[i_st], p[i_fc], "GNIS") + "\n")
                    n += 1
    return n


def build_antarctica(gpkg_path, out_path):
    """BGN Antarctic GPKG → antarctica pack (variants from AllNames)."""
    db = sqlite3.connect(gpkg_path)
    variants = {}
    for fid, nm, official in db.execute(
            "SELECT feature_id, feature_name, feature_name_official "
            "FROM AntarcticaAllNames"):
        if official != 1:                 # 1 = the official form
            variants.setdefault(fid, []).append(nm)
    n = 0
    with gzip.open(out_path, "wt", encoding="utf-8") as out:
        for fid, nm, fc, lat, lon in db.execute(
                "SELECT feature_id, feature_name, feature_class, "
                "prim_lat_dec, prim_long_dec FROM AntarcticaNames"):
            if lat is None or lon is None:
                continue
            out.write(_fmt(f"ATA:{fid}", nm, variants.get(fid, []),
                           lat, lon, "Antarctica", fc or "",
                           "BGN-Antarctic") + "\n")
            n += 1
    return n


# ── Phase 2 hook: NGA GNS ───────────────────────────────────────────────────
# GNS whole-world class files (Populated_Places.zip, Spot_Features.zip)
# are tab-delimited with one row per NAME (not per feature): UFI is the
# feature id, NT the name type ('N' = BGN approved, others = variants),
# FULL_NAME_RO the romanized name.  bake: group rows by UFI, primary =
# the approved row, everything else = variants.  Column names are taken
# from the header row, so minor NGA schema drift is tolerated.
def build_gns(gns_zip, out_path, source="GNS"):
    n = 0
    # per-UFI: [primary, variants-list, lat, lon, cc, dsg] — lean lists,
    # not dicts: the worldwide Populated_Places file holds millions of
    # features and this whole map lives in RAM during the bake.
    feats = {}
    z = zipfile.ZipFile(gns_zip)
    # each GNS zip also carries disclaimer.txt / GNS_User_Guide.txt —
    # the data member is the LARGEST .txt
    member = max((m for m in z.infolist()
                  if m.filename.lower().endswith(".txt")),
                 key=lambda m: m.file_size).filename
    with z.open(member) as f:
        cols = f.readline().decode("utf-8-sig").rstrip("\r\n").split("\t")

        def idx(*names):
            for nm in names:
                if nm in cols:
                    return cols.index(nm)
            raise KeyError(names)
        i_ufi, i_nt = idx("UFI", "ufi"), idx("NT", "nt")
        i_nm = idx("full_name", "FULL_NAME_RO", "full_name_ro", "FULL_NAME")
        i_la = idx("lat_dd", "LAT", "lat")
        i_lo = idx("long_dd", "LONG", "long", "LON")
        i_cc = idx("cc_ft", "CC1", "cc1", "COUNTRY_CODE")
        i_dsg = idx("desig_cd", "DSG", "dsg")
        try:
            i_rank = idx("name_rank", "NAME_RANK")
        except KeyError:
            i_rank = None
        for raw in f:
            p = raw.decode("utf-8", "replace").rstrip("\r\n").split("\t")
            if len(p) <= max(i_nm, i_lo, i_dsg):
                continue
            try:
                lat, lon = float(p[i_la]), float(p[i_lo])
            except ValueError:
                continue
            fe = feats.get(p[i_ufi])
            if fe is None:
                fe = feats[p[i_ufi]] = [None, [], lat, lon,
                                        p[i_cc], p[i_dsg]]
            approved = p[i_nt] == "N"
            top_rank = i_rank is None or p[i_rank] in ("1", "")
            if approved and (fe[0] is None or top_rank):
                if fe[0] is not None:
                    fe[1].append(fe[0])
                fe[0] = p[i_nm]
            else:
                fe[1].append(p[i_nm])
    with gzip.open(out_path, "wt", encoding="utf-8") as out:
        for ufi, fe in feats.items():
            primary = fe[0] or (fe[1][0] if fe[1] else None)
            if not primary:
                continue
            out.write(_fmt(f"GNS:{ufi}", primary, fe[1],
                           fe[2], fe[3], fe[4], fe[5], source) + "\n")
            n += 1
    return n


_SHARD_LIMIT = 95 * 1024 * 1024        # stay under GitHub's 100 MB/file


def shard_if_needed(out_path):
    """Split an oversized pack into <name>_a/_b/… parts (whole lines,
    round-robin).  gazetteer.py reads every *.txt.gz in the directory,
    so sharding is invisible to the search code."""
    size = os.path.getsize(out_path)
    if size <= _SHARD_LIMIT:
        return [out_path]
    n_shards = size // _SHARD_LIMIT + 1
    base = out_path[:-len(".txt.gz")]
    outs = [gzip.open(f"{base}_{chr(97 + k)}.txt.gz", "wt",
                      encoding="utf-8") for k in range(n_shards)]
    with gzip.open(out_path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            outs[i % n_shards].write(line)
    for o in outs:
        o.close()
    os.remove(out_path)
    return [f"{base}_{chr(97 + k)}.txt.gz" for k in range(n_shards)]


def main(raw_dir, out_dir=OUT_DIR_DEFAULT):
    os.makedirs(out_dir, exist_ok=True)
    done = {}
    gnis = glob.glob(os.path.join(raw_dir, "DomesticNames_AllStates*.zip"))
    if gnis:
        done["gnis_us"] = build_gnis(
            gnis[0], os.path.join(out_dir, "gnis_us.txt.gz"))
    gpkg = (glob.glob(os.path.join(raw_dir, "*Antarctica*.gpkg"))
            or glob.glob(os.path.join(raw_dir, "*Antarctica*GPKG*.zip")))
    if gpkg:
        path = gpkg[0]
        if path.endswith(".zip"):
            z = zipfile.ZipFile(path)
            member = next(m for m in z.namelist() if m.endswith(".gpkg"))
            z.extract(member, raw_dir)
            path = os.path.join(raw_dir, member)
        done["antarctica"] = build_antarctica(
            path, os.path.join(out_dir, "antarctica.txt.gz"))
    # All nine GNS class files (decision 2026-08-17: bake everything —
    # thinning is a judgment call the data policy avoids).  The source
    # label carries the class so search can rank cities and facilities
    # above the world's ten thousand identically-named creeks.
    for name, src, label in (
            ("gns_pp", "Populated_Places.zip", "GNS-P"),
            ("gns_spot", "Spot_Features.zip", "GNS-S"),
            ("gns_admin", "Administrative_Regions.zip", "GNS-A"),
            ("gns_hydro", "Hydrographic.zip", "GNS-H"),
            ("gns_hypso", "Hypsographic.zip", "GNS-T"),
            ("gns_areas", "Areas_Localities.zip", "GNS-L"),
            ("gns_veg", "Vegetation.zip", "GNS-V"),
            ("gns_transport", "Transportation_Networks.zip", "GNS-R"),
            ("gns_undersea", "Undersea.zip", "GNS-U")):
        p = os.path.join(raw_dir, src)
        if os.path.exists(p):
            out = os.path.join(out_dir, f"{name}.txt.gz")
            done[name] = build_gns(p, out, source=label)
            shard_if_needed(out)
    for k, v in done.items():
        print(f"{k}: {v} features")
    return done


if __name__ == "__main__":
    main(sys.argv[1], *(sys.argv[2:3]))
