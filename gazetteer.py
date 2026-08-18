"""Bundled offline gazetteer — the always-present place lookup.

Reads the baked packs in data/gazetteer/ (official BGN-lineage sources
only: GNIS domestic, BGN Antarctic, NGA GNS worldwide when baked — see
gazetteer_build.py and data/gazetteer/MANIFEST.md for provenance) and
serves name search over EVERY name each source carries: the BGN-approved
primary AND all variants, matched after normalization (casefold +
diacritic folding + apostrophe/hyphen stripping), so "sohae" finds
Sŏhae and "dongchang" finds Tongch'ang-ni.  Results always display the
approved primary; the matched variant is reported alongside, never
silently swapped in.

First use builds a SQLite index at ~/.gui_missile_flyout/gazetteer.db
(a cache, keyed to the packs' fingerprint — delete it freely; it
rebuilds).  The packs are the source of truth; the DB is never shipped.

No third-party dependencies, no network."""

import gzip
import hashlib
import os
import sqlite3
import unicodedata
from pathlib import Path

PACK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "data", "gazetteer")
_DB_PATH = Path.home() / ".gui_missile_flyout" / "gazetteer.db"


def normalize(name):
    """Search key: casefold, strip diacritics (NFKD, drop combining
    marks), drop apostrophes/quotes, hyphens→spaces, collapse spaces.
    Tongch'ang-ri, Tongchang-ni and Dongchang all reduce comparably."""
    s = unicodedata.normalize("NFKD", name or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.casefold()
    for ch in ("'", "’", "ʻ", "ʼ", "`"):
        s = s.replace(ch, "")
    s = s.replace("-", " ")
    return " ".join(s.split())


def packs(pack_dir=None):
    d = pack_dir or PACK_DIR
    if not os.path.isdir(d):
        return []
    return sorted(os.path.join(d, f) for f in os.listdir(d)
                  if f.endswith(".txt.gz"))


def available(pack_dir=None):
    return bool(packs(pack_dir))


def _fingerprint(pack_files):
    h = hashlib.sha256()
    for p in pack_files:
        st = os.stat(p)
        h.update(f"{os.path.basename(p)}:{st.st_size}".encode())
    return h.hexdigest()[:16]


def _build_db(db, pack_files, progress=None):
    # nvar (variant-name count, primary excluded) is the map overlays'
    # prominence proxy: places the world writes about accumulate
    # romanizations.  Older indexes lack the column; readers degrade.
    db.executescript("""
        DROP TABLE IF EXISTS places;
        DROP TABLE IF EXISTS names;
        DROP TABLE IF EXISTS meta;
        CREATE TABLE places(
            id INTEGER PRIMARY KEY, ext_id TEXT, primary_name TEXT,
            lat REAL, lon REAL, admin TEXT, fclass TEXT, source TEXT,
            nvar INTEGER);
        CREATE TABLE names(
            norm TEXT, display TEXT, place_id INTEGER);
        CREATE TABLE meta(k TEXT PRIMARY KEY, v TEXT);
    """)
    pid = 0
    for pf in pack_files:
        with gzip.open(pf, "rt", encoding="utf-8") as f:
            prows, nrows = [], []
            for line in f:
                parts = line.rstrip("\n").split("|")
                if len(parts) != 8:
                    continue
                ext_id, primary, vs, lat, lon, admin, fclass, source = parts
                pid += 1
                variants = [v for v in vs.split(";") if v]
                prows.append((pid, ext_id, primary, float(lat), float(lon),
                              admin, fclass, source, len(variants)))
                nrows.append((normalize(primary), primary, pid))
                for v in variants:
                    nrows.append((normalize(v), v, pid))
                if len(prows) >= 50000:
                    db.executemany(
                        "INSERT INTO places VALUES(?,?,?,?,?,?,?,?,?)", prows)
                    db.executemany("INSERT INTO names VALUES(?,?,?)", nrows)
                    prows, nrows = [], []
                    if progress:
                        progress(pid)
            db.executemany("INSERT INTO places VALUES(?,?,?,?,?,?,?,?,?)",
                           prows)
            db.executemany("INSERT INTO names VALUES(?,?,?)", nrows)
    db.execute("CREATE INDEX idx_names ON names(norm)")
    db.execute("CREATE INDEX idx_lat ON places(lat)")
    db.execute("INSERT INTO meta VALUES('fingerprint', ?)",
               (_fingerprint(pack_files),))
    db.execute("INSERT INTO meta VALUES('schema', '2')")
    db.commit()


def index_ready(pack_dir=None, db_path=None):
    """True when a CURRENT index exists on disk — i.e. open() would
    return instantly without a (multi-minute) rebuild.  Lets the GUI
    decide whether to use the offline gazetteer or defer the build to
    an explicit user action."""
    pack_files = packs(pack_dir)
    if not pack_files:
        return False
    path = Path(db_path) if db_path else _DB_PATH
    if not path.exists():
        return False
    try:
        db = sqlite3.connect(str(path))
        fp = db.execute("SELECT v FROM meta WHERE k='fingerprint'"
                        ).fetchone()
        db.close()
    except sqlite3.OperationalError:
        return False
    return bool(fp) and fp[0] == _fingerprint(pack_files)


def ensure_index(pack_dir=None, db_path=None, progress=None):
    """Open (building/refreshing if needed) the index.  Returns a
    sqlite3 connection, or None when no packs exist.  The build is
    O(minutes) for the worldwide packs, so callers that must stay
    responsive should gate on index_ready() first."""
    pack_files = packs(pack_dir)
    if not pack_files:
        return None
    path = Path(db_path) if db_path else _DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(path), check_same_thread=False)
    try:
        fp = db.execute("SELECT v FROM meta WHERE k='fingerprint'"
                        ).fetchone()
    except sqlite3.OperationalError:
        fp = None
    if fp is None or fp[0] != _fingerprint(pack_files):
        _build_db(db, pack_files, progress)
    return db


def feature_count(pack_dir=None):
    """Total feature lines across the packs (cheap: gzip member sizes are
    not summed — this counts lines, used only for the build prompt)."""
    total = 0
    for pf in packs(pack_dir):
        with gzip.open(pf, "rt", encoding="utf-8") as f:
            total += sum(1 for _ in f)
    return total


def search(query, limit=50, db=None, pack_dir=None):
    """Ranked name search across primaries and variants.  Returns dicts:
    {primary, matched, lat, lon, admin, fclass, source, ext_id}, exact
    normalized matches first, then prefix, then substring; populated
    places before other classes within each band."""
    q = normalize(query)
    if not q:
        return []
    db = db or ensure_index(pack_dir=pack_dir)
    if db is None:
        return []
    rows = []
    seen = set()
    plans = (("exact", "norm = ?", (q,)),
             ("prefix", "norm LIKE ?", (q + "%",)),
             ("sub", "norm LIKE ?", ("%" + q + "%",)))
    for band, cond, args in plans:
        if len(rows) >= limit:
            break
        for norm, display, place_id in db.execute(
                f"SELECT norm, display, place_id FROM names WHERE {cond} "
                f"LIMIT ?", args + (4 * limit,)):
            if place_id in seen:
                continue
            seen.add(place_id)
            p = db.execute("SELECT ext_id, primary_name, lat, lon, admin, "
                           "fclass, source FROM places WHERE id=?",
                           (place_id,)).fetchone()
            rows.append(dict(ext_id=p[0], primary=p[1], matched=display,
                             lat=p[2], lon=p[3], admin=p[4], fclass=p[5],
                             source=p[6], band=band))
    rows.sort(key=lambda r: (("exact", "prefix", "sub").index(r["band"]),
                             _tier(r), r["primary"]))
    return rows[:limit]


def _tier(r):
    """Class rank inside a match band: cities, then facilities, then
    admin/areas and general features, then water/terrain, then
    vegetation/transport/undersea — so the world's ten thousand
    identically-named creeks sort below the town the user meant."""
    if (r["fclass"].startswith("PPL")
            or r["fclass"].startswith("Populated Place")):
        return 0
    src = r.get("source", "")
    if src == "GNS-S":
        return 1
    if src in ("GNS-H", "GNS-T"):
        return 3
    if src in ("GNS-V", "GNS-R", "GNS-U"):
        return 4
    return 2          # admin, areas, GNIS non-populated, Antarctic, unknown


_KM_PER_DEG_LAT = 111.19          # R·π/180; the meridional lower bound


def nearest(lat, lon, n=1, db=None, pack_dir=None, fclass_prefix=None):
    """The n nearest features to (lat, lon) by great-circle distance,
    searched over an expanding latitude band.  Returns dicts with a
    'km' field added.

    Correctness (bug fixed 2026-08-18): a latitude band is a strip over
    ALL longitudes, so the nearest feature INSIDE a band can be far in
    longitude while a closer one sits just outside it in latitude.  A
    point outside the ±h° band is at least h·111.19 km away (great-
    circle distance ≥ the latitude difference), so we only stop once
    the n-th kept candidate is within that bound — never merely because
    a band had n hits."""
    import math
    db = db or ensure_index(pack_dir=pack_dir)
    if db is None:
        return []

    def dist(row):
        la1, lo1, la2, lo2 = map(math.radians, (lat, lon, row[2], row[3]))
        a = (math.sin((la2 - la1) / 2) ** 2
             + math.cos(la1) * math.cos(la2)
             * math.sin((lo2 - lo1) / 2) ** 2)
        return 6371.0 * 2 * math.asin(min(1.0, math.sqrt(a)))

    ranked = []
    for half_band in (0.5, 2.0, 8.0, 30.0, 90.1):
        cond = "lat BETWEEN ? AND ?"
        args = [lat - half_band, lat + half_band]
        if fclass_prefix:
            cond += " AND fclass LIKE ?"
            args.append(fclass_prefix + "%")
        cands = db.execute(
            f"SELECT ext_id, primary_name, lat, lon, admin, fclass, source "
            f"FROM places WHERE {cond}", args).fetchall()
        ranked = sorted(cands, key=dist)[:n]
        # Safe to stop only when the band already reaches past the worst
        # kept distance — then nothing outside it (Δlat > half_band, so
        # > half_band·111.19 km away) can beat what we have.
        if (len(ranked) >= n
                and dist(ranked[-1]) <= half_band * _KM_PER_DEG_LAT):
            break
        if half_band > 90.0:
            break
    return [dict(ext_id=r[0], primary=r[1], lat=r[2], lon=r[3],
                 admin=r[4], fclass=r[5], source=r[6], km=dist(r))
            for r in ranked]


# Human words for the common NGA GNS designation codes (from the GNS
# User Guide's designation-code list; the long tail falls back to the
# raw code, still traceable via the ext_id).  GNIS and BGN-Antarctic
# class names are already words and pass through.
_DESIG_WORDS = {
    "PPL": "populated place", "PPLC": "capital", "PPLA": "seat of a "
    "first-order admin division", "PPLA2": "seat of a second-order "
    "admin division", "PPLA3": "seat of a third-order admin division",
    "PPLA4": "seat of a fourth-order admin division", "PPLX": "section "
    "of populated place", "PPLL": "populated locality", "PPLQ":
    "abandoned populated place", "PPLF": "farm village", "PPLW":
    "destroyed populated place", "STLMT": "settlement",
    "AIRB": "airbase", "AIRP": "airport", "AIRF": "airfield",
    "INSM": "military installation", "CTRS": "space center",
    "FT": "fort", "DAM": "dam", "PRT": "port", "STNM": "meteorological "
    "station", "STNR": "radio station", "STNS": "satellite station",
    "PSTB": "border post", "RSTN": "railroad station", "OBPT":
    "observation point", "FRM": "farm", "SCH": "school", "CH": "church",
    "MSQE": "mosque", "TMPL": "temple", "MSTY": "monastery",
    "RUIN": "ruin", "CMP": "camp", "MN": "mine", "BDG": "bridge",
    "OCN": "ocean", "SEA": "sea", "GULF": "gulf", "STRT": "strait",
    "CHN": "channel", "BAY": "bay", "COVE": "cove", "LGN": "lagoon",
    "LK": "lake", "LKI": "intermittent lake", "STM": "stream",
    "STMI": "intermittent stream", "RSV": "reservoir", "SPNG":
    "spring", "WAD": "wadi", "SWMP": "swamp", "MRSH": "marsh",
    "CNL": "canal", "HBR": "harbor", "ANCH": "anchorage", "RF": "reef",
    "SHOL": "shoal", "WLL": "well", "PND": "pond", "FLLS": "waterfall",
    "MT": "mountain", "MTS": "mountains", "PK": "peak", "HLL": "hill",
    "HLLS": "hills", "RDGE": "ridge", "VAL": "valley", "PASS": "pass",
    "ISL": "island", "ISLS": "islands", "PEN": "peninsula",
    "CAPE": "cape", "PT": "point", "PLAT": "plateau", "DSRT":
    "desert", "VLC": "volcano", "CLF": "cliff", "GRGE": "gorge",
    "DUNE": "dune", "SPUR": "spur", "RK": "rock", "SLP": "slope",
    "DPR": "depression", "PLN": "plain",
    "SMU": "seamount", "TRGU": "trough", "TRNU": "trench",
    "RDGU": "undersea ridge", "BSNU": "undersea basin", "PLNU":
    "abyssal plain", "CNYU": "undersea canyon", "BNKU": "bank",
    "KNLU": "knoll", "TMTU": "tablemount", "FURU": "undersea furrow",
    "VALU": "undersea valley", "RFU": "undersea reef", "SHLU":
    "undersea shoal", "ESCU": "escarpment", "FRZU": "fracture zone",
    "RISU": "undersea rise", "SPRU": "undersea spur", "HLU":
    "undersea hill",
    "ADM1": "first-order admin division", "ADM2": "second-order admin "
    "division", "PCLI": "independent political entity", "LCTY":
    "locality", "AREA": "area", "RGN": "region", "TRB": "tribal area",
    "CULT": "cultivated area", "FRST": "forest", "GRSLD": "grassland",
    "RD": "road", "RR": "railroad", "TNL": "tunnel", "TRIG":
    "triangulation station",
}


def class_word(fclass, source=""):
    """The feature class in words: GNS designation codes decoded via
    the table above, GNIS / Antarctic class names (already words)
    lowercased, unknown codes returned verbatim."""
    if (source or "").startswith("GNS"):
        return _DESIG_WORDS.get(fclass, fclass)
    return (fclass or "").lower() or "feature"


def family(fclass, source):
    """Coarse display family for map coloring — TOTAL over the packs
    (everything gets a family; 'other' catches admin/areas/vegetation/
    transport and unmapped classes).  GNS rides its source-pack letter;
    GNIS/Antarctic map by class name."""
    src = source or ""
    if src.startswith("GNS-"):
        return {"P": "populated", "S": "facilities", "H": "water",
                "T": "terrain", "U": "undersea"}.get(src[4:5], "other")
    fc = fclass or ""
    if fc in ("Populated Place", "Camp"):
        return "populated"
    if fc in ("Military", "Airport", "Locale", "Station", "Base"):
        return "facilities"
    if fc in ("Sea", "Gulf", "Bay", "Lake", "Stream", "Channel",
              "Reservoir", "Spring", "Canal", "Swamp", "Harbor",
              "Lagoon", "Gut", "Falls", "Rapids", "Strait", "Arroyo"):
        return "water"
    if fc in ("Island", "Islands", "Summit", "Ridge", "Range", "Cape",
              "Valley", "Basin", "Beach", "Cliff", "Peninsula",
              "Glacier", "Mountain", "Point", "Flat", "Gap", "Bar",
              "Bend", "Area", "Pillar", "Crater", "Ledge", "Rock"):
        return "terrain"
    return "other"


def compass_8(from_lat, from_lon, to_lat, to_lon):
    """8-point compass direction of (to) as seen FROM (from) — initial
    great-circle bearing, so 'impact 12 km SE of X' reads correctly."""
    import math
    la1, lo1, la2, lo2 = map(math.radians,
                             (from_lat, from_lon, to_lat, to_lon))
    y = math.sin(lo2 - lo1) * math.cos(la2)
    x = (math.cos(la1) * math.sin(la2)
         - math.sin(la1) * math.cos(la2) * math.cos(lo2 - lo1))
    brg = (math.degrees(math.atan2(y, x)) + 360.0) % 360.0
    return ("N", "NE", "E", "SE", "S", "SW", "W", "NW")[
        int((brg + 22.5) // 45.0) % 8]


def nearest_populated(lat, lon, n=1, db=None, pack_dir=None):
    """The n nearest POPULATED places, merged across the sources' own
    vocabularies (GNS PPL* designation codes, GNIS/Antarctic 'Populated
    Place'), sorted by distance.  Each dict gains 'km' and 'dir' (the
    8-point compass direction of the query point as seen from the
    place), ready for 'impact ~12 km SE of <place>' phrasing."""
    db = db or ensure_index(pack_dir=pack_dir)
    if db is None:
        return []
    rows = (nearest(lat, lon, n=n, db=db, fclass_prefix="PPL")
            + nearest(lat, lon, n=n, db=db,
                      fclass_prefix="Populated Place"))
    rows.sort(key=lambda r: r["km"])
    seen, out = set(), []
    for r in rows:
        if r["ext_id"] in seen:
            continue
        seen.add(r["ext_id"])
        r["dir"] = compass_8(r["lat"], r["lon"], lat, lon)
        out.append(r)
    return out[:n]


def has_variant_counts(db):
    """True when the index carries the nvar column (schema 2, built
    after 2026-08-18).  Older indexes still serve every query; overlay
    prominence ranking just loses its variant-count bonus until the
    user rebuilds via Analysis ▸ Reference Data."""
    cols = {r[1] for r in db.execute("PRAGMA table_info(places)")}
    return "nvar" in cols


def features_in_bbox(lat0, lat1, lon0, lon1, fclasses=None, limit=25000,
                     db=None, pack_dir=None, sample_mod=1):
    """Every feature inside the box, optionally restricted to an fclass
    set (exact designation codes / class names, matched verbatim).
    lon0..lon1 must be an ordinary interval in [-180, 180] — callers
    handle antimeridian wrap by querying the two sub-intervals.
    sample_mod > 1 keeps only ids ≡ 0 (mod k) — the ids are assigned
    uniformly at build time, so this is an unbiased 1-in-k spatial
    sample (the Explorer's wide-view thinning).  Returns
    (rows, truncated): rows are dicts with an nvar field (0 on a
    pre-schema-2 index); truncated is True when `limit` cut the
    result — never silently."""
    db = db or ensure_index(pack_dir=pack_dir)
    if db is None:
        return [], False
    nv = "nvar" if has_variant_counts(db) else "0"
    cond = "lat BETWEEN ? AND ? AND lon BETWEEN ? AND ?"
    args = [min(lat0, lat1), max(lat0, lat1),
            min(lon0, lon1), max(lon0, lon1)]
    if sample_mod > 1:
        cond += " AND id % ? = 0"
        args.append(int(sample_mod))
    if fclasses:
        fl = sorted(fclasses)
        cond += f" AND fclass IN ({','.join('?' * len(fl))})"
        args += fl
    rows = db.execute(
        f"SELECT ext_id, primary_name, lat, lon, admin, fclass, source, "
        f"{nv} FROM places WHERE {cond} LIMIT ?", args + [limit + 1]
    ).fetchall()
    truncated = len(rows) > limit
    return [dict(ext_id=r[0], primary=r[1], lat=r[2], lon=r[3], admin=r[4],
                 fclass=r[5], source=r[6], nvar=r[7])
            for r in rows[:limit]], truncated


def viewport_sample(lat0, lat1, lon0, lon1, budget=20000, db=None,
                    k_hint=1):
    """The Explorer's viewport fetch: everything in the box when it
    fits the point budget, else an unbiased 1-in-k id-modulo sample
    with k escalated until it fits.  k starts from an area-fraction
    guess against the DB's true feature count (MAX(id) — ids are
    sequential), refined by k_hint (the previous redraw's k), so a
    typical view needs ONE scan; it also de-escalates when the guess
    overshot (a feature-sparse ocean viewport), so a view that could
    have shown everything never stays needlessly thinned.  Returns
    (rows, k) — k > 1 tells the caller to say 'showing a 1-in-k
    sample' out loud."""
    db = db or ensure_index()
    if db is None:
        return [], 1
    total = (db.execute("SELECT MAX(id) FROM places").fetchone()[0]
             or 0)
    area_frac = (abs(lat1 - lat0) * abs(lon1 - lon0)) / (180.0 * 360.0)
    guess = max(1, int(total * min(1.0, area_frac) / max(1, budget)))
    k = 1
    while k < max(guess, k_hint // 4):
        k *= 4
    rows, trunc = features_in_bbox(lat0, lat1, lon0, lon1, limit=budget,
                                   db=db, sample_mod=k)
    while trunc and k < 65536:                    # too dense: thin more
        k *= 4
        rows, trunc = features_in_bbox(lat0, lat1, lon0, lon1,
                                       limit=budget, db=db,
                                       sample_mod=k)
    while not trunc and k > 1 and len(rows) <= budget // 8:
        k_try = max(1, k // 16)                   # too sparse: walk back
        rows2, trunc2 = features_in_bbox(lat0, lat1, lon0, lon1,
                                         limit=budget, db=db,
                                         sample_mod=k_try)
        if trunc2:
            break                                 # k was right after all
        rows, k = rows2, k_try
    return rows, k


def iter_features_in_bbox(lat0, lat1, lon0, lon1, fclasses=None, db=None):
    """Streaming variant of features_in_bbox — one pass over the lat
    band, no LIMIT (the caller applies its own caps).  Lets a consumer
    with per-category budgets (the map overlays) do ONE index scan for
    all categories instead of one scan each."""
    if db is None:
        db = ensure_index()
    if db is None:
        return
    nv = "nvar" if has_variant_counts(db) else "0"
    cond = "lat BETWEEN ? AND ? AND lon BETWEEN ? AND ?"
    args = [min(lat0, lat1), max(lat0, lat1),
            min(lon0, lon1), max(lon0, lon1)]
    if fclasses:
        fl = sorted(fclasses)
        cond += f" AND fclass IN ({','.join('?' * len(fl))})"
        args += fl
    for r in db.execute(
            f"SELECT ext_id, primary_name, lat, lon, admin, fclass, "
            f"source, {nv} FROM places WHERE {cond}", args):
        yield dict(ext_id=r[0], primary=r[1], lat=r[2], lon=r[3],
                   admin=r[4], fclass=r[5], source=r[6], nvar=r[7])
