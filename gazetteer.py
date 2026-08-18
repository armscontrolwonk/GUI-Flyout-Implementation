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


def nearest(lat, lon, n=1, db=None, pack_dir=None, fclass_prefix=None):
    """The n nearest features to (lat, lon) by great-circle distance,
    searched over an expanding latitude band.  Returns dicts with a
    'km' field added."""
    import math
    db = db or ensure_index(pack_dir=pack_dir)
    if db is None:
        return []
    for half_band in (0.5, 2.0, 8.0, 30.0, 90.1):
        cond = "lat BETWEEN ? AND ?"
        args = [lat - half_band, lat + half_band]
        if fclass_prefix:
            cond += " AND fclass LIKE ?"
            args.append(fclass_prefix + "%")
        cands = db.execute(
            f"SELECT ext_id, primary_name, lat, lon, admin, fclass, source "
            f"FROM places WHERE {cond}", args).fetchall()
        if len(cands) >= n or half_band > 90.0:
            break
    def dist(row):
        la1, lo1, la2, lo2 = map(math.radians, (lat, lon, row[2], row[3]))
        a = (math.sin((la2 - la1) / 2) ** 2
             + math.cos(la1) * math.cos(la2)
             * math.sin((lo2 - lo1) / 2) ** 2)
        return 6371.0 * 2 * math.asin(min(1.0, math.sqrt(a)))
    out = []
    for row in sorted(cands, key=dist)[:n]:
        out.append(dict(ext_id=row[0], primary=row[1], lat=row[2],
                        lon=row[3], admin=row[4], fclass=row[5],
                        source=row[6], km=dist(row)))
    return out


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
                     db=None, pack_dir=None):
    """Every feature inside the box, optionally restricted to an fclass
    set (exact designation codes / class names, matched verbatim).
    lon0..lon1 must be an ordinary interval in [-180, 180] — callers
    handle antimeridian wrap by querying the two sub-intervals.
    Returns (rows, truncated): rows are dicts with an nvar field (0 on
    a pre-schema-2 index); truncated is True when `limit` cut the
    result — never silently."""
    db = db or ensure_index(pack_dir=pack_dir)
    if db is None:
        return [], False
    nv = "nvar" if has_variant_counts(db) else "0"
    cond = "lat BETWEEN ? AND ? AND lon BETWEEN ? AND ?"
    args = [min(lat0, lat1), max(lat0, lat1),
            min(lon0, lon1), max(lon0, lon1)]
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
