"""
locations_db — offline location catalogs for Thrusty.

The hand-curated launch-site list (launch_sites.json) stays authoritative
and small.  This module adds *generated* catalogs under data/locations/,
built from three public gazetteers:

  gcat        Jonathan McDowell's GCAT sites table — every orbital and
              suborbital launch site, test range, and impact zone.
              https://planet4589.org/space/gcat/   (CC-BY 4.0, cite GCAT)

  nga         NGA GEOnet Names Server country files — foreign place names
              maintained by the US Board on Geographic Names.
              https://geonames.nga.mil/            (US Gov, public domain)

  gnis        USGS Geographic Names Information System — US domestic
              names, including the Pacific territories and the
              U.S. Minor Outlying Islands (Wake, Johnston, Midway).
              https://www.usgs.gov/us-board-on-geographic-names
              (US Gov, public domain)

Two catalog kinds are produced, distinguished by their "format" field:

  thrusty-sites-v1    {"sites":  [{name, country, lat, lon, type, code}]}
                      merged into the Launch Site picker alongside the
                      bundled and user-defined sites (read-only).

  thrusty-places-v1   {"places": [{name, lat, lon, cc, class}]}
                      searched offline by the Find Location dialog.

CLI (stdlib only, no third-party packages):

  python3 locations_db.py gcat
  python3 locations_db.py nga RM MH --classes P,T,L
  python3 locations_db.py gnis --states HI GU UM
  python3 locations_db.py gnis --national --classes Military,Island,Range

Downloads honor the usual proxy environment variables.  Every subcommand
also accepts a pre-downloaded input (--from-tsv / --from-zip) so catalogs
can be rebuilt fully offline.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import zipfile
from pathlib import Path

LOCATIONS_DIR = Path(__file__).parent / "data" / "locations"

SITES_FORMAT = "thrusty-sites-v1"
PLACES_FORMAT = "thrusty-places-v1"

# ---------------------------------------------------------------------------
# GCAT — https://planet4589.org/space/gcat/tsv/tables/sites.tsv
# ---------------------------------------------------------------------------

GCAT_SITES_URL = "https://planet4589.org/space/gcat/tsv/tables/sites.tsv"
GCAT_LICENSE = ("GCAT (J. McDowell, planet4589.org/space/gcat), "
                "CC-BY 4.0 International")

# GCAT StateCode → display country.  Codes not listed fall back to the
# raw code, which is still meaningful (GCAT uses ISO-like codes plus
# historical entities).
GCAT_STATES = {
    "AF": "Afghanistan", "AR": "Argentina", "AU": "Australia",
    "BR": "Brazil", "CA": "Canada", "CH": "Switzerland", "CN": "China",
    "CU": "Cuba", "D": "Germany", "DD": "East Germany", "DE": "Germany",
    "DZ": "Algeria", "EG": "Egypt", "ES": "Spain", "F": "France",
    "FR": "France", "GB": "UK", "ID": "Indonesia", "IL": "Israel",
    "IN": "India", "IQ": "Iraq", "IR": "Iran", "IT": "Italy",
    "J": "Japan", "JP": "Japan", "KE": "Kenya", "KI": "Kiribati",
    "KP": "DPRK", "KR": "South Korea", "KZ": "Kazakhstan", "LY": "Libya",
    "MH": "Marshall Islands", "MX": "Mexico", "MY": "Malaysia",
    "NO": "Norway", "NZ": "New Zealand", "PE": "Peru", "PH": "Philippines",
    "PK": "Pakistan", "PL": "Poland", "RU": "Russia", "SA": "Saudi Arabia",
    "SE": "Sweden", "SU": "USSR", "SY": "Syria", "TH": "Thailand",
    "TW": "Taiwan", "UA": "Ukraine", "UK": "UK", "US": "USA",
    "VN": "Vietnam", "YE": "Yemen", "ZA": "South Africa",
}

_GCAT_NAME_COLS = ("Name", "EName", "ShortEName", "ShortName", "Site")
_GCAT_LAT_COLS = ("Latitude", "Lat")
_GCAT_LON_COLS = ("Longitude", "Lon", "Long")


def _gcat_field(row: dict, cols) -> str:
    """First non-empty value among aliased GCAT columns ('-' = empty)."""
    for c in cols:
        v = row.get(c, "").strip()
        if v and v != "-":
            return v
    return ""


def parse_gcat_sites(text: str) -> list:
    """
    Parse GCAT sites.tsv into a list of row dicts keyed by header name.

    GCAT TSV convention: the first line is the header, prefixed with '#';
    subsequent '#' lines are comments; '-' marks an empty field.
    """
    rows = []
    header = None
    for line in text.splitlines():
        if not line.strip():
            continue
        if line.startswith("#"):
            if header is None:
                header = [h.strip() for h in line.lstrip("#").split("\t")]
            continue
        if header is None:
            # File without the '#' header convention — treat first line as header.
            header = [h.strip() for h in line.split("\t")]
            continue
        parts = line.split("\t")
        rows.append({h: (parts[i] if i < len(parts) else "")
                     for i, h in enumerate(header)})
    return rows


def gcat_to_sites(rows: list, types=None) -> list:
    """
    Convert parsed GCAT rows into thrusty-sites entries.

    types — optional iterable of Type-code prefixes (case-insensitive);
    a row is kept when its Type starts with any of them.  None keeps all
    rows that have usable coordinates.
    """
    prefixes = None
    if types:
        prefixes = tuple(t.strip().upper() for t in types if t.strip())
    sites, seen = [], set()
    for row in rows:
        name = _gcat_field(row, _GCAT_NAME_COLS)
        lat_s = _gcat_field(row, _GCAT_LAT_COLS)
        lon_s = _gcat_field(row, _GCAT_LON_COLS)
        if not name or not lat_s or not lon_s:
            continue
        try:
            lat, lon = float(lat_s), float(lon_s)
        except ValueError:
            continue
        typ = _gcat_field(row, ("Type",)).upper()
        if prefixes is not None and not typ.startswith(prefixes):
            continue
        code = _gcat_field(row, ("StateCode",))
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        sites.append({
            "name": name,
            "country": GCAT_STATES.get(code, code or "?"),
            "lat": lat,
            "lon": lon,
            "type": typ,
            "code": _gcat_field(row, ("Site",)),
        })
    return sites


# ---------------------------------------------------------------------------
# NGA GEOnet country files — https://geonames.nga.mil/
# ---------------------------------------------------------------------------

# The GNS download portal has moved several times; try known layouts in
# order.  {cc} is the country code as given (tried lower- and upper-case);
# the current portal uses GENC 2-letter codes, the legacy one used FIPS.
NGA_URL_PATTERNS = (
    "https://geonames.nga.mil/geonames/GNSData/fc_files/{cc}.zip",
    "https://geonames.nga.mil/gns/html/cntyfile/{cc}.zip",
)
NGA_LICENSE = "NGA GEOnet Names Server (US Government, public domain)"

# Column aliases across the classic (FULL_NAME_RO/LAT/LONG) and the
# post-2022 GIS-style (full_name/lat_dd/long_dd) country-file schemas.
_NGA_ALIASES = {
    "name": ("full_name_ro", "full_name", "full_name_nd_ro"),
    "lat": ("lat", "lat_dd"),
    "lon": ("long", "long_dd", "lon"),
    "fc": ("fc",),
    "dsg": ("dsg", "desig_cd"),
    "nt": ("nt",),
    "cc": ("cc1", "cc_ft", "cc_gnc"),
    "ufi": ("ufi",),
}

# Name types kept by default: N (BGN approved), C (conventional).
# Variant (V), unverified (D), provisional (P), historic (H) names are
# noise for a coordinate picker unless explicitly requested.
_NGA_DEFAULT_NT = ("N", "C", "NS", "CS")


def _resolve_columns(header: list, aliases: dict) -> dict:
    """Map logical field → column index using case-insensitive aliases."""
    lower = [h.strip().lower().lstrip("﻿") for h in header]
    out = {}
    for field, names in aliases.items():
        for n in names:
            if n in lower:
                out[field] = lower.index(n)
                break
    return out


def parse_nga_text(text: str, default_cc: str = "", classes=None,
                   all_names: bool = False) -> list:
    """
    Parse one NGA GEOnet country file (tab-delimited, header row) into
    thrusty-places entries.

    classes — optional iterable of feature-class letters (A P V L U R T H S).
    all_names — keep every name type, not just approved/conventional.
    """
    lines = text.splitlines()
    if not lines:
        return []
    cols = _resolve_columns(lines[0].split("\t"), _NGA_ALIASES)
    if "name" not in cols or "lat" not in cols or "lon" not in cols:
        raise ValueError(
            "unrecognized NGA country-file header: " + lines[0][:120])
    keep_fc = None
    if classes:
        keep_fc = {c.strip().upper() for c in classes if c.strip()}
    places, seen_ufi = [], set()

    def _get(parts, field, default=""):
        i = cols.get(field)
        return parts[i].strip() if i is not None and i < len(parts) else default

    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split("\t")
        name = _get(parts, "name")
        if not name:
            continue
        try:
            lat = float(_get(parts, "lat"))
            lon = float(_get(parts, "lon"))
        except ValueError:
            continue
        nt = _get(parts, "nt").upper()
        if not all_names and nt and nt not in _NGA_DEFAULT_NT:
            continue
        fc = _get(parts, "fc").upper()
        if keep_fc is not None and fc and fc not in keep_fc:
            continue
        ufi = _get(parts, "ufi")
        if ufi:
            if ufi in seen_ufi:
                continue
            seen_ufi.add(ufi)
        places.append({
            "name": name,
            "lat": lat,
            "lon": lon,
            "cc": (_get(parts, "cc") or default_cc).upper(),
            "class": _get(parts, "dsg") or fc,
        })
    return places


# ---------------------------------------------------------------------------
# USGS GNIS domestic names — https://www.usgs.gov/us-board-on-geographic-names
# ---------------------------------------------------------------------------

GNIS_URL = ("https://prd-tnm.s3.amazonaws.com/StagedProducts/GeographicNames/"
            "DomesticNames/DomesticNames_{key}_Text.zip")
GNIS_LICENSE = "USGS GNIS / US Board on Geographic Names (public domain)"

# Default feature classes: the ones relevant to a trajectory tool.  GNIS
# has ~60 classes; Populated Place alone is ~200k entries nationally, so
# it is opt-in via --classes.
GNIS_DEFAULT_CLASSES = ("Military", "Island", "Range")

# The Minor Outlying Islands (Wake, Johnston, Midway, …) have no per-state
# file; they only appear in the National file under this state_name.
GNIS_UM_STATE = "U.S. Minor Outlying Islands"

_GNIS_STATE_POSTAL = {
    "Alabama": "AL", "Alaska": "AK", "American Samoa": "AS",
    "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
    "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "District of Columbia": "DC", "Florida": "FL", "Georgia": "GA",
    "Guam": "GU", "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL",
    "Indiana": "IN", "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY",
    "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN",
    "Mississippi": "MS", "Missouri": "MO", "Montana": "MT",
    "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH",
    "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND",
    "Northern Mariana Islands": "MP", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Puerto Rico": "PR",
    "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD",
    "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT",
    "Virgin Islands": "VI", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
    GNIS_UM_STATE: "UM",
}


def parse_gnis_text(text: str, classes=None, state_names=None) -> list:
    """
    Parse a GNIS DomesticNames text file (pipe-delimited, header row)
    into thrusty-places entries.

    classes     — optional iterable of feature_class strings to keep.
    state_names — optional iterable of state_name strings to keep (the
                  National file mixes all states).
    """
    lines = text.splitlines()
    if not lines:
        return []
    header = [h.strip().lstrip("﻿") for h in lines[0].split("|")]
    idx = {h: i for i, h in enumerate(header)}
    try:
        i_name = idx["feature_name"]
        i_class = idx["feature_class"]
        i_state = idx["state_name"]
        i_lat = idx["prim_lat_dec"]
        i_lon = idx["prim_long_dec"]
        i_id = idx["feature_id"]
    except KeyError as exc:
        raise ValueError(f"unrecognized GNIS header, missing {exc}")
    keep_cls = {c.strip() for c in classes if c.strip()} if classes else None
    keep_st = set(state_names) if state_names else None
    places, seen = [], set()
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) <= max(i_name, i_class, i_state, i_lat, i_lon, i_id):
            continue
        if keep_cls is not None and parts[i_class] not in keep_cls:
            continue
        if keep_st is not None and parts[i_state] not in keep_st:
            continue
        try:
            lat = float(parts[i_lat])
            lon = float(parts[i_lon])
        except ValueError:
            continue
        if lat == 0.0 and lon == 0.0:
            continue
        fid = parts[i_id]
        if fid in seen:
            continue
        seen.add(fid)
        places.append({
            "name": parts[i_name],
            "lat": lat,
            "lon": lon,
            "cc": _GNIS_STATE_POSTAL.get(parts[i_state], "US"),
            "class": parts[i_class],
        })
    return places


# ---------------------------------------------------------------------------
# Catalog files — write, load, search
# ---------------------------------------------------------------------------

def write_catalog(path: Path, fmt: str, entries: list, source: str,
                  url: str, license_str: str) -> None:
    key = "sites" if fmt == SITES_FORMAT else "places"
    doc = {
        "format": fmt,
        "source": source,
        "url": url,
        "license": license_str,
        "retrieved": time.strftime("%Y-%m-%d"),
        "count": len(entries),
        key: entries,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, ensure_ascii=False))


_CATALOG_CACHE = {}   # path → (mtime, parsed doc)


def _load_catalog(path: Path):
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    cached = _CATALOG_CACHE.get(path)
    if cached and cached[0] == mtime:
        return cached[1]
    try:
        doc = json.loads(path.read_text())
    except Exception as exc:
        print(f"Warning: could not load location catalog {path.name}: {exc}")
        return None
    _CATALOG_CACHE[path] = (mtime, doc)
    return doc


def _iter_catalogs(dirpath: Path):
    if not dirpath.is_dir():
        return
    for p in sorted(dirpath.glob("*.json")):
        doc = _load_catalog(p)
        if doc is not None:
            yield doc


def load_extra_sites(dirpath: Path = LOCATIONS_DIR) -> list:
    """All launch/test sites from generated site catalogs (GCAT)."""
    sites = []
    for doc in _iter_catalogs(dirpath):
        if doc.get("format") == SITES_FORMAT:
            sites.extend(doc.get("sites", []))
    return sites


def load_places(dirpath: Path = LOCATIONS_DIR) -> list:
    """All named places from generated places catalogs (NGA, GNIS)."""
    places = []
    for doc in _iter_catalogs(dirpath):
        if doc.get("format") == PLACES_FORMAT:
            places.extend(doc.get("places", []))
    return places


def search_places(query: str, limit: int = 50,
                  dirpath: Path = LOCATIONS_DIR) -> list:
    """
    Case-insensitive substring search over the places catalogs.

    Returns [(name, lat, lon, cc, class), …], prefix matches first, then
    shorter names first — so 'Kwajalein' surfaces the atoll before every
    pier and school on it.
    """
    q = query.strip().lower()
    if not q:
        return []
    hits = []
    for p in load_places(dirpath):
        n = p["name"].lower()
        i = n.find(q)
        if i < 0:
            continue
        hits.append((i > 0, len(n), p["name"], p["lat"], p["lon"],
                     p.get("cc", ""), p.get("class", "")))
    hits.sort()
    return [(h[2], h[3], h[4], h[5], h[6]) for h in hits[:limit]]


# ---------------------------------------------------------------------------
# Downloads (CLI only — the GUI never touches the network here)
# ---------------------------------------------------------------------------

def _download(url: str) -> bytes:
    import urllib.request
    req = urllib.request.Request(
        url, headers={"User-Agent": "thrusty-locations-db/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def _zip_texts(data: bytes):
    """Yield (name, text) for every .txt member of a zip archive."""
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for info in zf.infolist():
            if info.filename.lower().endswith(".txt"):
                yield info.filename, zf.read(info).decode(
                    "utf-8", errors="replace")


# ---------------------------------------------------------------------------
# CLI subcommands
# ---------------------------------------------------------------------------

def _cmd_gcat(args) -> int:
    if args.from_tsv:
        text = Path(args.from_tsv).read_text()
        src_url = f"file:{args.from_tsv}"
    else:
        url = args.url or GCAT_SITES_URL
        print(f"Downloading {url} …")
        text = _download(url).decode("utf-8", errors="replace")
        src_url = url
    rows = parse_gcat_sites(text)
    types = args.types.split(",") if args.types else None
    sites = gcat_to_sites(rows, types=types)
    if not sites:
        print("No sites parsed — check the input file format.")
        return 1
    out = Path(args.out) if args.out else LOCATIONS_DIR / "gcat_sites.json"
    write_catalog(out, SITES_FORMAT, sites, "GCAT sites table",
                  src_url, GCAT_LICENSE)
    print(f"Wrote {len(sites)} sites → {out}")
    return 0


def _cmd_nga(args) -> int:
    classes = args.classes.split(",") if args.classes else None
    status = 0
    for cc in args.countries:
        data = None
        if args.from_zip:
            data = Path(args.from_zip).read_bytes()
            src_url = f"file:{args.from_zip}"
        else:
            tried = []
            urls = ([args.url] if args.url else
                    [pat.format(cc=c)
                     for pat in NGA_URL_PATTERNS
                     for c in (cc.lower(), cc.upper())])
            for url in urls:
                try:
                    print(f"Trying {url} …")
                    data = _download(url)
                    src_url = url
                    break
                except Exception as exc:
                    tried.append(f"  {url}  ({exc})")
            if data is None:
                print(f"Could not download NGA country file for '{cc}'. "
                      f"Tried:\n" + "\n".join(tried))
                print("Download it manually from https://geonames.nga.mil/ "
                      "and re-run with --from-zip <file>.")
                status = 1
                continue
        places = []
        for name, text in _zip_texts(data):
            try:
                places.extend(parse_nga_text(
                    text, default_cc=cc, classes=classes,
                    all_names=args.all_names))
            except ValueError as exc:
                print(f"  Skipping {name}: {exc}")
        if not places:
            print(f"No places parsed for '{cc}'.")
            status = 1
            continue
        out = (Path(args.out) if args.out and len(args.countries) == 1
               else LOCATIONS_DIR / f"places_nga_{cc.lower()}.json")
        write_catalog(out, PLACES_FORMAT, places,
                      f"NGA GEOnet country file ({cc.upper()})",
                      src_url, NGA_LICENSE)
        print(f"Wrote {len(places)} places → {out}")
    return status


def _cmd_gnis(args) -> int:
    classes = (args.classes.split(",") if args.classes
               else list(GNIS_DEFAULT_CLASSES))
    states = [s.upper() for s in (args.states or [])]
    # UM has no per-state file — it rides along in the National file.
    need_national = args.national or "UM" in states
    keys = ["National"] if need_national else states
    if not keys:
        print("Specify --states (postal codes, UM allowed) or --national.")
        return 1
    all_places = []
    for key in keys:
        if args.from_zip:
            data = Path(args.from_zip).read_bytes()
            src_url = f"file:{args.from_zip}"
        else:
            url = GNIS_URL.format(key=key)
            print(f"Downloading {url} …")
            data = _download(url)
            src_url = url
        state_names = None
        if key == "National" and states and not args.national:
            postal_to_name = {v: k for k, v in _GNIS_STATE_POSTAL.items()}
            state_names = [postal_to_name[s] for s in states
                           if s in postal_to_name]
        for _name, text in _zip_texts(data):
            all_places.extend(parse_gnis_text(
                text, classes=classes, state_names=state_names))
    if not all_places:
        print("No places parsed — check --classes / --states.")
        return 1
    # The per-state files include border-straddling features from
    # neighbours; dedupe across files by (name, lat, lon).
    seen, places = set(), []
    for p in all_places:
        k = (p["name"], p["lat"], p["lon"])
        if k not in seen:
            seen.add(k)
            places.append(p)
    tag = "national" if args.national else "_".join(s.lower() for s in states)
    out = Path(args.out) if args.out else (
        LOCATIONS_DIR / f"places_gnis_{tag}.json")
    write_catalog(out, PLACES_FORMAT, places,
                  f"USGS GNIS domestic names ({tag})", src_url, GNIS_LICENSE)
    print(f"Wrote {len(places)} places → {out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="locations_db.py",
        description="Build offline location catalogs for Thrusty "
                    "(data/locations/*.json).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gcat", help="GCAT launch/test sites (planet4589.org)")
    g.add_argument("--types", help="comma-separated Type-code prefixes "
                                   "to keep (default: all)")
    g.add_argument("--url", help="override the sites.tsv URL")
    g.add_argument("--from-tsv", help="use a local sites.tsv instead of "
                                      "downloading")
    g.add_argument("--out", help="output JSON path")
    g.set_defaults(func=_cmd_gcat)

    n = sub.add_parser("nga", help="NGA GEOnet country files "
                                   "(geonames.nga.mil, foreign names)")
    n.add_argument("countries", nargs="+",
                   help="country codes, e.g. RM MH (try both FIPS and "
                        "GENC codes if one 404s)")
    n.add_argument("--classes", help="feature-class letters to keep, "
                                     "e.g. P,T,L (default: all)")
    n.add_argument("--all-names", action="store_true",
                   help="keep variant/unverified names too")
    n.add_argument("--url", help="override the country-file URL")
    n.add_argument("--from-zip", help="use a local country zip instead of "
                                      "downloading")
    n.add_argument("--out", help="output JSON path (single country only)")
    n.set_defaults(func=_cmd_nga)

    u = sub.add_parser("gnis", help="USGS GNIS domestic names (US + "
                                    "territories + Minor Outlying Islands)")
    u.add_argument("--states", nargs="+",
                   help="state/territory postal codes (UM = Wake, "
                        "Johnston, Midway …)")
    u.add_argument("--national", action="store_true",
                   help="whole-US file instead of per-state files")
    u.add_argument("--classes",
                   help="comma-separated feature classes (default: "
                        + ",".join(GNIS_DEFAULT_CLASSES) + ")")
    u.add_argument("--from-zip", help="use a local DomesticNames zip "
                                      "instead of downloading")
    u.add_argument("--out", help="output JSON path")
    u.set_defaults(func=_cmd_gnis)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
