"""Location catalogs — GCAT / NGA GEOnet / USGS GNIS parsers and the
catalog files the GUI merges into the site picker and Find Location
dialog.  All fixtures are inline; no network, no tkinter."""

import json

import locations_db as ldb


# ---------------------------------------------------------------------------
# GCAT sites.tsv
# ---------------------------------------------------------------------------

_GCAT = "\n".join([
    "#Site\tCode\tUCode\tType\tStateCode\tTStart\tTStop\tShortName\tName"
    "\tLocation\tLongitude\tLatitude\tError\tParent\tShortEName\tEName"
    "\tGroup\tUName",
    "# comment line to be ignored",
    "KMR\tKMR\tKMR\tLS\tMH\t1961\t-\tKwajalein\tKwajalein Missile Range"
    "\tKwajalein Atoll\t167.7431\t9.0477\t1.0\t-\t-\t-\t-\t-",
    "NOWHERE\tNW\tNW\tLS\tUS\t-\t-\tNowhere\tNowhere Range"
    "\tNowhere\t-\t-\t-\t-\t-\t-\t-\t-",             # no coords → dropped
    "WOO\tWOO\tWOO\tSS\tAU\t1947\t-\tWoomera\tWoomera Test Range"
    "\tSouth Australia\t136.82\t-31.15\t1.0\t-\t-\t-\t-\t-",
    "KMR2\tKMR2\tKMR2\tLS\tMH\t1961\t-\tKwajalein\tKwajalein Missile Range"
    "\tdupe name\t167.74\t9.05\t1.0\t-\t-\t-\t-\t-",  # dupe name → dropped
])


def test_gcat_parse_and_convert():
    rows = ldb.parse_gcat_sites(_GCAT)
    assert len(rows) == 4
    sites = ldb.gcat_to_sites(rows)
    names = [s["name"] for s in sites]
    assert names == ["Kwajalein Missile Range", "Woomera Test Range"]
    kmr = sites[0]
    assert kmr["country"] == "Marshall Islands"
    assert kmr["lat"] == 9.0477 and kmr["lon"] == 167.7431
    assert kmr["type"] == "LS"


def test_gcat_type_filter():
    rows = ldb.parse_gcat_sites(_GCAT)
    only_ls = ldb.gcat_to_sites(rows, types=["LS"])
    assert [s["name"] for s in only_ls] == ["Kwajalein Missile Range"]


def test_gcat_headerless_fallback():
    # A file that lost its '#' prefix still parses.
    rows = ldb.parse_gcat_sites(_GCAT.replace("#Site", "Site", 1))
    assert ldb.gcat_to_sites(rows)[0]["name"] == "Kwajalein Missile Range"


# ---------------------------------------------------------------------------
# NGA GEOnet country files — classic and GIS-style headers
# ---------------------------------------------------------------------------

def _tabs(*cols):
    return "\t".join(cols)


_NGA_CLASSIC = "\n".join([
    _tabs("RC", "UFI", "UNI", "LAT", "LONG", "DMS_LAT", "DMS_LONG", "MGRS",
          "JOG", "FC", "DSG", "PC", "CC1", "ADM1", "POP", "ELEV", "CC2",
          "NT", "LC", "SHORT_FORM", "GENERIC", "SORT_NAME_RO",
          "FULL_NAME_RO", "FULL_NAME_ND_RO"),
    _tabs("1", "-3067331", "1", "8.716667", "167.733333", "", "", "", "",
          "L", "ATOL", "", "RM", "", "", "", "", "N", "", "", "",
          "KWAJALEINATOLL", "Kwajalein Atoll", "Kwajalein Atoll"),
    _tabs("1", "-3067331", "2", "8.716667", "167.733333", "", "", "", "",
          "L", "ATOL", "", "RM", "", "", "", "", "V", "", "", "",
          "MENASCHI", "Menaschi", "Menaschi"),                # variant name
    _tabs("1", "-3067400", "1", "9.396700", "167.470000", "", "", "", "",
          "P", "PPL", "", "RM", "", "", "", "", "N", "", "", "",
          "ROINAMUR", "Roi-Namur", "Roi-Namur"),
])


def test_nga_classic_parse():
    places = ldb.parse_nga_text(_NGA_CLASSIC, default_cc="rm")
    # Variant name row shares the UFI and is a V-type → dropped.
    assert [p["name"] for p in places] == ["Kwajalein Atoll", "Roi-Namur"]
    atoll = places[0]
    assert atoll["cc"] == "RM"
    assert atoll["class"] == "ATOL"
    assert abs(atoll["lat"] - 8.716667) < 1e-9


def test_nga_class_filter_and_all_names():
    only_p = ldb.parse_nga_text(_NGA_CLASSIC, classes=["P"])
    assert [p["name"] for p in only_p] == ["Roi-Namur"]
    every = ldb.parse_nga_text(_NGA_CLASSIC, all_names=True)
    # Variant still deduped by UFI even when name types are kept.
    assert [p["name"] for p in every] == ["Kwajalein Atoll", "Roi-Namur"]


_NGA_GIS = "\n".join([
    _tabs("rk", "ufi", "full_name", "nt", "lat_dd", "long_dd", "fc",
          "desig_cd", "cc_ft"),
    _tabs("1", "99", "Kwajalein Atoll", "N", "8.716667", "167.733333",
          "L", "ATOL", "MH"),
])


def test_nga_gis_style_header():
    places = ldb.parse_nga_text(_NGA_GIS)
    assert places == [{"name": "Kwajalein Atoll", "lat": 8.716667,
                       "lon": 167.733333, "cc": "MH", "class": "ATOL"}]


def test_nga_unknown_header_raises():
    try:
        ldb.parse_nga_text("foo\tbar\n1\t2")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on unknown header")


# ---------------------------------------------------------------------------
# USGS GNIS domestic names
# ---------------------------------------------------------------------------

_GNIS = "\n".join([
    "﻿feature_id|feature_name|feature_class|state_name|state_numeric"
    "|county_name|county_numeric|map_name|date_created|date_edited"
    "|bgn_type|bgn_authority|bgn_date|prim_lat_dms|prim_long_dms"
    "|prim_lat_dec|prim_long_dec|source_lat_dms|source_long_dms"
    "|source_lat_dec|source_long_dec",
    "1|Wake Atoll|Island|U.S. Minor Outlying Islands|74|||||||||192855N"
    "|1663901E|19.2819444|166.6502778|||0.0|0.0",
    "2|Vandenberg Space Force Base|Military|California|06|||||||||"
    "333311N|1203241W|34.5590053|-120.5451795|||0.0|0.0",
    "3|Pearl Harbor|Bay|Hawaii|15|||||||||212100N|1575800W"
    "|21.3500000|-157.9666667|||0.0|0.0",
    "1|Wake Atoll|Island|U.S. Minor Outlying Islands|74|||||||||192855N"
    "|1663901E|19.2819444|166.6502778|||0.0|0.0",       # dupe feature_id
    "4|Ghost Feature|Island|Hawaii|15|||||||||||0.0|0.0|||0.0|0.0",
])


def test_gnis_parse_default():
    places = ldb.parse_gnis_text(_GNIS)
    # Bay not in default class filter is only applied by the CLI; the
    # parser itself keeps everything unless classes= is passed.
    names = [p["name"] for p in places]
    assert names == ["Wake Atoll", "Vandenberg Space Force Base",
                     "Pearl Harbor"]
    wake = places[0]
    assert wake["cc"] == "UM" and wake["class"] == "Island"


def test_gnis_class_and_state_filters():
    mil = ldb.parse_gnis_text(_GNIS, classes=["Military"])
    assert [p["name"] for p in mil] == ["Vandenberg Space Force Base"]
    um = ldb.parse_gnis_text(_GNIS, state_names=[ldb.GNIS_UM_STATE])
    assert [p["name"] for p in um] == ["Wake Atoll"]


# ---------------------------------------------------------------------------
# Catalog files: write → load → search, and the site-catalog merge
# ---------------------------------------------------------------------------

def test_catalog_roundtrip_and_search(tmp_path):
    places = ldb.parse_gnis_text(_GNIS)
    ldb.write_catalog(tmp_path / "places_test.json", ldb.PLACES_FORMAT,
                      places, "test", "file:test", "public domain")
    sites = ldb.gcat_to_sites(ldb.parse_gcat_sites(_GCAT))
    ldb.write_catalog(tmp_path / "gcat_sites.json", ldb.SITES_FORMAT,
                      sites, "test", "file:test", "CC-BY")

    doc = json.loads((tmp_path / "places_test.json").read_text())
    assert doc["format"] == ldb.PLACES_FORMAT and doc["count"] == 3

    # Sites and places load from their respective formats only.
    assert [s["name"] for s in ldb.load_extra_sites(tmp_path)] == \
        ["Kwajalein Missile Range", "Woomera Test Range"]
    assert len(ldb.load_places(tmp_path)) == 3

    # Prefix match ranks ahead of substring match.
    hits = ldb.search_places("wake", dirpath=tmp_path)
    assert hits[0][0] == "Wake Atoll"
    assert hits[0][3] == "UM"
    assert ldb.search_places("harbor", dirpath=tmp_path)[0][0] == \
        "Pearl Harbor"
    assert ldb.search_places("", dirpath=tmp_path) == []


def test_bundled_gnis_catalog_present():
    """The repo ships a GNIS national extract; Wake must be findable."""
    hits = ldb.search_places("wake atoll")
    assert any(h[0] == "Wake Atoll" and h[3] == "UM" for h in hits)
    hits = ldb.search_places("vandenberg")
    assert any("Vandenberg" in h[0] for h in hits)
