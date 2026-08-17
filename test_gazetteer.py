"""Bundled gazetteer: packs, index, variant-aware search, nearest.

House rules: identities on the round-trips (what goes into a pack comes
back out), the variants requirement is pinned (romanization differences
MUST match), and the shipped packs are exercised for real."""

import gzip
import io
import json
import os
import zipfile

import pytest

import gazetteer as gz
import gazetteer_build as gb


# ── fixture pack ────────────────────────────────────────────────────────────
@pytest.fixture
def fixture_env(tmp_path):
    d = tmp_path / "packs"
    d.mkdir()
    lines = [
        "GNS:1|Sŏhae|Sohae;Tongch'ang-ni;Dongchang-ri|39.66000|124.70500"
        "|KP|PPL|GNS",
        "GNS:2|P'yŏngyang|Pyongyang|39.01950|125.75470|KP|PPLC|GNS",
        "GNIS:3|Kodiak|"
        "|57.79000|-152.40000|Alaska|Populated Place|GNIS",
        "GNIS:4|Kodiak Island||57.40000|-153.30000|Alaska|Island|GNIS",
    ]
    with gzip.open(d / "fixture.txt.gz", "wt", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    db = gz.ensure_index(pack_dir=str(d), db_path=str(tmp_path / "gaz.db"))
    return d, db


def test_normalize_folds_romanization_differences():
    assert gz.normalize("Sŏhae") == "sohae"
    assert gz.normalize("Tongch'ang-ni") == "tongchang ni"
    assert gz.normalize("P'yŏngyang") == "pyongyang"
    assert gz.normalize("  Foo   Bar ") == "foo bar"


def test_search_matches_variants_and_shows_the_primary(fixture_env):
    """The variants requirement: 'sohae' (any romanization) finds the
    feature; the display name is ALWAYS the BGN primary, the matched
    variant reported alongside — never silently swapped."""
    _d, db = fixture_env
    r = gz.search("sohae", db=db)
    assert r and r[0]["primary"] == "Sŏhae"
    r = gz.search("dongchang", db=db)
    assert r and r[0]["primary"] == "Sŏhae"
    assert r[0]["matched"] == "Dongchang-ri"
    r = gz.search("pyongyang", db=db)
    assert r and r[0]["primary"] == "P'yŏngyang"
    assert r[0]["ext_id"] == "GNS:2"          # traceable to the source


def test_search_ranks_exact_before_substring_and_populated_first(fixture_env):
    _d, db = fixture_env
    r = gz.search("kodiak", db=db)
    assert [x["primary"] for x in r[:2]] == ["Kodiak", "Kodiak Island"]
    assert r[0]["fclass"] == "Populated Place"


def test_nearest_reports_distance(fixture_env):
    _d, db = fixture_env
    r = gz.nearest(39.6, 124.7, n=1, db=db)
    assert r[0]["primary"] == "Sŏhae"
    assert r[0]["km"] < 10.0


def test_no_packs_degrades_to_empty(tmp_path):
    empty = tmp_path / "none"
    empty.mkdir()
    assert not gz.available(str(empty))
    assert gz.search("anything", pack_dir=str(empty)) == []


def test_index_rebuilds_when_packs_change(fixture_env, tmp_path):
    d, db = fixture_env
    assert gz.search("kodiak", db=db)
    with gzip.open(d / "extra.txt.gz", "wt", encoding="utf-8") as f:
        f.write("GNS:9|Testville||1.00000|2.00000|XX|PPL|GNS\n")
    db2 = gz.ensure_index(pack_dir=str(d),
                          db_path=str(tmp_path / "gaz.db"))
    assert gz.search("testville", db=db2)


# ── builder round-trips ─────────────────────────────────────────────────────
def test_gns_builder_groups_names_by_feature(tmp_path):
    """The Phase-2 hook, proven before the data arrives: GNS rows are
    one-per-NAME; the bake groups by UFI, takes the BGN-approved row
    (NT='N') as primary and keeps everything else as variants."""
    rows = [
        "UFI\tUNI\tLAT\tLONG\tCC1\tNT\tDSG\tFULL_NAME_RO",
        "100\t1\t39.66\t124.705\tKP\tN\tPPL\tSŏhae",
        "100\t2\t39.66\t124.705\tKP\tV\tPPL\tTongch'ang-ni",
        "100\t3\t39.66\t124.705\tKP\tV\tPPL\tDongchang-ri",
        "200\t4\t39.0195\t125.7547\tKP\tN\tPPLC\tP'yŏngyang",
    ]
    zp = tmp_path / "gns.zip"
    with zipfile.ZipFile(zp, "w") as z:
        z.writestr("Countries.txt", "\n".join(rows) + "\n")
    out = tmp_path / "gns_pp.txt.gz"
    n = gb.build_gns(str(zp), str(out))
    assert n == 2
    got = {ln.split("|")[0]: ln.split("|")
           for ln in gzip.open(out, "rt", encoding="utf-8")}
    assert got["GNS:100"][1] == "Sŏhae"
    assert set(got["GNS:100"][2].split(";")) == {"Tongch'ang-ni",
                                                 "Dongchang-ri"}
    assert got["GNS:200"][1] == "P'yŏngyang" and got["GNS:200"][2] == ""
    # and the pack is immediately searchable by any variant
    db = gz.ensure_index(pack_dir=str(tmp_path),
                         db_path=str(tmp_path / "g.db"))
    assert gz.search("dongchang", db=db)[0]["primary"] == "Sŏhae"


def test_gnis_builder_skips_unlocated_rows(tmp_path):
    hdr = ("feature_id|feature_name|feature_class|state_name|state_numeric|"
           "county_name|county_numeric|map_name|date_created|date_edited|"
           "bgn_type|bgn_authority|bgn_date|prim_lat_dms|prim_long_dms|"
           "prim_lat_dec|prim_long_dec|source_lat_dms|source_long_dms|"
           "source_lat_dec|source_long_dec")
    mk = lambda fid, nm, lat, lon: (f"{fid}|{nm}|Populated Place|Alaska|02|"
                                    f"||||||||||{lat}|{lon}||||0.0")
    zp = tmp_path / "DomesticNames_AllStates_Text.zip"
    with zipfile.ZipFile(zp, "w") as z:
        z.writestr("Text/DomesticNames_AK.txt", "\n".join(
            [hdr, mk(1, "Realtown", "57.79", "-152.4"),
             mk(2, "Nowhere", "0.0", "0.0")]) + "\n")
    out = tmp_path / "gnis_us.txt.gz"
    assert gb.build_gnis(str(zp), str(out)) == 1
    (line,) = list(gzip.open(out, "rt", encoding="utf-8"))
    assert line.split("|")[1] == "Realtown"


# ── the SHIPPED packs ───────────────────────────────────────────────────────
def test_shipped_packs_exist_and_serve_search(tmp_path):
    """data/gazetteer must actually work: both Phase-1 packs present,
    Kodiak (GNIS) and an Antarctic variant name both resolve."""
    names = {os.path.basename(p) for p in gz.packs()}
    assert {"gnis_us.txt.gz", "antarctica.txt.gz"} <= names
    db = gz.ensure_index(db_path=str(tmp_path / "real.db"))
    r = gz.search("Kodiak", db=db)
    assert r and r[0]["source"] == "GNIS"
    r = gz.search("Annenkow", db=db)      # German variant → official name
    assert r and r[0]["primary"] == "Annenkov Island"
    assert r[0]["matched"] != r[0]["primary"]
    assert r[0]["ext_id"].startswith("ATA:")
