"""Gazetteer map overlays (TODO 3e design of record): class tables,
zoom gates, corridor bonus, grid spread, label deconfliction, payload.

House style: identities against the stated rule sets — the tables in
map_overlays.py ARE the design, so the tests pin representative rows of
each rather than re-deriving them."""

import gzip
import sqlite3

import pytest

import gazetteer as gz
import map_overlays as mo


# ── fixture: a small baked index with variants ─────────────────────────────
@pytest.fixture
def db(tmp_path):
    d = tmp_path / "packs"
    d.mkdir()
    lines = [
        # capital (rank 0), a town (4), a district seat (2), an admin-1
        # seat (1) — all GNS populated codes
        "GNS:1|P'yŏngyang|Pyongyang;Heijo|39.01950|125.75470|KP|PPLC|GNS-P",
        "GNS:2|Sŏhae|Sohae;Tongch'ang-ni;Dongchang-ri|39.66000|124.70500"
        "|KP|PPL|GNS-P",
        "GNS:3|Ch'ŏlsan|Cholsan|39.80000|124.53000|KP|PPLA2|GNS-P",
        "GNS:4|Sinŭiju|Sinuiju;Shingishu|40.10000|124.40000|KP|PPLA|GNS-P",
        # facilities near the launch end of the fixture track
        "GNS:5|Sŏhae Satellite Launching Station|Sohae|39.66000|124.70000"
        "|KP|INSM|GNS-S",
        # water / undersea near the impact end
        "GNS:6|Korea Bay|Sŏjosŏn-man|39.20000|124.30000|KP|BAY|GNS-H",
        "GNS:7|Yellow Sea|Hwang Hae;Huang Hai|38.90000|123.90000|KP|SEA"
        "|GNS-H",
        "GNS:8|East Deep|Tong Trough|40.10000|126.20000|KP|TRGU|GNS-U",
        # a class outside the five families → never drawn
        "GNS:9|Somefarm||39.50000|125.00000|KP|FRM|GNS-S",
        # GNIS + Antarctic vocabularies
        "GNIS:10|Kodiak||57.79000|-152.40000|Alaska|Populated Place|GNIS",
        "ATA:11|Fildes Peninsula||-62.20000|-59.00000|Antarctica"
        "|Peninsula|BGN-Antarctic",
    ]
    with gzip.open(d / "fx.txt.gz", "wt", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return gz.ensure_index(pack_dir=str(d), db_path=str(tmp_path / "g.db"))


BBOX = (38.5, 40.5, 123.5, 126.5)
TRACK = [(39.0, 124.0), (39.5, 125.0), (40.0, 126.0)]
LAUNCH, IMPACT = TRACK[0], TRACK[-1]


# ── class tables ────────────────────────────────────────────────────────────
def test_classify_pins_the_rank_tables():
    assert mo.classify("PPLC", "GNS-P") == ("populated", 0)
    assert mo.classify("PPLA", "GNS-P") == ("populated", 1)
    assert mo.classify("PPL", "GNS-P") == ("populated", 4)
    assert mo.classify("AIRB", "GNS-S") == ("facilities", 1)
    assert mo.classify("SEA", "GNS-H") == ("water", 0)
    assert mo.classify("STM", "GNS-H") == ("water", 4)
    assert mo.classify("TRGU", "GNS-U") == ("undersea", 1)
    assert mo.classify("Populated Place", "GNIS") == ("populated", 4)
    assert mo.classify("Peninsula", "BGN-Antarctic") == ("terrain", 1)
    # deliberate gate, not an oversight: farm-tier classes never appear
    assert mo.classify("FRM", "GNS-S") is None
    assert mo.classify("Civil", "GNIS") is None


def test_eligible_fclasses_track_the_gates():
    cont = mo.eligible_fclasses("continental", corridor=False)
    assert "PPLC" in cont and "SEA" in cont
    assert "PPLA" not in cont                     # rank 1 > gate 0
    # corridor headroom: fetch keeps everything promotable (≤ gate+2)
    cont_c = mo.eligible_fclasses("continental", corridor=True)
    assert "PPLA" in cont_c and "PPLA2" in cont_c
    assert "PPL" not in cont_c                    # rank 4 unreachable
    assert "FRM" not in mo.eligible_fclasses("local")


# ── corridor bonus ──────────────────────────────────────────────────────────
def test_corridor_bonus_geometry():
    near = dict(lat=39.05, lon=124.05)            # ~7 km off the track
    far = dict(lat=42.5, lon=124.0)               # ~330 km north
    assert mo.corridor_bonus(near, "populated", TRACK) == 1
    assert mo.corridor_bonus(far, "populated", TRACK) == 0
    assert mo.corridor_bonus(near, "populated", None) == 0
    # impact zone prefers water/undersea; launch area prefers facilities
    at_impact = dict(lat=40.05, lon=126.05)
    assert mo.corridor_bonus(at_impact, "water", TRACK,
                             impact=IMPACT) == 2
    assert mo.corridor_bonus(at_impact, "populated", TRACK,
                             impact=IMPACT) == 1
    at_launch = dict(lat=39.05, lon=124.05)
    assert mo.corridor_bonus(at_launch, "facilities", TRACK,
                             launch=LAUNCH) == 2
    # capped at two tiers even when every term fires
    assert mo.corridor_bonus(at_launch, "facilities", TRACK,
                             launch=LAUNCH, impact=at_launch) == 2


# ── density pipeline ────────────────────────────────────────────────────────
def _feat(name, lat, lon, fclass, source, nvar=0):
    return dict(primary=name, lat=lat, lon=lon, fclass=fclass,
                source=source, nvar=nvar)


def test_eligibility_gates_by_zoom_bucket():
    town = [_feat("Sŏhae", 39.66, 124.7, "PPL", "GNS-P")]
    assert not mo.select(town, BBOX, "continental")["populated"]
    assert mo.select(town, BBOX, "local")["populated"]


def test_corridor_promotes_one_tier_into_the_bucket():
    seat = [_feat("Sinŭiju", 40.0, 126.0, "PPLA", "GNS-P")]  # on the track
    assert not mo.select(seat, BBOX, "continental")["populated"]
    got = mo.select(seat, BBOX, "continental", track=TRACK)
    assert [c["name"] for c in got["populated"]] == ["Sinŭiju"]


def test_grid_keeps_the_best_per_cell_and_spread_survives():
    a = _feat("Lowtown", 39.01, 124.51, "PPLA2", "GNS-P", nvar=0)
    b = _feat("Fameville", 39.02, 124.52, "PPLA2", "GNS-P", nvar=6)
    c = _feat("Elsewhere", 40.31, 126.01, "PPLA2", "GNS-P", nvar=0)
    got = mo.select([a, b, c], BBOX, "regional")["populated"]
    names = {x["name"] for x in got}
    assert names == {"Fameville", "Elsewhere"}   # same cell → nvar wins;
    # ...while intrinsic top tiers (capitals) BYPASS the grid entirely
    caps = [_feat("Cap1", 39.01, 124.51, "PPLC", "GNS-P"),
            _feat("Cap2", 39.02, 124.52, "PPLC", "GNS-P")]
    got = mo.select(caps, BBOX, "regional")["populated"]
    assert {x["name"] for x in got} == {"Cap1", "Cap2"}


def test_labels_budget_and_deconfliction():
    seas = [_feat(f"Sea {i}", 38.6 + 0.2 * i, 123.6 + 0.35 * i,
                  "SEA", "GNS-H") for i in range(6)]
    got = mo.select(seas, BBOX, "continental")["water"]
    assert len(got) == 6                          # every dot survives
    n_lab = sum(c["labeled"] for c in got)
    assert 1 <= n_lab <= 6 // mo.LABEL_FRACTION   # budget ≈ ⅓ of dots
    # the placed label boxes never overlap (recomputed from the output)
    zrep = 4
    ppd_lon, ppd_lat = mo._px_per_deg(zrep, 39.5)
    boxes = []
    for c in got:
        if not c["labeled"]:
            continue
        w, h = 7.0 * len(c["name"]) + 12.0, 16.0
        x = c["lon"] * ppd_lon + 8.0
        y = -c["lat"] * ppd_lat - h / 2.0
        boxes.append((x, y, x + w, y + h))
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            assert (a[2] < b[0] or a[0] > b[2]
                    or a[3] < b[1] or a[1] > b[3])


# ── fetch layer ─────────────────────────────────────────────────────────────
def test_wrap_intervals_handle_the_antimeridian():
    assert mo._wrap_intervals(-10.0, 10.0) == [(-10.0, 10.0, 0.0)]
    two = mo._wrap_intervals(170.0, 190.0)
    assert (170.0, 180.0, 0.0) in two
    assert (-180.0, -170.0, 360.0) in two
    assert mo._wrap_intervals(0.0, 400.0) == [(-180.0, 180.0, 0.0)]


def test_features_in_bbox_counts_variants_and_flags_truncation(db):
    rows, trunc = gz.features_in_bbox(38.5, 40.5, 123.5, 126.5, db=db)
    assert not trunc
    by = {r["primary"]: r for r in rows}
    assert by["Sŏhae"]["nvar"] == 3               # the variants ship
    assert by["P'yŏngyang"]["nvar"] == 2
    # fclass filter is verbatim
    only, _ = gz.features_in_bbox(38.5, 40.5, 123.5, 126.5,
                                  fclasses={"SEA"}, db=db)
    assert [r["primary"] for r in only] == ["Yellow Sea"]
    # a hit cap is reported, never silent
    _rows, trunc = gz.features_in_bbox(38.5, 40.5, 123.5, 126.5,
                                       limit=2, db=db)
    assert trunc


def test_old_index_without_nvar_degrades_gracefully():
    old = sqlite3.connect(":memory:")
    old.executescript("""
        CREATE TABLE places(
            id INTEGER PRIMARY KEY, ext_id TEXT, primary_name TEXT,
            lat REAL, lon REAL, admin TEXT, fclass TEXT, source TEXT);
        CREATE TABLE meta(k TEXT PRIMARY KEY, v TEXT);
        INSERT INTO places VALUES(1,'GNS:1','X',10.0,20.0,'','PPLC',
                                  'GNS-P');
    """)
    assert not gz.has_variant_counts(old)
    rows, _ = gz.features_in_bbox(9, 11, 19, 21, db=old)
    assert rows[0]["nvar"] == 0                   # served, bonus-free


def test_fetch_features_unwraps_across_the_antimeridian(tmp_path):
    d = tmp_path / "packs"
    d.mkdir()
    with gzip.open(d / "p.txt.gz", "wt", encoding="utf-8") as f:
        f.write("GNS:1|Adak||51.80000|-176.60000|US|PPLC|GNS-P\n")
    db2 = gz.ensure_index(pack_dir=str(d), db_path=str(tmp_path / "g.db"))
    # a track crossing 180°E: unwrapped lons run past +180
    feats, trunc = mo.fetch_features(db2, (50.0, 53.0, 175.0, 185.0),
                                     "continental")
    assert not trunc and len(feats) == 1
    assert feats[0]["lon"] == pytest.approx(183.4)   # shifted into frame


# ── payload ─────────────────────────────────────────────────────────────────
def test_build_payload_is_corridor_aware_and_honest(db):
    lats, lons = [39.0, 39.5, 40.0], [124.0, 125.0, 126.0]
    pay = mo.build_payload(db, lats, lons)
    assert pay["flags"] == []                     # fresh index: nvar there
    assert [b["name"] for b in pay["buckets"]] == [
        "continental", "country", "regional", "local"]
    cont = pay["buckets"][0]["families"]
    # capital + sea are intrinsically continental
    assert any(d["n"] == "P'yŏngyang" for d in cont["populated"])
    assert any(d["n"] == "Yellow Sea" for d in cont["water"])
    # the launch-area facility and impact-zone trough are PROMOTED in
    assert any("Launching" in d["n"] for d in cont["facilities"])
    assert any(d["n"] == "East Deep" for d in cont["undersea"])
    # ...and stay out when the corridor term is off (neutral maps)
    neutral = mo.build_payload(db, lats, lons, corridor=False)
    ncont = neutral["buckets"][0]["families"]
    assert not ncont["facilities"] and not ncont["undersea"]
    # the farm never appears anywhere at any zoom
    for b in pay["buckets"]:
        for dots in b["families"].values():
            assert all(d["n"] != "Somefarm" for d in dots)
    # every entry carries the label decision the JS renders
    assert all(d["lb"] in (0, 1)
               for b in pay["buckets"]
               for dots in b["families"].values() for d in dots)
