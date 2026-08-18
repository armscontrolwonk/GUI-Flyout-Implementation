"""Gazetteer map overlays — the design of record from TODO 3e (agreed
2026-08-17, trajectory-aware by default on the trajectory map).

Turns the baked gazetteer into toggleable per-class dot / dot+label
layers for the folium trajectory map and the cartopy location picker.
Three rule sets, all stated here (derive, don't invent — the tables ARE
the documentation, nothing is scored on data the packs don't carry):

PRIORITY (lower = more prominent), per feature:
  designation rank from the class tables below (PPLC > PPLA > PPLA2 >
  PPL; SEA > GULF > BAY > STM; AIRB/CTRS > INSM > generic)
  − corridor bonus (trajectory maps only): ≈1 tier within ~100 km of
    the ground track; +1 more for Water/Undersea within ~150 km of the
    impact point, and for Facilities within ~150 km of the launch point
  − a variant-name-count bonus, capped, for ORDERING only (prominence
    proxy: places the world writes about accumulate romanizations —
    the packs carry no population data, so this is the honest stand-in)

DENSITY, per zoom bucket:
  zoom-tiered class gates (the GATES table) decide ELIGIBILITY →
  best-per-grid-cell (GRID_NX × GRID_NY over the view bbox; effective
  rank ≤ 1 bypasses the grid) guarantees SPREAD → greedy label-box
  deconfliction in priority order decides LABELS (the dot stays when
  its label is culled; label budget ≈ ⅓ of the dot budget).

CLASS FAMILIES: Populated, Facilities (GNS-S), Water (GNS-H),
Terrain/Islands (GNS-T), Undersea (GNS-U); GNIS and BGN-Antarctic
classes are mapped into the same five families.  Classes not in the
tables (farms, schools, springs, …) never appear at ANY zoom — a
deliberate, visible gate, not a silent cap; genuine caps (a bbox fetch
hitting its row limit) are flagged in the payload, never swallowed.

Rejected alternative (recorded 2026-08-17): folium MarkerCluster —
count-bubbles hide exactly the names the reader needs.

Pure module: no Tk, no folium, no cartopy imports here."""

import math

import gazetteer as gz

# ── class families ──────────────────────────────────────────────────────────
# key → (display label, dot colour)
FAMILIES = {
    "populated":  ("Populated places",  "#4d4d4d"),
    "facilities": ("Facilities",        "#b2182b"),
    "water":      ("Water",             "#2166ac"),
    "terrain":    ("Terrain / islands", "#8c510a"),
    "undersea":   ("Undersea",          "#35978f"),
}

# Designation-rank tables.  Rank semantics (the zoom tiers they unlock):
#   0 continental · 1 country · 2–3 regional · 4–5 local
# GNS keys are NGA designation codes; GNIS/Antarctic keys are the class
# names those sources use verbatim.
_GNS_RANKS = {
    # populated (GNS-P)
    "PPLC": ("populated", 0), "PPLA": ("populated", 1),
    "PPLA2": ("populated", 2), "PPLA3": ("populated", 3),
    "PPLA4": ("populated", 3), "PPL": ("populated", 4),
    "PPLX": ("populated", 5), "PPLL": ("populated", 5),
    "PPLQ": ("populated", 5), "PPLF": ("populated", 5),
    "PPLW": ("populated", 5), "STLMT": ("populated", 5),
    # facilities (GNS-S; PRT rides in GNS-L but is a facility)
    "CTRS": ("facilities", 1), "AIRB": ("facilities", 1),
    "INSM": ("facilities", 2), "AIRP": ("facilities", 2),
    "STNS": ("facilities", 3), "FT": ("facilities", 3),
    "AIRF": ("facilities", 3), "DAM": ("facilities", 3),
    "PRT": ("facilities", 3), "STNR": ("facilities", 4),
    "STNM": ("facilities", 4), "PSTB": ("facilities", 4),
    "RSTN": ("facilities", 4), "OBPT": ("facilities", 4),
    # water (GNS-H)
    "OCN": ("water", 0), "SEA": ("water", 0), "GULF": ("water", 1),
    "STRT": ("water", 1), "CHN": ("water", 2), "BAY": ("water", 2),
    "LGN": ("water", 3), "LK": ("water", 3), "HBR": ("water", 3),
    "RSV": ("water", 4), "COVE": ("water", 4), "STM": ("water", 4),
    "ANCH": ("water", 4), "STMI": ("water", 5),
    # terrain / islands (GNS-T)
    "PEN": ("terrain", 1), "MTS": ("terrain", 1), "DSRT": ("terrain", 2),
    "VLC": ("terrain", 2), "ISL": ("terrain", 2), "ISLS": ("terrain", 2),
    "PLAT": ("terrain", 2), "CAPE": ("terrain", 3), "MT": ("terrain", 3),
    "PK": ("terrain", 3), "PASS": ("terrain", 3), "RDGE": ("terrain", 4),
    "VAL": ("terrain", 4), "PT": ("terrain", 4), "HLLS": ("terrain", 4),
    "HLL": ("terrain", 5), "RK": ("terrain", 5), "SPUR": ("terrain", 5),
    "SLP": ("terrain", 5), "CLF": ("terrain", 5), "DPR": ("terrain", 5),
    # undersea (GNS-U)
    "TRGU": ("undersea", 1), "TRNU": ("undersea", 1),
    "RDGU": ("undersea", 2), "BSNU": ("undersea", 2),
    "PLNU": ("undersea", 2), "SMU": ("undersea", 3),
    "CNYU": ("undersea", 3), "BNKU": ("undersea", 3),
    "TMTU": ("undersea", 3), "KNLU": ("undersea", 4),
    "FURU": ("undersea", 4), "VALU": ("undersea", 4),
    "RFU": ("undersea", 4), "SHLU": ("undersea", 4),
}
_GNIS_RANKS = {
    "Populated Place": ("populated", 4),
    "Military": ("facilities", 2), "Airport": ("facilities", 3),
    "Locale": ("facilities", 5),
    "Sea": ("water", 0), "Gulf": ("water", 1), "Bay": ("water", 2),
    "Channel": ("water", 3), "Lake": ("water", 4),
    "Reservoir": ("water", 5), "Stream": ("water", 5),
    "Harbor": ("water", 3), "Lagoon": ("water", 4),
    "Range": ("terrain", 2), "Island": ("terrain", 2),
    "Cape": ("terrain", 3), "Summit": ("terrain", 4),
    "Ridge": ("terrain", 4), "Valley": ("terrain", 5),
    "Basin": ("terrain", 5), "Beach": ("terrain", 5),
    "Cliff": ("terrain", 5),
}
_ATA_RANKS = {
    "Station": ("facilities", 2), "Camp": ("facilities", 4),
    "Sea": ("water", 0), "Gulf": ("water", 1), "Bay": ("water", 3),
    "Strait": ("water", 2), "Lake": ("water", 4),
    "Island": ("terrain", 2), "Islands": ("terrain", 2),
    "Peninsula": ("terrain", 1), "Cape": ("terrain", 3),
    "Mountain": ("terrain", 3), "Range": ("terrain", 2),
    "Summit": ("terrain", 4), "Glacier": ("terrain", 3),
    "Ridge": ("terrain", 4), "Point": ("terrain", 4),
}

# Zoom buckets: (name, leaflet zmin, leaflet zmax, representative zoom
# for label-box pixel math).  The gate is the maximum effective rank a
# feature may carry to be ELIGIBLE in that bucket.
ZOOM_BUCKETS = (("continental", 0, 4, 4), ("country", 5, 6, 6),
                ("regional", 7, 9, 8), ("local", 10, 19, 11))
GATES = {"continental": 0, "country": 1, "regional": 3, "local": 5}

GRID_NX, GRID_NY = 12, 8
CORRIDOR_KM = 100.0          # ground-track promotion band
ENDPOINT_KM = 150.0          # launch/impact preference band
VARIANT_BONUS_CAP = 8        # ordering bonus caps here
LABEL_FRACTION = 3           # label budget = dots // 3 (min 1)


def classify(fclass, source):
    """(family, rank) for a gazetteer feature, or None when the class
    is outside the five families (admin units, vegetation, farms, …)."""
    src = source or ""
    if src.startswith("GNS"):
        return _GNS_RANKS.get(fclass)
    if src.startswith("GNIS"):
        return _GNIS_RANKS.get(fclass)
    if src.startswith("BGN"):
        return _ATA_RANKS.get(fclass)
    return None


def eligible_fclasses(bucket, families=None, corridor=True):
    """The fclass strings worth FETCHING for a bucket: every class whose
    table rank could pass the gate after the maximum corridor promotion
    (2 tiers: track + endpoint).  This is the SQL prefilter that keeps
    bbox queries bounded; exact eligibility is re-applied per feature."""
    gate = GATES[bucket] + (2 if corridor else 0)
    fams = set(families or FAMILIES)
    out = set()
    for table in (_GNS_RANKS, _GNIS_RANKS, _ATA_RANKS):
        for fc, (fam, rank) in table.items():
            if fam in fams and rank <= gate:
                out.add(fc)
    return out


def _haversine_km(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(math.radians, (la1, lo1, la2, lo2))
    a = (math.sin((la2 - la1) / 2) ** 2
         + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2)
    return 6371.0 * 2 * math.asin(min(1.0, math.sqrt(a)))


def downsample_track(lats, lons, n=240):
    """≤ n (lat, lon) samples along the track — enough that the min
    point distance approximates distance-to-track well inside the
    100 km corridor threshold."""
    m = len(lats)
    if m <= n:
        return [(float(lats[i]), float(lons[i])) for i in range(m)]
    idx = [round(i * (m - 1) / (n - 1)) for i in range(n)]
    return [(float(lats[i]), float(lons[i])) for i in idx]


def _min_track_km(lat, lon, track):
    best = float("inf")
    for tla, tlo in track:
        # cheap reject: 1° lat ≈ 111 km
        if abs(tla - lat) * 111.0 - 1.0 > best:
            continue
        d = _haversine_km(lat, lon, tla, tlo)
        if d < best:
            best = d
    return best


def corridor_bonus(feat, family, track=None, launch=None, impact=None):
    """Tier promotion (0–2) for the trajectory map: near the ground
    track everything gains one tier; the impact zone prefers water and
    undersea names, the launch area prefers facilities."""
    if not track:
        return 0
    bonus = 0
    if _min_track_km(feat["lat"], feat["lon"], track) <= CORRIDOR_KM:
        bonus += 1
    if (impact is not None and family in ("water", "undersea")
            and _haversine_km(feat["lat"], feat["lon"],
                              impact[0], impact[1]) <= ENDPOINT_KM):
        bonus += 1
    if (launch is not None and family == "facilities"
            and _haversine_km(feat["lat"], feat["lon"],
                              launch[0], launch[1]) <= ENDPOINT_KM):
        bonus += 1
    return min(bonus, 2)


def _score(eff_rank, rank, nvar):
    """Ordering priority (lower wins): effective rank first, then the
    capped variant-count prominence proxy, then INTRINSIC rank — so at
    equal effective tier a real capital out-labels a corridor-promoted
    provincial seat even on an index without variant counts."""
    return (eff_rank - 0.1 * min(nvar or 0, VARIANT_BONUS_CAP)
            + 0.02 * rank)


def family_fclasses(family):
    """Every fclass string the tables map into a family — the per-family
    SQL prefilter for the one-shot fetch."""
    return {fc for tbl in (_GNS_RANKS, _GNIS_RANKS, _ATA_RANKS)
            for fc, (fam, _r) in tbl.items() if fam == family}


def _annotate(feats, track=None, launch=None, impact=None):
    """Classify + corridor-promote every feature ONCE (the expensive
    part — bucket selection then just gates on the result).  Corridor
    distance is vectorised equirectangular (fine at the 100 km band;
    lons are frame-consistent by construction).  Returns dicts with
    family, rank, eff, _s added; unclassified features drop out."""
    import numpy as np
    ann = []
    for ft in feats:
        cl = classify(ft["fclass"], ft["source"])
        if cl is None:
            continue
        ann.append(dict(ft, family=cl[0], rank=cl[1]))
    if not ann:
        return ann
    if track:
        fla = np.array([a["lat"] for a in ann])
        flo = np.array([a["lon"] for a in ann])
        tla = np.array([t[0] for t in track])
        tlo = np.array([t[1] for t in track])
        dmin = np.full(len(ann), np.inf)
        for i0 in range(0, len(ann), 4000):        # bound the N×T matrix
            sl = slice(i0, i0 + 4000)
            midc = np.cos(np.radians(
                0.5 * (fla[sl, None] + tla[None, :])))
            dx = (flo[sl, None] - tlo[None, :]) * midc * 111.32
            dy = (fla[sl, None] - tla[None, :]) * 111.32
            dmin[sl] = np.sqrt(dx * dx + dy * dy).min(axis=1)

        def _pt_km(pt):
            c = np.cos(np.radians(0.5 * (fla + pt[0])))
            return np.hypot((flo - pt[1]) * c * 111.32,
                            (fla - pt[0]) * 111.32)
        dlaunch = _pt_km(launch) if launch is not None else None
        dimpact = _pt_km(impact) if impact is not None else None
        for i, a in enumerate(ann):
            bonus = int(dmin[i] <= CORRIDOR_KM)
            if (dimpact is not None and a["family"] in ("water", "undersea")
                    and dimpact[i] <= ENDPOINT_KM):
                bonus += 1
            if (dlaunch is not None and a["family"] == "facilities"
                    and dlaunch[i] <= ENDPOINT_KM):
                bonus += 1
            a["eff"] = max(0, a["rank"] - min(bonus, 2))
    else:
        for a in ann:
            a["eff"] = a["rank"]
    for a in ann:
        a["_s"] = _score(a["eff"], a["rank"], a.get("nvar", 0))
    return ann


def _px_per_deg(zoom, mid_lat):
    """Web-Mercator pixels per degree at a zoom level (lon exact; lat
    scaled by the local Mercator stretch, clamped near the poles)."""
    ppd_lon = 256.0 * (2 ** zoom) / 360.0
    c = max(0.1, math.cos(math.radians(mid_lat)))
    return ppd_lon, ppd_lon / c


def select(feats, bbox, bucket, families=None, track=None, launch=None,
           impact=None):
    """The full density pipeline for one zoom bucket: eligibility gate →
    best-per-grid-cell (top tiers bypass) → greedy label deconfliction.
    feats: dicts with lat, lon, primary, fclass, source, nvar (raw), or
    already _annotate()d — annotation is computed here if absent so the
    per-bucket callers stay cheap.  bbox: (lat0, lat1, lon0, lon1) in
    the map's (possibly unwrapped) longitude frame.  Returns per-family
    lists of {name, lat, lon, fclass, labeled} sorted by priority."""
    if feats and "eff" not in feats[0]:
        feats = _annotate(feats, track, launch, impact)
    lat0, lat1, lon0, lon1 = bbox
    gate = GATES[bucket]
    fams = list(families or FAMILIES)
    dlat = max(1e-9, lat1 - lat0)
    dlon = max(1e-9, lon1 - lon0)

    kept = {f: {} for f in fams}          # family → cell → best candidate
    bypass = {f: [] for f in fams}
    for ft in feats:
        family = ft["family"]
        if family not in kept or ft["eff"] > gate:
            continue
        cand = dict(name=ft["primary"], lat=ft["lat"], lon=ft["lon"],
                    fclass=ft["fclass"], _s=ft["_s"])
        # Grid bypass is for INTRINSIC top tiers (capitals, seas) only —
        # a corridor promotion earns eligibility, not exemption from the
        # spread guarantee, else a long track over dense country floods
        # the map with promoted district seats.
        if ft["rank"] <= 1:
            bypass[family].append(cand)
            continue
        cx = min(GRID_NX - 1, int((ft["lon"] - lon0) / dlon * GRID_NX))
        cy = min(GRID_NY - 1, int((ft["lat"] - lat0) / dlat * GRID_NY))
        cell = kept[family].get((cx, cy))
        if cell is None or cand["_s"] < cell["_s"]:
            kept[family][(cx, cy)] = cand

    zrep = next(zr for nm, _a, _b, zr in ZOOM_BUCKETS if nm == bucket)
    ppd_lon, ppd_lat = _px_per_deg(zrep, 0.5 * (lat0 + lat1))
    placed = []                            # label boxes across ALL families

    out = {}
    for family in fams:
        dots = sorted(bypass[family] + list(kept[family].values()),
                      key=lambda c: (c["_s"], c["name"]))
        budget = max(1, len(dots) // LABEL_FRACTION) if dots else 0
        n_lab = 0
        for c in dots:
            c["labeled"] = False
            if n_lab >= budget:
                continue
            w = 7.0 * len(c["name"]) + 12.0          # px, ~11 px font
            h = 16.0
            x = c["lon"] * ppd_lon + 8.0             # box right of the dot
            y = -c["lat"] * ppd_lat - h / 2.0
            box = (x, y, x + w, y + h)
            if any(not (box[2] < b[0] or box[0] > b[2]
                        or box[3] < b[1] or box[1] > b[3])
                   for b in placed):
                continue                             # culled: dot stays
            placed.append(box)
            c["labeled"] = True
            n_lab += 1
        for c in dots:
            del c["_s"]
        out[family] = dots
    return out


# ── fetch + payload ─────────────────────────────────────────────────────────
def _wrap_intervals(lon0, lon1):
    """Split an (possibly unwrapped, possibly >360°-spanning) longitude
    interval into ≤ 3 [-180, 180] query intervals, each with the offset
    that maps its features back into the caller's frame."""
    if lon1 - lon0 >= 360.0:
        return [(-180.0, 180.0, 0.0)]
    out = []
    for k in (-360.0, 0.0, 360.0):
        a, b = lon0 + k, lon1 + k
        lo, hi = max(a, -180.0), min(b, 180.0)
        if lo < hi:
            out.append((lo, hi, -k))
    return out


def fetch_features(db, bbox, bucket, families=None, corridor=True,
                   limit=25000):
    """One bucket's candidate features from the gazetteer index, with
    antimeridian handling: features come back with lon shifted into the
    bbox's longitude frame.  Returns (feats, truncated)."""
    lat0, lat1, lon0, lon1 = bbox
    fcl = eligible_fclasses(bucket, families, corridor)
    feats, truncated = [], False
    for lo, hi, shift in _wrap_intervals(lon0, lon1):
        rows, trunc = gz.features_in_bbox(lat0, lat1, lo, hi,
                                          fclasses=fcl, limit=limit, db=db)
        truncated = truncated or trunc
        if shift:
            rows = [dict(r, lon=r["lon"] + shift) for r in rows]
        feats.extend(rows)
    return feats, truncated


def track_bbox(lats, lons, pad_deg=None):
    """The overlay bbox around a ground track (unwrapped lons), padded
    so corridor-band features just outside the track extent appear."""
    la0, la1 = float(min(lats)), float(max(lats))
    lo0, lo1 = float(min(lons)), float(max(lons))
    pad = pad_deg if pad_deg is not None else max(
        1.5, 0.05 * max(la1 - la0, lo1 - lo0))
    return (max(-90.0, la0 - pad), min(90.0, la1 + pad),
            lo0 - pad, lo1 + pad)


def build_payload(db, lats, lons, families=None, corridor=True,
                  limit=25000):
    """Everything the folium map's overlay JS needs, precomputed per
    zoom bucket: dots + label decisions per family, plus honest flags
    (fetch truncation, missing variant counts).  lons are the map's
    UNWRAPPED longitudes; launch = first sample, impact = last.
    One bbox fetch PER FAMILY (each with its own row cap, so a flood of
    streams can never crowd out the towns' budget), one classification
    + corridor pass per feature; the four zoom buckets then just gate,
    grid and label the shared annotated set."""
    bbox = track_bbox(lats, lons)
    lat0, lat1, lon0, lon1 = bbox
    track = downsample_track(lats, lons) if corridor else None
    launch = (float(lats[0]), float(lons[0])) if corridor else None
    impact = (float(lats[-1]), float(lons[-1])) if corridor else None
    flags = []
    if not gz.has_variant_counts(db):
        flags.append("gazetteer index predates variant-count ranking — "
                     "rebuild via Analysis ▸ Reference Data for better "
                     "label choices")
    fams = list(families or FAMILIES)
    # Two fetch passes over the lat-band index.  Pass A: the intrinsic
    # top tiers (rank ≤ 1 — capitals, seas, straits, space centres),
    # UNCAPPED: these classes are small by construction and must never
    # be lost to a row cap.  Pass B: everything else in ONE streaming
    # scan with per-family caps — and because the index streams in
    # LATITUDE order, a cap hit there biases only the long-tail classes
    # (villages, creeks), where the grid samples anyway; the bias is
    # flagged, never silent.
    fam_of, top = {}, set()
    for f in fams:
        for fc in family_fclasses(f):
            fam_of[fc] = f
    for tbl in (_GNS_RANKS, _GNIS_RANKS, _ATA_RANKS):
        for fc, (fam, rank) in tbl.items():
            if fam in fams and rank <= 1:
                top.add(fc)
    counts = {f: 0 for f in fams}
    capped = set()
    feats = []
    for lo, hi, shift in _wrap_intervals(lon0, lon1):
        rows, _tr = gz.features_in_bbox(lat0, lat1, lo, hi, fclasses=top,
                                        limit=10 * limit, db=db)
        if shift:
            rows = [dict(r, lon=r["lon"] + shift) for r in rows]
        feats.extend(rows)
        tail = set(fam_of) - top
        for r in gz.iter_features_in_bbox(lat0, lat1, lo, hi,
                                          fclasses=tail, db=db):
            fam = fam_of[r["fclass"]]
            if counts[fam] >= limit:
                capped.add(fam)
                continue
            counts[fam] += 1
            if shift:
                r = dict(r, lon=r["lon"] + shift)
            feats.append(r)
    for fam in sorted(capped):
        flags.append(f"{FAMILIES[fam][0]} fetch hit the {limit:,}-row "
                     "cap — that layer is a sample (top tiers are "
                     "complete; the thinning bias is southernmost-first)")
    ann = _annotate(feats, track, launch, impact)
    buckets = []
    for name, zmin, zmax, _zrep in ZOOM_BUCKETS:
        sel = select(ann, bbox, name, fams)
        buckets.append(dict(
            name=name, zmin=zmin, zmax=zmax,
            families={fam: [dict(n=c["name"], la=round(c["lat"], 5),
                                 lo=round(c["lon"], 5), fc=c["fclass"],
                                 lb=1 if c["labeled"] else 0)
                            for c in dots]
                      for fam, dots in sel.items()}))
    meta = {k: dict(label=v[0], color=v[1]) for k, v in FAMILIES.items()
            if k in fams}
    return dict(buckets=buckets, families=meta, flags=flags)


def folium_overlay_html(payload, map_var):
    """The self-contained <style>+<script> block that renders a
    build_payload() result on a folium map (Leaflet var name map_var):
    a top-right control with an off / dots / dots+labels selector per
    family, zoomend bucket switching, hover tooltips on every dot,
    permanent tooltips for the label-elected ones.  Static HTML — no
    network, no plugins (MarkerCluster explicitly rejected in the
    design: count-bubbles hide exactly the names the reader needs)."""
    import json
    return """
        <style>
        .gz-label { background: transparent; border: none; box-shadow:
            none; font: 11px/1.1 sans-serif; color: #333; text-shadow:
            -1px -1px 0 #fff, 1px -1px 0 #fff, -1px 1px 0 #fff,
            1px 1px 0 #fff; }
        .gz-label::before { display: none; }
        .gz-ctl { background: rgba(255,255,255,0.92); padding: 6px 8px;
            border-radius: 4px; box-shadow: 0 1px 4px rgba(0,0,0,0.3);
            font: 11px sans-serif; }
        .gz-ctl select { font: 11px sans-serif; margin-left: 4px; }
        .gz-ctl .gz-flag { color: #a33; max-width: 200px; }
        </style>
        <script>
        (function() {
            var GZ = __PAYLOAD__;
            // off | dots | labels, per family (populated on by default)
            var MODES = {};
            Object.keys(GZ.families).forEach(function(f) {
                MODES[f] = (f === 'populated') ? 'labels' : 'off';
            });
            setTimeout(function() {
                var map = __MAPVAR__;
                var layer = L.layerGroup().addTo(map);
                function bucket() {
                    var z = map.getZoom();
                    for (var i = 0; i < GZ.buckets.length; i++)
                        if (z >= GZ.buckets[i].zmin
                                && z <= GZ.buckets[i].zmax)
                            return GZ.buckets[i];
                    return GZ.buckets[GZ.buckets.length - 1];
                }
                function refresh() {
                    layer.clearLayers();
                    var b = bucket();
                    Object.keys(GZ.families).forEach(function(fam) {
                        if (MODES[fam] === 'off') return;
                        var col = GZ.families[fam].color;
                        (b.families[fam] || []).forEach(function(d) {
                            var mk = L.circleMarker([d.la, d.lo], {
                                radius: 3, color: col, weight: 1,
                                fillColor: col, fillOpacity: 0.85 });
                            var perm = (MODES[fam] === 'labels' && d.lb);
                            mk.bindTooltip(d.n, perm
                                ? {permanent: true, direction: 'right',
                                   offset: [6, 0], className: 'gz-label'}
                                : {direction: 'right', sticky: true});
                            if (!perm) mk.bindPopup(
                                '<b>' + d.n + '</b><br>' + d.fc);
                            layer.addLayer(mk);
                        });
                    });
                }
                var ctl = L.control({position: 'topright'});
                ctl.onAdd = function() {
                    var div = L.DomUtil.create('div', 'gz-ctl');
                    L.DomEvent.disableClickPropagation(div);
                    var h = '<b>Place names</b><br>';
                    Object.keys(GZ.families).forEach(function(fam) {
                        h += '<span style="color:'
                          + GZ.families[fam].color + '">&#9679;</span> '
                          + GZ.families[fam].label
                          + ' <select data-fam="' + fam + '">'
                          + ['off', 'dots', 'dots + labels'].map(
                              function(t, i) {
                                  var v = ['off', 'dots', 'labels'][i];
                                  return '<option value="' + v + '"'
                                    + (MODES[fam] === v ? ' selected'
                                                        : '')
                                    + '>' + t + '</option>'; }).join('')
                          + '</select><br>';
                    });
                    (GZ.flags || []).forEach(function(fl) {
                        h += '<div class="gz-flag">&#9888; ' + fl
                          + '</div>';
                    });
                    div.innerHTML = h;
                    div.addEventListener('change', function(ev) {
                        var f = ev.target.getAttribute('data-fam');
                        if (f) { MODES[f] = ev.target.value; refresh(); }
                    });
                    return div;
                };
                ctl.addTo(map);
                map.on('zoomend', refresh);
                refresh();
            }, 60);
        })();
        </script>""".replace("__PAYLOAD__", json.dumps(payload)) \
                    .replace("__MAPVAR__", map_var)
