"""Screening-envelope thresholds — the user-adjustable benchmark numbers.

Design decision (see README "Adjustable screening thresholds"): a Thrusty user
is a policy-focused modeler.  That person is far likelier to model a reentry
object that *survived* and want to adjust the ENVELOPE — how long a glider is
demonstrated to endure, how hard a MaRV is demonstrated to pull, how much heat
load an ablator family has flown and survived — in light of a new flight or
test, than to integrate new coupon data for one material.  So the first
editable surface is these ~10 curated ENVELOPE numbers, not the full material
catalog or the anchor datasets (both deferred to a future spreadsheet project).

The numbers are curated BY USER STORY, not by where they live in code: each is
pulled from wherever its consumer reads it (a module scalar, a material field).

Two disciplines are structural, not procedural:
  * SHIPPED DEFAULTS ARE FROZEN.  This module's REGISTRY holds the shipped
    default + its citation; a user edit lives ONLY in an overlay file
    (`benchmark_overrides.json`).  "Restore defaults" = discard the overlay.
    `test_thresholds.py` asserts the REGISTRY defaults equal the live module
    constants, so drift is caught, not hoped away.
  * MODIFIED BENCHMARKS SELF-DISCLOSE.  `modified()` lists any overridden
    number; the survivability report stamps the headline and prints a
    "Modified benchmarks" block so a changed number never rides on the shipped
    numbers' citations.
"""

from __future__ import annotations
import json
import os

# Overlay file for user edits (only the DIFFERENCES from default live here).
OVERRIDE_PATH = os.environ.get("THRUSTY_BENCHMARK_OVERRIDES", "benchmark_overrides.json")

# ── The registry: one row per curated envelope number ──────────────────────
# default = the shipped value (frozen; MUST match the live module constant —
# pinned by test_thresholds.py).  source = its one-line citation of record.
REGISTRY = [
    # ---- Glide endurance ----------------------------------------------------
    dict(key="uhtc_dwell_floor_s", group="Glide endurance",
         label="UHTC demonstrated dwell floor", units="s", default=300.0,
         kind="float", lo=1.0, hi=100000.0,
         source="Monteverde & Savino 2013 (300 s at 1973 K, zero recession)"),
    # ---- Maneuver envelope --------------------------------------------------
    dict(key="marv_g_operational", group="Maneuver envelope",
         label="Operational-MaRV pullout g", units="g", default=25.0,
         kind="float", lo=1.0, hi=1000.0,
         source="Pershing II pullout — Yengst 2010; corroborated by Lund 1984"),
    dict(key="marv_g_demonstrated", group="Maneuver envelope",
         label="Flight-demonstrated maneuver ceiling", units="g", default=100.0,
         kind="float", lo=1.0, hi=1000.0,
         source="AMaRV flight-measured (Bell XI accelerometers) — Yengst 2010"),
    # ---- Ablator flight record ----------------------------------------------
    # Demonstrated stagnation heat load per material family: the ablator
    # analogue of the UHTC dwell floor.  The verdict compares the flown load
    # against these — it no longer computes a recession point-value (see
    # METHODS §13.6).  A new recovered flight is a data edit here.
    dict(key="graphite_load_floor_MJ_m2", group="Ablator flight record",
         label="Graphite / bare-C-C demonstrated load", units="MJ/m²",
         default=3870.0, kind="float", lo=1.0, hi=100000.0,
         source="Reentry-F graphite nosetip flew Q ≈ 3.87 GJ/m² (NASA CR-154044 / LWP-460, pixel-traced, ±20%)"),
    dict(key="pica_load_floor_MJ_m2", group="Ablator flight record",
         label="PICA demonstrated load", units="MJ/m²",
         default=276.0, kind="float", lo=1.0, hi=100000.0,
         source="Stardust PICA forebody flew Q ≈ 276 MJ/m² and was recovered (Stackpoole et al. AIAA 2008-1202)"),
    dict(key="cp_load_floor_MJ_m2", group="Ablator flight record",
         label="Carbon-phenolic demonstrated load", units="MJ/m²",
         default=60.0, kind="float", lo=1.0, hi=100000.0,
         source="Pioneer Venus Large Probe survived ~60 MJ/m² (Cabrera & West 2026, coupled reconstruction; figure-integrated ±25%, short CO₂ pulse — conservative; Hayabusa corroborates higher)"),
    # ---- Interior survivability ---------------------------------------------
    dict(key="bondline_limit_C", group="Interior survivability",
         label="TPS-structure bondline limit", units="°C", default=250.0,
         kind="float", lo=50.0, hi=1000.0,
         source="Ablative-TPS sizing criterion — Dec & Braun NTRS 20060004824 (≤250 °C); Orion used 260 °C (NTRS 20080013535)"),
    # ---- Model conservatism -------------------------------------------------
    dict(key="acreage_flux_fraction", group="Model conservatism",
         label="Body-acreage flux fraction", units="× body stagnation",
         default=0.13, kind="float", lo=0.01, hi=1.0,
         source="Lu/Shi & Zhang 2024 (cone-tail/stagnation, <9% vs NASA TN D-5450)"),
    dict(key="windward_alpha_lo", group="Model conservatism",
         label="Windward AoA band — low", units="deg", default=5.0,
         kind="float", lo=0.0, hi=89.0,
         source="Thompson 1989 AoA error anchor (~40% near α≈3°)"),
    dict(key="windward_alpha_hi", group="Model conservatism",
         label="Windward AoA band — high", units="deg", default=20.0,
         kind="float", lo=0.0, hi=89.0,
         source="Thompson 1989 AoA error anchor (~15% at α≈20°)"),
]

_BY_KEY = {e["key"]: e for e in REGISTRY}
_overrides: dict = {}          # in-memory user edits (key → value)


# ── State ──────────────────────────────────────────────────────────────────
def defaults() -> dict:
    """The frozen shipped defaults (key → value)."""
    return {e["key"]: e["default"] for e in REGISTRY}


def current() -> dict:
    """Effective values = defaults overlaid with the user's overrides."""
    d = defaults()
    d.update({k: v for k, v in _overrides.items() if k in _BY_KEY})
    return d


def modified() -> list:
    """Rows whose current value differs from the shipped default.
    Each: dict(key, label, value, default, units, source)."""
    cur = current()
    out = []
    for e in REGISTRY:
        if cur[e["key"]] != e["default"]:
            out.append(dict(key=e["key"], label=e["label"], value=cur[e["key"]],
                            default=e["default"], units=e["units"],
                            source=e["source"]))
    return out


def is_modified() -> bool:
    return bool(modified())


def set_override(key, value):
    """Set/clear one override (value == default clears it)."""
    e = _BY_KEY.get(key)
    if e is None:
        raise KeyError(key)
    v = float(value)
    lo, hi = e.get("lo"), e.get("hi")
    if lo is not None:
        v = max(v, lo)
    if hi is not None:
        v = min(v, hi)
    if v == e["default"]:
        _overrides.pop(key, None)
    else:
        _overrides[key] = v


def set_overrides(d):
    for k, v in (d or {}).items():
        if k in _BY_KEY:
            set_override(k, v)


def reset():
    """Return to shipped defaults (clears every override)."""
    _overrides.clear()


# ── Persistence ─────────────────────────────────────────────────────────────
def load(path=None):
    """Load overrides from the overlay file (silently no-ops if absent)."""
    path = path or OVERRIDE_PATH
    _overrides.clear()
    try:
        with open(path) as f:
            set_overrides(json.load(f))
    except (FileNotFoundError, ValueError, OSError):
        pass
    return dict(_overrides)


def save(path=None):
    """Write current overrides to the overlay file; delete it when back to
    defaults so the shipped state is the on-disk default too."""
    path = path or OVERRIDE_PATH
    if _overrides:
        with open(path, "w") as f:
            json.dump(_overrides, f, indent=2, sort_keys=True)
    else:
        try:
            os.remove(path)
        except OSError:
            pass


# ── Apply into the live modules ─────────────────────────────────────────────
def apply():
    """Push the current values into the live model modules.  Consumers keep
    reading their module constants / material fields unchanged — this is the
    single place that writes them.  Lazy imports avoid an import cycle."""
    import heating
    import survivability_report as sr
    c = current()
    heating.BODY_FLUX_FRACTION = c["acreage_flux_fraction"]
    heating._WINDWARD_ALPHA_BAND = (c["windward_alpha_lo"], c["windward_alpha_hi"])
    _uhtc = heating.TPS_MATERIALS.get("uhtc")
    if _uhtc is not None:
        _uhtc["oxidation_dwell_s"] = c["uhtc_dwell_floor_s"]
    # Ablator demonstrated-load records (per material family).
    for _key in ("carbon_carbon", "carbon_ablator"):
        _m = heating.TPS_MATERIALS.get(_key)
        if _m is not None and _m.get("demonstrated_load_MJ_m2") is not None:
            _m["demonstrated_load_MJ_m2"] = c["graphite_load_floor_MJ_m2"]
    _pica = heating.TPS_MATERIALS.get("pica")
    if _pica is not None:
        _pica["demonstrated_load_MJ_m2"] = c["pica_load_floor_MJ_m2"]
    _cp = heating.TPS_MATERIALS.get("carbon_phenolic")
    if _cp is not None:
        _cp["demonstrated_load_MJ_m2"] = c["cp_load_floor_MJ_m2"]
    heating.BONDLINE_LIMIT_C = c["bondline_limit_C"]
    sr._MARV_G_OPERATIONAL = c["marv_g_operational"]
    sr._MARV_G_DEMONSTRATED = c["marv_g_demonstrated"]
