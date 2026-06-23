"""
mass_estimator.py — Dry-mass (inert/structural) estimator for rocket stages.

Thrusty asks the user to supply each stage's burnout mass (``mass_final``)
directly.  This module provides an independent, physics- and regression-based
*cross-check* on that number: given a stage's geometry, propellant load and
thrust it estimates the structural (dry / inert) mass, and reports how far a
stated dry mass diverges from the estimate.  Two complementary approaches are
implemented, with different appetites for input detail:

    1. COMPONENT-LEVEL  (Wilhite-school mass estimating relationships, MERs)
       The classic launch-vehicle conceptual-design approach: sum a set of
       per-component regressions (tanks, insulation, engines, thrust
       structure, gimbals, fairing, avionics, wiring …).  The primary
       coefficient set used here is the SI compilation taught in
       D. L. Akin, *Mass Estimating Relations*, U. Maryland ENAE 791 (2016),
       which descends from the historical Heineman / MacConochie-Klich /
       Glatt (WAATS) MERs — the same lineage as A. W. Wilhite's relations.
       Cross-check engine and structure coefficients are taken from
       Rohrschneider (SSDL, 2002) and Zandbergen (EUCASS, 2015).

    2. AGGREGATE  (whole-stage relations)
       A single regression or coefficient for the entire stage, requiring
       almost no geometry.  Useful as a sanity bound.  Implemented:
         * Pietrobon (2009) LOX/LH2 stage-mass power law  ms = a·mp^0.848
         * solid-stage inert regressions — Zandbergen (2026) broad-sample, and
           Lewis (2026, NG catalog) best-in-class composite/steel fits
         * structural coefficient   ε = m_inert / (m_inert + m_prop)
         * solid-motor propellant mass fraction  ζ = m_prop / m_total

Solid and liquid motors are treated separately because their mass drivers
differ (a solid's case is a loaded pressure vessel sized by chamber pressure
and grain volume; a liquid's tanks are sized by propellant volume and are
near-unloaded in flight).

UNITS — SI throughout unless a name says otherwise:
    mass kg · length m · area m² · volume m³ · force N · pressure Pa · time s.

The module has no third-party dependencies and no GUI coupling, so it can be
imported by Thrusty (see thrusty.py ``MassEstimatorDialog``) or run on its own:

    python mass_estimator.py --demo
    python mass_estimator.py liquid --prop LOX/RP1 --prop-mass 280000 \
                             --thrust 7.6e6 --engines 1 --stated-dry 18000

References (PDFs collected in the project's reference folder):
    Akin, D. L.  "Mass Estimating Relations", ENAE 791, U. Maryland, 2016.
    Rohrschneider, R. R.  "Development of a Mass Estimating Relationship
        Database for Launch Vehicle Conceptual Design", Georgia Tech / SSDL,
        AIAA 2001-4542 and the accompanying MER database, 2002.
    Zandbergen, B. T. C.  "Simple mass and size estimation relationships of
        pump-fed rocket engines for launch vehicle conceptual design",
        6th EUCASS, 2015.
    Zandbergen, B. T. C.  "Simple Parametric Relations for Solid Rocket Stage
        Inert Mass Estimation", TU Delft, 2026 (extends Zandbergen 2019).
    Northrop Grumman Propulsion Products Catalog (Jan 2023), pp. 9-39 — the
        29-motor data set behind the "Lewis 2026 (NG catalog)" best-in-class
        solid inert regression (size + L/D); data in mass_data/ng_motors.csv.
    Pietrobon, S. S.  "Analysis of Propellant Tank Masses", 2009.
    Heineman, W. Jr.  NASA TN-D-6349 (1971); NASA JSC-26098 (1994).
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from typing import Optional

_G0 = 9.80665  # standard gravity (m/s²)


# ---------------------------------------------------------------------------
# Propellant database
# ---------------------------------------------------------------------------
# Densities (kg/m³) at storage/cryogenic conditions and a representative engine
# oxidiser-to-fuel mixture ratio (O/F, by mass) used to split a total
# propellant load into oxidiser and fuel volumes.  Densities for LOX/LH2/RP-1
# follow Akin (ENAE 791); storables follow Sutton & Biblarz, "Rocket
# Propulsion Elements".  Mixture ratios are typical flight values, not optima.
#
# "cryo" flags a fluid that needs cryogenic tank insulation in the MER sum.
@dataclass(frozen=True)
class Fluid:
    name: str
    density: float      # kg/m³
    cryo: bool = False
    # tank MER family for the per-propellant-mass relation (Akin): one of
    # "LH2", "LOX", "RP1", or "" → fall back to the generic volume relation.
    tank_family: str = ""


_FLUIDS = {
    "LH2":      Fluid("Liquid hydrogen", 71.0, cryo=True,  tank_family="LH2"),
    "LCH4":     Fluid("Liquid methane",  423.0, cryo=True, tank_family="LOX"),
    "RP1":      Fluid("RP-1 kerosene",   820.0, tank_family="RP1"),
    "KEROSENE": Fluid("Kerosene",        810.0, tank_family="RP1"),
    "LOX":      Fluid("Liquid oxygen",   1140.0, cryo=True, tank_family="LOX"),
    "N2O4":     Fluid("Nitrogen tetroxide", 1450.0, tank_family="LOX"),
    "UDMH":     Fluid("UDMH",            793.0, tank_family="RP1"),
    "MMH":      Fluid("Monomethylhydrazine", 880.0, tank_family="RP1"),
    "A50":      Fluid("Aerozine-50",     899.0, tank_family="RP1"),
    "UH25":     Fluid("UH-25",           827.0, tank_family="RP1"),
    "IRFNA":    Fluid("IRFNA",           1510.0, tank_family="LOX"),
    "APCP":     Fluid("Solid (AP/Al composite)", 1800.0),  # solid grain
    "HTPB":     Fluid("Solid (HTPB composite)",  1810.0),
}

# Named bipropellant combinations: (oxidiser, fuel, O/F by mass, engine class).
# engine class selects the Zandbergen cross-check: "hydrolox" or "kerolox"
# (the latter covers all dense / storable bipropellants).
@dataclass(frozen=True)
class PropCombo:
    oxidizer: str
    fuel: str
    of_ratio: float
    engine_class: str  # "hydrolox" | "kerolox"


_COMBOS = {
    "LOX/LH2":   PropCombo("LOX", "LH2", 6.0, "hydrolox"),
    "LOX/RP1":   PropCombo("LOX", "RP1", 2.27, "kerolox"),
    "LOX/LCH4":  PropCombo("LOX", "LCH4", 3.6, "kerolox"),
    "N2O4/UDMH": PropCombo("N2O4", "UDMH", 2.6, "kerolox"),
    "N2O4/MMH":  PropCombo("N2O4", "MMH", 1.65, "kerolox"),
    "N2O4/A50":  PropCombo("N2O4", "A50", 2.0, "kerolox"),
    "N2O4/UH25": PropCombo("N2O4", "UH25", 2.65, "kerolox"),
    "IRFNA/UDMH": PropCombo("IRFNA", "UDMH", 3.1, "kerolox"),
}


def available_propellants() -> list[str]:
    """Liquid bipropellant combinations known to the estimator."""
    return list(_COMBOS.keys())


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class Component:
    """One line of a component-level mass breakdown."""
    name: str
    mass_kg: float
    basis: str = ""     # what drove the estimate (e.g. "V=270 m³")
    source: str = ""    # citation tag


@dataclass
class MassEstimate:
    """Result of one estimation method."""
    method: str                       # human-readable method name
    total_kg: float
    components: list[Component] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def add(self, name, mass, basis="", source=""):
        self.components.append(Component(name, float(mass), basis, source))
        self.total_kg += float(mass)
        return mass

    def table(self) -> str:
        """Plain-text component table."""
        if not self.components:
            return f"{self.method}: {self.total_kg:,.0f} kg"
        w = max(len(c.name) for c in self.components)
        lines = [f"{self.method}", "-" * (w + 30)]
        for c in self.components:
            extra = f"  ({c.basis})" if c.basis else ""
            lines.append(f"  {c.name:<{w}}  {c.mass_kg:>9,.0f} kg{extra}")
        lines.append("-" * (w + 30))
        lines.append(f"  {'TOTAL dry/inert':<{w}}  {self.total_kg:>9,.0f} kg")
        for n in self.notes:
            lines.append(f"  · {n}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Individual component MERs  (Akin / ENAE 791, SI)
# ---------------------------------------------------------------------------
def tank_mass_by_volume(volume_m3: float, family: str) -> float:
    """Propellant tank mass from tank volume.

    Akin ENAE 791:  M_LH2 = 9.09·V ;  M_other = 12.16·V   (V in m³, M in kg).
    The Akin coefficients are for aluminium / Al-Li launch-vehicle tankage;
    apply MATERIAL_TANK_FACTOR for other materials.
    """
    if volume_m3 <= 0:
        return 0.0
    coeff = 9.09 if family == "LH2" else 12.16
    return coeff * volume_m3


# Tank-wall material multiplier on the (aluminium-baselined) Akin tank MER.
# Only the tanks are material-sensitive; engines, thrust structure and avionics
# are essentially material-independent and are not scaled.
#   aluminium  1.00  — Akin baseline (Al 2219 class)
#   al-li      0.74  — Al-Li 2195 is ~26% lighter for equal strength
#                      (Pietrobon 2009, from specific-yield-strength ratio)
#   composite  0.45  — graphite/epoxy tankage ≈0.43–0.47× aluminium
#                      (Rohrschneider/SSDL 1970→2015 tank coefficients)
#   steel      1.60  — heavier thin-gauge / pressure-fed tankage; rare on
#                      pump-fed stages (Rohrschneider high-pressure steel MER)
MATERIAL_TANK_FACTOR = {
    "aluminium": 1.00, "aluminum": 1.00,
    "al-li": 0.74, "alli": 0.74,
    "composite": 0.45,
    "steel": 1.60,
}


def material_tank_factor(material: str) -> float:
    return MATERIAL_TANK_FACTOR.get((material or "aluminium").lower(), 1.0)


def tank_mass_offset(prop_mass_kg: float, family: str) -> float:
    """Alternate ("offset") Akin tank MER vintage with a fixed-mass term.

    A second Akin lecture set gives linear-plus-offset tank relations, which
    behave better for small stages than the pure-proportional volume form:
        LOX  M = 0.0152·m + 318 ;  LH2  M = 0.0694·m + 363   (kg).
    For storables/RP-1 the source's dedicated relation could not be read
    reliably from the scan, so the RP-1 proportional coefficient is reused.
    """
    if prop_mass_kg <= 0:
        return 0.0
    if family == "LOX":
        return 0.0152 * prop_mass_kg + 318.0
    if family == "LH2":
        return 0.0694 * prop_mass_kg + 363.0
    return 0.0148 * prop_mass_kg          # RP-1 family fallback (proportional)


def structural_index_ceiling(delta_v_ms: float, isp_s: float) -> float:
    """Maximum physically-attainable structural coefficient (Goldyn 2025):

        ε_max = 1 / exp(Δv / (g₀·Isp)).

    Above this, the rocket equation drives propellant mass negative — a stage
    with a higher ε simply cannot reach the required Δv with this Isp.  Used as
    a feasibility ceiling on a stated/estimated structural coefficient.
    """
    if delta_v_ms <= 0 or isp_s <= 0:
        return 1.0
    return 1.0 / math.exp(delta_v_ms / (_G0 * isp_s))


def tank_mass_by_propmass(prop_mass_kg: float, family: str) -> float:
    """Propellant tank mass from contained propellant mass.

    Akin ENAE 791 (volume relation evaluated at the nominal density):
        LH2  M = 0.128·m ;  LOX  M = 0.0107·m ;  RP-1  M = 0.0148·m.
    """
    coeff = {"LH2": 0.128, "LOX": 0.0107, "RP1": 0.0148}.get(family)
    if coeff is None:
        return 0.0
    return coeff * max(prop_mass_kg, 0.0)


def cryo_insulation_mass(tank_area_m2: float, fluid: str) -> float:
    """Cryogenic tank insulation mass per unit tank surface area.

    Akin ENAE 791:  LH2  2.88 kg/m² ;  LOX  1.123 kg/m².  Used for any
    cryogenic fluid (LCH4 treated as LOX-like).
    """
    if tank_area_m2 <= 0:
        return 0.0
    coeff = 2.88 if fluid == "LH2" else 1.123
    return coeff * tank_area_m2


def engine_mass_akin(thrust_n: float, expansion_ratio: float = 30.0) -> float:
    """Pump-fed liquid rocket engine mass (per engine), Akin ENAE 791:

        M = 7.81e-4·T + 3.37e-5·T·(Ae/At) + 59     (T in N, M in kg).
    """
    if thrust_n <= 0:
        return 0.0
    return 7.81e-4 * thrust_n + 3.37e-5 * thrust_n * expansion_ratio + 59.0


def engine_mass_zandbergen(thrust_n: float, engine_class: str) -> float:
    """Pump-fed engine mass cross-check, Zandbergen (EUCASS 2015), per engine:

        hydrolox  M = 0.00514·T^0.92068          (RSE ~13%)
        kerolox / storable  M = 1.104e-3·T + 27.702   (RSE ~26%)
    """
    if thrust_n <= 0:
        return 0.0
    if engine_class == "hydrolox":
        return 0.00514 * thrust_n ** 0.92068
    return 1.104e-3 * thrust_n + 27.702


def total_engine_mass(thrust_n: float, n_engines: int = 1,
                      engine_class: str = "kerolox",
                      expansion_ratio: float = 30.0,
                      model: str = "akin") -> float:
    """Total stage engine mass = n × per-engine MER (Akin or Zandbergen)."""
    n = max(int(n_engines), 1)
    t_each = thrust_n / n
    per = (engine_mass_zandbergen(t_each, engine_class) if model == "zandbergen"
           else engine_mass_akin(t_each, expansion_ratio))
    return n * per


# Typical engine-to-structure mass ratios κ_E = M_engine / M_structure by stage
# role, from Shu et al. (2020) — narrower and more stable than the structural
# ratio itself (their Fig. 2).  Anchors: KSLV-II 0.252/0.177/0.094 (stages
# 1/2/3); Titan II 0.250/0.111.  Used only as defaults; κ_E is an input.
_KAPPA_E_DEFAULT = {"lower": 0.25, "upper": 0.12, "": 0.18}


def engine_mass_ratio_inert(thrust_n: float, kappa_e: float,
                            n_engines: int = 1, engine_class: str = "kerolox",
                            expansion_ratio: float = 30.0,
                            model: str = "akin") -> tuple[float, float]:
    """Stage inert mass via the engine-mass-ratio method (Shu et al. 2020).

    The structural mass (excluding engines) is M_struct = M_engine / κ_E, where
    κ_E = M_engine/M_struct is taken from historical data (it varies over a much
    narrower band than the structural ratio).  Total inert = M_struct +
    M_engine = M_engine·(1 + 1/κ_E).  Engine mass is *predicted* from thrust, so
    — unlike assuming ε directly — this carries real information.

    Returns (total_inert_kg, engine_mass_kg).
    """
    m_eng = total_engine_mass(thrust_n, n_engines, engine_class,
                              expansion_ratio, model)
    if kappa_e <= 0:
        return 0.0, m_eng
    m_struct = m_eng / kappa_e
    return m_struct + m_eng, m_eng


def thrust_structure_mass(total_thrust_n: float) -> float:
    """Thrust structure mass, Akin ENAE 791:  M = 2.55e-4·T   (T total, N)."""
    return 2.55e-4 * max(total_thrust_n, 0.0)


def gimbal_mass(total_thrust_n: float, chamber_pressure_pa: float) -> float:
    """Engine gimbal / TVC actuator mass, Akin ENAE 791:

        M = 237.8·(T / P0)^0.9375     (T total thrust N, P0 chamber pressure Pa).
    """
    if total_thrust_n <= 0 or chamber_pressure_pa <= 0:
        return 0.0
    return 237.8 * (total_thrust_n / chamber_pressure_pa) ** 0.9375


def fairing_mass(area_m2: float) -> float:
    """Fairing / shroud mass, Akin ENAE 791:  M = 4.95·A^1.15   (A in m²)."""
    if area_m2 <= 0:
        return 0.0
    return 4.95 * area_m2 ** 1.15


def avionics_mass(gross_mass_kg: float) -> float:
    """Avionics mass, Akin ENAE 791:  M = 10·M0^0.361   (M0 gross mass, kg)."""
    if gross_mass_kg <= 0:
        return 0.0
    return 10.0 * gross_mass_kg ** 0.361


def wiring_mass(gross_mass_kg: float, length_m: float) -> float:
    """Wiring mass, Akin ENAE 791:  M = 1.058·sqrt(M0)·L^0.25   (M0 kg, L m)."""
    if gross_mass_kg <= 0 or length_m <= 0:
        return 0.0
    return 1.058 * math.sqrt(gross_mass_kg) * length_m ** 0.25


def solid_casing_mass_akin(propellant_mass_kg: float) -> float:
    """Solid rocket motor *case* mass, Akin ENAE 791:  M = 0.135·m_prop.

    NOTE: this is the case alone.  Nozzle, internal insulation, igniter and
    TVC are not covered by an open Wilhite-school MER (see module docstring and
    MASS_ESTIMATOR.md); use the aggregate propellant-mass-fraction method for a
    full solid-motor inert estimate.
    """
    return 0.135 * max(propellant_mass_kg, 0.0)


def pressure_vessel_mass(pressure_pa: float, volume_m3: float,
                         sigma_pa: float, density_kg_m3: float,
                         shape: str = "cylinder",
                         safety_factor: float = 1.5) -> float:
    """First-principles loaded pressure-vessel (motor case) mass.

    For a thin-walled vessel the structural mass is independent of wall
    thickness and set by the pressure-volume product divided by the material
    specific strength (σ/ρ):

        sphere:    m = 1.5 · SF · P·V / (σ/ρ)
        cylinder:  m = 2.0 · SF · P·V / (σ/ρ)   (membrane, hoop-stress driven)

    This is the physics the user's "Wilhite Solid" spreadsheet approximates via
    yield strength; it is a lower bound (no domes/joints/bosses margin beyond
    SF).  σ = material yield strength (Pa), density in kg/m³.
    """
    if min(pressure_pa, volume_m3, sigma_pa, density_kg_m3) <= 0:
        return 0.0
    k = 1.5 if shape == "sphere" else 2.0
    specific_strength = sigma_pa / density_kg_m3      # m²/s²  (=σ/ρ)
    return k * safety_factor * pressure_pa * volume_m3 / specific_strength


# ---------------------------------------------------------------------------
# Physics-based tank structural mass (Hutchinson & Olds, GT-STRESS, 2004)
# ---------------------------------------------------------------------------
# Structural-material property sets (SI):
#   density kg/m³, yield Pa, ultimate Pa, Young's modulus Pa, min gauge m.
#   Aluminium 2219 and Al-Li 2195 strengths follow Pietrobon (2009); steel is a
#   maraging/high-strength case; composite is a representative graphite/epoxy
#   laminate allowable.  These feed the failure-mode shell sizing below.
MATERIALS = {
    "aluminium": dict(rho=2840.0, sy=352e6,  su=455e6,  E=73.1e9, tmg=1.0e-3),
    "al-li":     dict(rho=2700.0, sy=520e6,  su=565e6,  E=76.0e9, tmg=1.0e-3),
    "steel":     dict(rho=7850.0, sy=1380e6, su=1500e6, E=200e9,  tmg=0.5e-3),
    "composite": dict(rho=1600.0, sy=600e6,  su=700e6,  E=70.0e9, tmg=1.3e-3),
}

# Stiffened-shell configuration factors (Hutchinson & Olds Table 1, after
# Ardema PDCYL): buckling efficiency ε and minimum-gauge multiplier K_mg.
SHELL_CONFIG = {
    "integral":   dict(eff=0.656, kmg=2.463),  # unflanged integrally stiffened ≈ isogrid
    "z-stiffened": dict(eff=0.911, kmg=2.475),
    "truss-core": dict(eff=0.605, kmg=4.310),
}

# Single-station simplification undershoots the full station-by-station
# GT-STRESS (which itself needs ×1.3665 to reach actual mass for secondary
# structure + vibration).  This factor folds frames + secondary structure +
# the station simplification together; calibrated so the model reproduces the
# aluminium EELV/ET tanks of Hutchinson & Olds (2004) to ≈±30%.
TANK_CORRELATION = 1.50

_FRAME_SPACING_M = 0.50   # representative ring-frame pitch for the buckling term


def tank_structural_mass(prop_mass_kg: float, prop_density_kg_m3: float,
                         diameter_m: float, length_m: float,
                         thrust_n: float, gross_mass_kg: float,
                         lateral_g: float = 0.5, ullage_pa: float = 2.5e5,
                         material: str = "aluminium",
                         shell_config: str = "integral") -> tuple[float, dict]:
    """Load- and material-driven tank shell structural mass (kg).

    A simplified single-critical-station implementation of the GT-STRESS beam /
    shell method of Hutchinson & Olds (AIAA 2004-3661).  The tank is sized at
    its most critical station as a thin circular shell under the worst of two
    load cases:

        * burnout / max axial : peak axial compression (≈ thrust reacted at the
          base), tank nearly empty so head pressure is small;
        * liftoff / max-q-α   : full tank (full head pressure) plus a lateral
          inertial bending load set by ``lateral_g``.

    Shell thickness at each case is the max over ultimate-tensile, yield, axial
    buckling (stiffened wide-column) and minimum-gauge criteria; the governing
    thickness across cases is multiplied by tank surface area, material density
    and TANK_CORRELATION (frames + secondary structure).  Returns (mass_kg,
    detail dict).  Internal pressure relieves axial compression, as in flight.
    """
    mat = MATERIALS.get(material, MATERIALS["aluminium"])
    cfg = SHELL_CONFIG.get(shell_config, SHELL_CONFIG["integral"])
    R = diameter_m / 2.0
    if R <= 0 or length_m <= 0 or prop_density_kg_m3 <= 0:
        return 0.0, {}

    # Axial compression reacted at the tank base ≈ stage thrust (at burnout
    # n_axial·g0·m_burnout = thrust; at liftoff n_axial·g0·m_gross ≈ thrust).
    P = max(thrust_n, gross_mass_kg * _G0)        # N
    M = lateral_g * _G0 * gross_mass_kg * (length_m / 2.0)   # N·m bending
    V = lateral_g * _G0 * gross_mass_kg                       # N shear
    head_full = prop_density_kg_m3 * _G0 * length_m           # Pa, full column

    def thickness(pct_fuel, with_lateral):
        p = ullage_pa + head_full * pct_fuel
        n_ax = -P / (2.0 * math.pi * R)                  # axial running load (compression)
        n_bend = (M / (math.pi * R ** 2)) if with_lateral else 0.0
        n_pr_x = p * R / 2.0                             # pressure (tensile)
        n_xy = (V / (math.pi * R)) if with_lateral else 0.0
        nx = n_bend + n_ax + n_pr_x
        ny = p * R                                       # hoop (tensile)
        n1 = (nx + ny) / 2.0 + math.sqrt(((nx - ny) / 2.0) ** 2 + n_xy ** 2)
        neq = math.sqrt(nx ** 2 + ny ** 2 - nx * ny + 3 * n_xy ** 2)
        n_comp = max(0.0, -(n_ax + n_bend))              # net compression
        t_uts = abs(n1) / mat["su"]
        t_ys = neq / mat["sy"]
        t_b = math.sqrt(n_comp * _FRAME_SPACING_M / (cfg["eff"] * mat["E"])) \
            if n_comp > 0 else 0.0
        return max(t_uts, t_ys, t_b, cfg["kmg"] * mat["tmg"])

    t = max(thickness(0.08, with_lateral=False),   # burnout: empty, axial
            thickness(1.00, with_lateral=True))     # liftoff/max-q-α: full
    area = tank_surface_area(prop_mass_kg / prop_density_kg_m3, diameter_m)
    mass = TANK_CORRELATION * mat["rho"] * area * t
    return mass, {"t_mm": t * 1e3, "area_m2": area,
                  "axial_kN": P / 1e3, "head_kPa": head_full / 1e3}


# ---------------------------------------------------------------------------
# Aggregate relations
# ---------------------------------------------------------------------------
# Pietrobon (2009) LOX/LH2 stage-mass power law, ms & mp in TONNES.  ms is the
# stage mass *less engines*; the variants differ by construction technology.
_PIETROBON = {
    "average":         0.19,    # all hydrolox stages averaged
    "common_bulkhead": 0.1583,  # rescaled through Saturn S-II
    "alli_2195":       0.1171,  # Al-Li 2195 common bulkhead
}


def pietrobon_stage_mass(prop_mass_kg: float, variant: str = "average") -> float:
    """Hydrolox stage structural mass (less engines), Pietrobon (2009):

        ms[t] = a · mp[t]^0.848 ,  a = 0.19 / 0.1583 / 0.1171.

    Returns kg.  Strictly valid only for LOX/LH2 stages.
    """
    a = _PIETROBON.get(variant, 0.19)
    mp_t = max(prop_mass_kg, 0.0) / 1000.0
    return a * mp_t ** 0.848 * 1000.0


def inert_from_structural_coefficient(prop_mass_kg: float, epsilon: float) -> float:
    """Inert mass from structural coefficient ε = m_inert/(m_inert+m_prop):

        m_inert = ε·m_prop / (1 − ε).
    """
    if not (0.0 < epsilon < 1.0):
        return 0.0
    return epsilon * prop_mass_kg / (1.0 - epsilon)


def inert_from_mass_fraction(prop_mass_kg: float, zeta: float) -> float:
    """Inert mass from propellant mass fraction ζ = m_prop/m_total:

        m_inert = m_prop·(1/ζ − 1).

    Typical modern large solid motors ζ ≈ 0.90–0.93; smaller / older ζ ≈ 0.85.
    """
    if not (0.0 < zeta < 1.0):
        return 0.0
    return prop_mass_kg * (1.0 / zeta - 1.0)


# Solid rocket *stage* inert-mass regressions, Zandbergen (2026, "Simple
# Parametric Relations for Solid Rocket Stage Inert Mass Estimation"; extends
# Zandbergen 2019).  Inert = whole stage dry mass (case, nozzle, insulation,
# igniter, skirts, TVC, avionics, separation …) — NOT motor case alone.  All
# masses in TONNES inside the fit; helpers below take/return kg.  Excludes
# upper stages (payload-integration mass distorts the scaling).
#   Steel casings (N=17):
#     S2  m_i = 0.1689·m_p + 0.5090   (R²=0.997, RMSPE 23%) — linear+offset
#     S3  m_i = 0.2851·m_p^0.9030     (R²=0.993, RMSPE 22%) — power law
#   Composite casings (N=17):
#     C1  m_i = 0.1110·m_p            (R²=0.978, RMSPE 25%) — linear
#     C2  m_i = 0.1275·m_p^0.9678     (R²=0.966, RMSPE 24%) — power law
_SOLID_STAGE = {
    "steel":     {"linear": (0.1689, 0.5090), "power": (0.2851, 0.9030),
                  "rmspe": 0.22},
    "composite": {"linear": (0.1110, 0.0),    "power": (0.1275, 0.9678),
                  "rmspe": 0.24},
}

# "Lewis 2026 (NG catalog)" — Lewis regression of the Northrop Grumman
# Propulsion Products Catalog (Jan 2023, pp. 9-39): 29 flight-proven
# Orion/Castor/GEM motors, the best-in-class US composite/steel family.  These
# are mature, mass-optimised flight motors, so this fit runs ~10% LIGHTER than
# the broader Zandbergen sample — it is the lower (best-in-class) edge of the
# whole-stage inert band, useful for "fanciest US motor" sizing.
#   Power law (recommended):  m_inert = k · m_prop^0.947
#   Constant fraction:        m_inert = f/(1-f) · m_prop,  f = m_inert/m_loaded
# COEFFICIENTS ARE IN lbm, exactly as published; converted at the kg boundary.
# ~20% (1σ) per-motor scatter.  Steel rests on 3 Castor-IV motors (~20-35k lbm
# loaded); do not trust steel outside that band.  Valid m_prop ~1,700-114,000 lbm.
_SOLID_STAGE_LEWIS = {
    "composite": {"k": 0.172, "f": 0.092, "rmspe": 0.22},
    "steel":     {"k": 0.258, "f": 0.132, "rmspe": 0.22},
    "exp":       0.947,
    # Two-variable (size + slenderness) model, refit on the 26-motor composite
    # group with PROPELLANT mass as the size term (not loaded mass — see note):
    #   m_inert = k · m_prop^b · (L/D)^c          [lbm, L/D dimensionless]
    # The L/D exponent is POSITIVE: a more slender motor carries more inert per
    # unit propellant (case wall + insulation scale with surface area, not
    # volume).  This is what makes the fit usable on slender missile stages,
    # which the stubby-skewed catalogue under-covers.  Significant (p≈0.006),
    # cuts RMS 21.6%→18.1%.  Steel uses the composite coefficients × a reduced
    # 1.347 material multiplier — down from the size-only 1.50, because the
    # slender steel Castor-IV motors' extra inert is now explained by L/D, not
    # material.  Worst residual is still Castor 30XL at -26% (submerged nozzle /
    # advanced composite, not in the data).
    # NOTE: propellant-based, not the loaded-mass fit in the dataset README.
    # Loaded mass scores marginally better (RMS 16.6%) but m_loaded = m_prop +
    # m_inert, so predicting inert from loaded is partly circular — invalid for
    # a tool that VALIDATES a stated inert mass.  Propellant is the independent
    # design input, so it is the correct predictor here.
    "ld": {"k": 0.24087, "b": 0.8832, "c": 0.1834,
           "steel_mult": 1.347, "rmspe": 0.18},
}
_LB_PER_KG = 2.2046226218


def solid_stage_inert(prop_mass_kg: float, casing: str = "steel",
                      form: str = "power", source: str = "zandbergen",
                      ld_ratio: float = 0.0) -> float:
    """Whole-stage solid inert mass.

    casing ∈ {"steel", "composite"}; form ∈ {"power", "linear"}.
    source ∈ {"zandbergen", "lewis"}:
      "zandbergen" — Zandbergen (2026) broad-sample stage regression (default).
      "lewis"      — Lewis (2026) fit to the NG Propulsion Products Catalog
                     (2023); best-in-class US motors, ~10% lighter.  With
                     form="power", if ``ld_ratio`` > 0 the two-variable
                     size+slenderness model is used; otherwise the size-only
                     coefficient.  form="linear" = constant fraction.
                     ``ld_ratio`` is the MOTOR (case+nozzle) length/diameter,
                     matching the catalogue's "overall length" — NOT a stage
                     length padded with interstages/skirts, which would inflate
                     L/D and over-predict inert mass.
    Returns kg.  See _SOLID_STAGE / _SOLID_STAGE_LEWIS for coefficients.
    """
    if source == "lewis":
        fam = _SOLID_STAGE_LEWIS.get(casing, _SOLID_STAGE_LEWIS["composite"])
        mp_lb = max(prop_mass_kg, 0.0) * _LB_PER_KG
        if form == "linear":                    # constant inert-fraction model
            f = fam["f"]
            mi_lb = f / (1.0 - f) * mp_lb
        elif ld_ratio and ld_ratio > 0.0:       # size + slenderness (L/D) model
            ld = _SOLID_STAGE_LEWIS["ld"]
            mi_lb = ld["k"] * mp_lb ** ld["b"] * ld_ratio ** ld["c"]
            if casing == "steel":
                mi_lb *= ld["steel_mult"]
        else:                                    # size-only power law
            mi_lb = fam["k"] * mp_lb ** _SOLID_STAGE_LEWIS["exp"]
        return mi_lb / _LB_PER_KG

    fam = _SOLID_STAGE.get(casing, _SOLID_STAGE["steel"])
    mp_t = max(prop_mass_kg, 0.0) / 1000.0
    if form == "linear":
        a, b = fam["linear"]
        return (a * mp_t + b) * 1000.0
    a, n = fam["power"]
    return a * mp_t ** n * 1000.0


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def tank_surface_area(volume_m3: float, diameter_m: float = 0.0) -> float:
    """Surface area of a tank holding ``volume_m3``.

    If a body diameter is given the tank is modelled as a cylinder of that
    diameter capped with hemispherical domes; otherwise an equivalent sphere is
    used (the minimum-area shape, matching Akin's worked example).
    """
    if volume_m3 <= 0:
        return 0.0
    if diameter_m and diameter_m > 0:
        r = diameter_m / 2.0
        v_domes = 4.0 / 3.0 * math.pi * r ** 3            # two hemispheres
        if volume_m3 <= v_domes:                          # fits in a sphere
            r_eff = (volume_m3 / (4.0 / 3.0 * math.pi)) ** (1.0 / 3.0)
            return 4.0 * math.pi * r_eff ** 2
        l_cyl = (volume_m3 - v_domes) / (math.pi * r ** 2)
        return 2.0 * math.pi * r * l_cyl + 4.0 * math.pi * r ** 2
    r = (volume_m3 / (4.0 / 3.0 * math.pi)) ** (1.0 / 3.0)
    return 4.0 * math.pi * r ** 2


# ---------------------------------------------------------------------------
# Stage input dataclasses
# ---------------------------------------------------------------------------
@dataclass
class LiquidStageInputs:
    propellant: str = "LOX/RP1"      # key in available_propellants()
    prop_mass_kg: float = 0.0        # total propellant (ox + fuel)
    thrust_n: float = 0.0            # total stage vacuum thrust
    n_engines: int = 1
    expansion_ratio: float = 30.0
    chamber_pressure_pa: float = 6.9e6   # ~1000 psi, typical
    diameter_m: float = 0.0          # body diameter (for tank area; 0 → spheres)
    length_m: float = 0.0            # stage length (for wiring; 0 → skip)
    gross_mass_kg: float = 0.0       # stage gross mass (for avionics/wiring)
    fairing_area_m2: float = 0.0     # payload fairing area (0 → none)
    include_avionics: bool = True    # True only on the stage carrying the IU
    vehicle_gross_kg: float = 0.0    # vehicle GLOW for avionics sizing (0 → stage)
    of_ratio: float = 0.0            # override O/F (0 → combo default)
    tank_material: str = "aluminium"  # aluminium | al-li | composite | steel
    # Tank sizing model:
    #   akin_volume — empirical Akin volume MER (material-scaled) [default]
    #   akin_offset — alternate Akin linear+offset MER vintage
    #   physics     — GT-STRESS load/material shell sizing (Hutchinson)
    #   averaged    — mean of akin_volume and physics (SPSP multi-estimate)
    tank_model: str = "akin_volume"
    physics_tank: bool = False       # back-compat alias → tank_model="physics"
    lateral_g: float = 0.5           # design lateral load factor (bending)
    ullage_pa: float = 2.5e5         # design tank/ullage pressure
    shell_config: str = "integral"   # integral | z-stiffened | truss-core
    # Engine-mass-ratio (Shu) aggregate: κ_E = M_engine/M_structure.
    kappa_e: float = 0.0             # 0 → default by stage_role
    stage_role: str = ""             # "lower" | "upper" | "" (picks κ_E default)
    # Optional feasibility ceiling on the structural coefficient (Goldyn):
    delta_v_ms: float = 0.0          # stage Δv (0 → skip ceiling)
    isp_s: float = 0.0               # stage Isp (0 → skip ceiling)


@dataclass
class SolidStageInputs:
    prop_mass_kg: float = 0.0
    thrust_n: float = 0.0            # total thrust (for thrust structure / gimbal)
    chamber_pressure_pa: float = 0.0  # 0 → skip physics case estimate
    propellant: str = "APCP"
    casing: str = "steel"            # "steel" | "composite" (Zandbergen 2026)
    case_sigma_pa: float = 1.50e9    # case yield strength (HY-/maraging steel ~1.5 GPa)
    case_density_kg_m3: float = 7850.0
    case_shape: str = "cylinder"
    mass_fraction: float = 0.0       # ζ override for the aggregate estimate (0 → skip)
    diameter_m: float = 0.0
    length_m: float = 0.0
    gross_mass_kg: float = 0.0
    include_avionics: bool = True
    vehicle_gross_kg: float = 0.0


# ---------------------------------------------------------------------------
# High-level estimators
# ---------------------------------------------------------------------------
def estimate_liquid_stage(inp: LiquidStageInputs) -> MassEstimate:
    """Component-level (Wilhite/Akin) dry-mass estimate for a liquid stage."""
    est = MassEstimate("Component-level (Akin/Wilhite MERs)", 0.0)
    combo = _COMBOS.get(inp.propellant)
    if combo is None:
        raise ValueError(f"unknown propellant combo {inp.propellant!r}; "
                         f"choose from {available_propellants()}")
    of = inp.of_ratio if inp.of_ratio > 0 else combo.of_ratio
    ox = _FLUIDS[combo.oxidizer]
    fu = _FLUIDS[combo.fuel]

    # Split propellant by mixture ratio (O/F by mass).
    m_ox = inp.prop_mass_kg * of / (of + 1.0)
    m_fu = inp.prop_mass_kg - m_ox
    v_ox = m_ox / ox.density
    v_fu = m_fu / fu.density

    # Tank sizing model (back-compat: physics_tank=True forces "physics").
    model = "physics" if inp.physics_tank else inp.tank_model
    matf = material_tank_factor(inp.tank_material)
    gross = inp.gross_mass_kg if inp.gross_mass_kg > 0 else inp.prop_mass_kg
    v_tot = v_ox + v_fu

    def _empirical(vfl, mfl, fam):
        if model == "akin_offset":
            return tank_mass_offset(mfl, fam)
        return matf * tank_mass_by_volume(vfl, fam)

    def _physics(fl, mfl, vfl):
        l_tank = inp.length_m * (vfl / v_tot) if (inp.length_m > 0 and
                                                  v_tot > 0) else 0.0
        if l_tank <= 0 and inp.diameter_m > 0:
            l_tank = vfl / (math.pi * (inp.diameter_m / 2.0) ** 2)
        m_tank, det = tank_structural_mass(
            mfl, fl.density, inp.diameter_m, l_tank, inp.thrust_n, gross,
            lateral_g=inp.lateral_g, ullage_pa=inp.ullage_pa,
            material=inp.tank_material, shell_config=inp.shell_config)
        return m_tank, det

    for fl, mfl, vfl, tag in ((ox, m_ox, v_ox, combo.oxidizer),
                              (fu, m_fu, v_fu, combo.fuel)):
        if model == "physics":
            m_tank, det = _physics(fl, mfl, vfl)
            est.add(f"{tag} tank (GT-STRESS)", m_tank,
                    basis=f"t={det.get('t_mm', 0):.1f} mm, {inp.tank_material}",
                    source="Hutchinson")
        elif model == "averaged":
            m_emp = _empirical(vfl, mfl, fl.tank_family)
            m_phys, _ = _physics(fl, mfl, vfl)
            est.add(f"{tag} tank (avg MER+physics)", 0.5 * (m_emp + m_phys),
                    basis=f"empirical {m_emp:,.0f} / physics {m_phys:,.0f}",
                    source="SPSP")
        else:  # akin_volume | akin_offset
            label = "offset" if model == "akin_offset" else inp.tank_material
            est.add(f"{'Oxidiser' if tag == combo.oxidizer else 'Fuel'} "
                    f"tank ({tag})", _empirical(vfl, mfl, fl.tank_family),
                    basis=f"V={vfl:,.1f} m³, {label}", source="Akin")

    # Cryogenic insulation (only for cryo fluids).
    if ox.cryo:
        a = tank_surface_area(v_ox, inp.diameter_m)
        est.add(f"{combo.oxidizer} insulation",
                cryo_insulation_mass(a, combo.oxidizer),
                basis=f"A={a:,.1f} m²", source="Akin")
    if fu.cryo:
        a = tank_surface_area(v_fu, inp.diameter_m)
        est.add(f"{combo.fuel} insulation",
                cryo_insulation_mass(a, combo.fuel),
                basis=f"A={a:,.1f} m²", source="Akin")

    # Engines.
    n = max(int(inp.n_engines), 1)
    t_each = inp.thrust_n / n
    est.add(f"Engines (×{n})",
            n * engine_mass_akin(t_each, inp.expansion_ratio),
            basis=f"{t_each/1e3:,.0f} kN each, ε={inp.expansion_ratio:g}",
            source="Akin")

    # Thrust structure + gimbals.
    est.add("Thrust structure", thrust_structure_mass(inp.thrust_n),
            basis=f"T={inp.thrust_n/1e3:,.0f} kN", source="Akin")
    est.add("Gimbals / TVC",
            gimbal_mass(inp.thrust_n, inp.chamber_pressure_pa),
            basis=f"Pc={inp.chamber_pressure_pa/1e6:.1f} MPa", source="Akin")

    # Fairing (optional).
    if inp.fairing_area_m2 > 0:
        est.add("Fairing / shroud", fairing_mass(inp.fairing_area_m2),
                basis=f"A={inp.fairing_area_m2:,.1f} m²", source="Akin")

    # Wiring is present on every stage (sized on the stage's own gross mass).
    stage_m0 = inp.gross_mass_kg if inp.gross_mass_kg > 0 else \
        inp.prop_mass_kg + est.total_kg
    if inp.length_m > 0:
        est.add("Wiring", wiring_mass(stage_m0, inp.length_m),
                basis=f"L={inp.length_m:.1f} m", source="Akin")
    # Avionics is the guidance package — one per vehicle, carried on the upper
    # stage (never on the bus).  Sized on vehicle GLOW when supplied, since the
    # package scales with the vehicle it guides, not the stage it rides on.
    if inp.include_avionics:
        veh_m0 = inp.vehicle_gross_kg if inp.vehicle_gross_kg > 0 else stage_m0
        est.add("Avionics", avionics_mass(veh_m0),
                basis=f"GLOW={veh_m0/1e3:,.0f} t", source="Akin")

    est.notes.append(f"O/F={of:g}; ρ_ox={ox.density:.0f}, ρ_fu={fu.density:.0f}")
    # Engine cross-check note.
    z = n * engine_mass_zandbergen(t_each, combo.engine_class)
    est.notes.append(f"engine cross-check (Zandbergen {combo.engine_class}): "
                     f"{z:,.0f} kg")
    return est


def estimate_solid_stage(inp: SolidStageInputs) -> MassEstimate:
    """Component-level dry-mass estimate for a solid rocket stage.

    Solid component MERs are far less established than liquid ones; this sums
    the Akin case relation with a thrust structure / gimbal estimate and, when a
    chamber pressure is given, a first-principles pressure-vessel case mass for
    comparison.  For a complete inert estimate prefer the aggregate
    propellant-mass-fraction method (see aggregate_estimates / divergence).
    """
    est = MassEstimate("Component-level (solid, partial)", 0.0)
    est.add("Motor case (Akin 0.135·m_prop)",
            solid_casing_mass_akin(inp.prop_mass_kg),
            basis=f"m_prop={inp.prop_mass_kg/1e3:,.1f} t", source="Akin")
    if inp.thrust_n > 0:
        est.add("Thrust structure", thrust_structure_mass(inp.thrust_n),
                basis=f"T={inp.thrust_n/1e3:,.0f} kN", source="Akin")
        if inp.chamber_pressure_pa > 0:
            est.add("Gimbals / TVC",
                    gimbal_mass(inp.thrust_n, inp.chamber_pressure_pa),
                    basis=f"Pc={inp.chamber_pressure_pa/1e6:.1f} MPa",
                    source="Akin")
    if inp.include_avionics:
        veh_m0 = inp.vehicle_gross_kg if inp.vehicle_gross_kg > 0 else \
            (inp.gross_mass_kg if inp.gross_mass_kg > 0
             else inp.prop_mass_kg + est.total_kg)
        est.add("Avionics", avionics_mass(veh_m0),
                basis=f"GLOW={veh_m0/1e3:,.0f} t", source="Akin")

    # Physics-based pressure-vessel case (reported as a note, not summed, to
    # avoid double counting with the Akin case relation).
    if inp.chamber_pressure_pa > 0:
        grain = _FLUIDS.get(inp.propellant)
        rho_grain = grain.density if grain else 1800.0
        v_chamber = inp.prop_mass_kg / rho_grain
        pv = pressure_vessel_mass(inp.chamber_pressure_pa, v_chamber,
                                  inp.case_sigma_pa, inp.case_density_kg_m3,
                                  inp.case_shape)
        est.notes.append(
            f"pressure-vessel case cross-check: {pv:,.0f} kg "
            f"(Pc={inp.chamber_pressure_pa/1e6:.1f} MPa, V={v_chamber:,.1f} m³, "
            f"σ/ρ={inp.case_sigma_pa/inp.case_density_kg_m3/1e3:,.0f} kJ/kg)")
    est.notes.append("solid component MERs are incomplete (no open "
                     "nozzle/insulation MER) — see aggregate method")
    return est


def aggregate_estimates(prop_mass_kg: float, *, is_solid: bool,
                        hydrolox: bool = False,
                        epsilon: float = 0.0,
                        zeta: float = 0.0,
                        casing: str = "steel",
                        ld_ratio: float = 0.0) -> list[MassEstimate]:
    """Whole-stage aggregate dry-mass estimates.

    Returns one MassEstimate per applicable relation.  For solids the
    Zandbergen (2026) steel/composite stage regressions are the headline; for
    hydrolox liquids the Pietrobon power laws apply.  Non-hydrolox liquids
    have no fitted aggregate in the open literature, so the component-level
    buildup is the only true estimate for them.

    ``epsilon`` (liquid) and ``zeta`` (solid) are *assumed* fractions, not
    predictions — supplying them merely converts your own assumption into kg
    (m_inert = ε·m_prop/(1−ε)).  They are therefore opt-in: omitted unless
    explicitly given.  ε is otherwise used only as a reporting unit in the
    divergence table.
    """
    out: list[MassEstimate] = []
    if is_solid:
        rmspe = _SOLID_STAGE.get(casing, _SOLID_STAGE["steel"])["rmspe"]
        for form, label in (("power", "power-law"), ("linear", "linear")):
            m = solid_stage_inert(prop_mass_kg, casing, form)
            out.append(MassEstimate(
                f"Aggregate: Zandbergen {casing} stage ({label})", m,
                notes=[f"whole-stage inert; RMSPE ≈{rmspe*100:.0f}%"]))
        # Best-in-class cross-check: Lewis (2026) NG-catalog fit.  Runs ~10%
        # lighter than Zandbergen — the lower edge of the inert band, i.e. the
        # fanciest flight-proven US composite/steel motors.
        m = solid_stage_inert(prop_mass_kg, casing, "power", source="lewis",
                              ld_ratio=ld_ratio)
        if ld_ratio and ld_ratio > 0.0:
            lab = f"power-law, L/D={ld_ratio:.1f}"
            note = (f"best-in-class US motors, size+slenderness; "
                    f"RMSPE ≈{_SOLID_STAGE_LEWIS['ld']['rmspe']*100:.0f}%")
        else:
            lab = "power-law, size-only"
            note = (f"best-in-class US motors; "
                    f"RMSPE ≈{_SOLID_STAGE_LEWIS[casing]['rmspe']*100:.0f}% "
                    f"(supply length+diameter for the L/D correction)")
        out.append(MassEstimate(
            f"Aggregate: Lewis 2026 NG-catalog {casing} ({lab})", m, notes=[note]))
        mf = solid_stage_inert(prop_mass_kg, casing, "linear", source="lewis")
        out.append(MassEstimate(
            f"Aggregate: Lewis 2026 NG-catalog {casing} (constant-fraction)", mf,
            notes=[f"best-in-class inert fraction "
                   f"f={_SOLID_STAGE_LEWIS[casing]['f']:.3f}"]))
        if zeta > 0:
            out.append(MassEstimate(
                f"Assumed propellant mass fraction ζ={zeta:g}",
                inert_from_mass_fraction(prop_mass_kg, zeta),
                notes=["restates your assumption in kg — not a prediction"]))
    else:
        if hydrolox:
            for variant, label in (("average", "avg"),
                                    ("common_bulkhead", "common bulkhead"),
                                    ("alli_2195", "Al-Li 2195")):
                m = pietrobon_stage_mass(prop_mass_kg, variant)
                out.append(MassEstimate(
                    f"Aggregate: Pietrobon hydrolox ({label})", m,
                    notes=["stage mass less engines"]))
        if epsilon > 0:
            out.append(MassEstimate(
                f"Assumed structural coefficient ε={epsilon:g}",
                inert_from_structural_coefficient(prop_mass_kg, epsilon),
                notes=["restates your assumption in kg — not a prediction"]))
    return out


# ---------------------------------------------------------------------------
# Divergence reporting
# ---------------------------------------------------------------------------
@dataclass
class Divergence:
    method: str
    estimate_kg: float
    stated_kg: float
    prop_mass_kg: float = 0.0     # enables structural-coefficient reporting

    @staticmethod
    def _eps(dry, prop):
        tot = dry + prop
        return (dry / tot) if tot > 0 else float("nan")

    @property
    def eps_stated(self) -> float:
        """Structural coefficient ε = dry/(dry+prop) implied by the stated mass."""
        return self._eps(self.stated_kg, self.prop_mass_kg)

    @property
    def eps_estimate(self) -> float:
        """ε implied by this method's estimated dry mass."""
        return self._eps(self.estimate_kg, self.prop_mass_kg)

    @property
    def delta_kg(self) -> float:
        return self.stated_kg - self.estimate_kg

    @property
    def pct(self) -> float:
        """Stated relative to estimate: +x% means stated is x% heavier."""
        if self.estimate_kg == 0:
            return float("nan")
        return 100.0 * (self.stated_kg - self.estimate_kg) / self.estimate_kg

    def verdict(self) -> str:
        p = self.pct
        if math.isnan(p):
            return "n/a"
        a = abs(p)
        if a <= 15:
            return "consistent"
        if a <= 35:
            return "marginal"
        return ("optimistic (lighter than typical)" if p < 0
                else "conservative (heavier than typical)")


def divergence_report(stated_dry_kg: float,
                      estimates: list[MassEstimate],
                      prop_mass_kg: float = 0.0) -> list[Divergence]:
    """Compare a stated dry mass against every estimate's total.

    When ``prop_mass_kg`` is given, each Divergence also exposes the implied
    structural coefficient ε = dry/(dry+prop) for the stated and estimated
    masses.
    """
    return [Divergence(e.method, e.total_kg, stated_dry_kg, prop_mass_kg)
            for e in estimates]


def format_divergence(report: list[Divergence]) -> str:
    if not report:
        return "(no estimates)"
    wm = max(len(d.method) for d in report)
    show_eps = any(d.prop_mass_kg > 0 for d in report)
    if show_eps:
        # ε_stated is the same for every row; print it once as a header.
        eps_s = report[0].eps_stated
        head = [f"  stated dry mass → structural coefficient ε = "
                f"{eps_s:.3f}  (λ = dry/prop = {eps_s/(1-eps_s):.3f})",
                "",
                f"  {'method':<{wm}}  {'est dry':>9}  {'ε est':>6}  "
                f"{'Δ%':>7}   verdict",
                "  " + "-" * (wm + 40)]
        rows = [f"  {d.method:<{wm}}  {d.estimate_kg:>7,.0f}kg  "
                f"{d.eps_estimate:>6.3f}  {d.pct:>+6.1f}%   {d.verdict()}"
                for d in report]
        return "\n".join(head + rows)
    lines = [f"  {'method':<{wm}}  {'estimate':>10}  {'stated':>10}  "
             f"{'Δ%':>7}   verdict",
             "  " + "-" * (wm + 45)]
    for d in report:
        lines.append(f"  {d.method:<{wm}}  {d.estimate_kg:>8,.0f} kg  "
                     f"{d.stated_kg:>8,.0f} kg  {d.pct:>+6.1f}%   {d.verdict()}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Convenience driver used by both the CLI and Thrusty
# ---------------------------------------------------------------------------
def analyse_liquid(inp: LiquidStageInputs,
                   stated_dry_kg: float = 0.0) -> tuple[list[MassEstimate],
                                                        list[Divergence]]:
    combo = _COMBOS[inp.propellant]
    comp = estimate_liquid_stage(inp)
    aggs = aggregate_estimates(inp.prop_mass_kg, is_solid=False,
                               hydrolox=(combo.engine_class == "hydrolox"))

    # Engine-mass-ratio method (Shu 2020) — predictive, propellant-agnostic;
    # the key aggregate for non-hydrolox liquids that have no other.
    kappa = inp.kappa_e if inp.kappa_e > 0 else \
        _KAPPA_E_DEFAULT.get(inp.stage_role, _KAPPA_E_DEFAULT[""])
    if inp.thrust_n > 0:
        inert, m_eng = engine_mass_ratio_inert(
            inp.thrust_n, kappa, inp.n_engines, combo.engine_class,
            inp.expansion_ratio)
        aggs.append(MassEstimate(
            f"Engine-mass-ratio (Shu), κ_E={kappa:g}", inert,
            notes=[f"M_engine={m_eng:,.0f} kg; M_struct=M_eng/κ_E"]))

    estimates = [comp] + aggs

    # Feasibility ceiling on the structural coefficient (Goldyn 2025).
    if inp.delta_v_ms > 0 and inp.isp_s > 0:
        eps_max = structural_index_ceiling(inp.delta_v_ms, inp.isp_s)
        for e in estimates:
            eps = e.total_kg / (e.total_kg + inp.prop_mass_kg) \
                if (e.total_kg + inp.prop_mass_kg) > 0 else 0.0
            if eps > eps_max:
                e.notes.append(f"⚠ ε={eps:.3f} exceeds feasibility ceiling "
                               f"ε_max={eps_max:.3f} (Δv={inp.delta_v_ms:.0f}, "
                               f"Isp={inp.isp_s:.0f}s)")

    report = (divergence_report(stated_dry_kg, estimates, inp.prop_mass_kg)
              if stated_dry_kg else [])
    return estimates, report


def analyse_solid(inp: SolidStageInputs,
                  stated_dry_kg: float = 0.0) -> tuple[list[MassEstimate],
                                                       list[Divergence]]:
    comp = estimate_solid_stage(inp)
    # L/D for the Lewis slenderness correction.  NOTE: this uses the stage
    # length_m as a proxy for motor length; the Lewis fit wants the MOTOR
    # (case+nozzle) length, so a stage length padded with interstages/skirts
    # will overstate L/D and over-predict inert.  Supply motor length for an
    # accurate correction.
    ld = (inp.length_m / inp.diameter_m
          if inp.length_m > 0 and inp.diameter_m > 0 else 0.0)
    aggs = aggregate_estimates(inp.prop_mass_kg, is_solid=True,
                               casing=inp.casing, zeta=inp.mass_fraction,
                               ld_ratio=ld)
    estimates = [comp] + aggs
    report = (divergence_report(stated_dry_kg, estimates, inp.prop_mass_kg)
              if stated_dry_kg else [])
    return estimates, report


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------
def _print_analysis(estimates, report):
    for e in estimates:
        print()
        print(e.table())
    if report:
        print("\nDivergence of stated dry mass vs. estimates:")
        print(format_divergence(report))


def _demo():
    print("=" * 70)
    print("DEMO 1 — liquid stage  (Scud-B-like LOX-less storable SRBM, "
          "single stage)")
    print("=" * 70)
    inp = LiquidStageInputs(propellant="IRFNA/UDMH", prop_mass_kg=3771,
                            thrust_n=13_000 * _G0, n_engines=1,
                            diameter_m=0.88, length_m=11.25,
                            chamber_pressure_pa=6.9e6)
    est, rep = analyse_liquid(inp, stated_dry_kg=1235)
    _print_analysis(est, rep)

    print("\n" + "=" * 70)
    print("DEMO 2 — LOX/LH2 SSTO-class stage (Akin ENAE 791 worked example)")
    print("=" * 70)
    inp = LiquidStageInputs(propellant="LOX/LH2", prop_mass_kg=135_800,
                            thrust_n=6 * 324_900, n_engines=6,
                            expansion_ratio=30, diameter_m=4.2, length_m=40,
                            chamber_pressure_pa=6.9e6, gross_mass_kg=153_000)
    est, rep = analyse_liquid(inp, stated_dry_kg=12_240)
    _print_analysis(est, rep)

    print("\n" + "=" * 70)
    print("DEMO 3 — solid motor (Minuteman-III-1st-stage-like)")
    print("=" * 70)
    inp = SolidStageInputs(prop_mass_kg=20_780, thrust_n=792_000,
                           chamber_pressure_pa=4.5e6, diameter_m=1.67,
                           casing="steel", propellant="APCP")
    est, rep = analyse_solid(inp, stated_dry_kg=2300)
    _print_analysis(est, rep)


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Estimate rocket-stage dry (inert) mass and report "
                    "divergence from known mass relationships.")
    sub = p.add_subparsers(dest="cmd")

    p.add_argument("--demo", action="store_true",
                   help="run built-in demonstration cases")

    pl = sub.add_parser("liquid", help="liquid bipropellant stage")
    pl.add_argument("--prop", default="LOX/RP1",
                    help=f"propellant combo {available_propellants()}")
    pl.add_argument("--tank-material", default="aluminium",
                    choices=["aluminium", "al-li", "composite", "steel"],
                    help="tank wall material (scales tank mass)")
    pl.add_argument("--prop-mass", type=float, required=True, help="kg")
    pl.add_argument("--thrust", type=float, required=True, help="N (total vac)")
    pl.add_argument("--engines", type=int, default=1)
    pl.add_argument("--expansion", type=float, default=30.0)
    pl.add_argument("--pc", type=float, default=6.9e6, help="chamber pressure Pa")
    pl.add_argument("--diameter", type=float, default=0.0, help="m")
    pl.add_argument("--length", type=float, default=0.0, help="m")
    pl.add_argument("--gross-mass", type=float, default=0.0, help="kg")
    pl.add_argument("--fairing-area", type=float, default=0.0, help="m²")
    pl.add_argument("--tank-model", default="akin_volume",
                    choices=["akin_volume", "akin_offset", "physics", "averaged"],
                    help="tank sizing model")
    pl.add_argument("--physics-tank", action="store_true",
                    help="alias for --tank-model physics (Hutchinson GT-STRESS)")
    pl.add_argument("--lateral-g", type=float, default=0.5,
                    help="design lateral load factor for tank bending")
    pl.add_argument("--ullage", type=float, default=2.5e5,
                    help="design tank/ullage pressure (Pa)")
    pl.add_argument("--shell-config", default="integral",
                    choices=["integral", "z-stiffened", "truss-core"])
    pl.add_argument("--kappa-e", type=float, default=0.0,
                    help="engine-mass ratio M_eng/M_struct for Shu method "
                         "(0 → default by --stage-role)")
    pl.add_argument("--stage-role", default="", choices=["", "lower", "upper"],
                    help="picks the default κ_E when --kappa-e is 0")
    pl.add_argument("--delta-v", type=float, default=0.0,
                    help="stage Δv (m/s) for the Goldyn feasibility ceiling")
    pl.add_argument("--isp", type=float, default=0.0,
                    help="stage Isp (s) for the Goldyn feasibility ceiling")
    pl.add_argument("--no-avionics", action="store_true",
                    help="stage does not carry guidance avionics (booster)")
    pl.add_argument("--vehicle-gross", type=float, default=0.0,
                    help="vehicle GLOW (kg) for avionics sizing; 0 → stage gross")
    pl.add_argument("--stated-dry", type=float, default=0.0, help="kg")

    ps = sub.add_parser("solid", help="solid rocket stage")
    ps.add_argument("--prop-mass", type=float, required=True, help="kg")
    ps.add_argument("--thrust", type=float, default=0.0, help="N total")
    ps.add_argument("--pc", type=float, default=0.0, help="chamber pressure Pa")
    ps.add_argument("--casing", choices=["steel", "composite"], default="steel",
                    help="case material; reported for both the Zandbergen 2026 "
                         "and Lewis 2026 (NG-catalog, best-in-class) regressions")
    ps.add_argument("--zeta", type=float, default=0.0,
                    help="optional ASSUMED propellant mass fraction — restates "
                         "your assumption in kg, not a prediction")
    ps.add_argument("--diameter", type=float, default=0.0, help="m")
    ps.add_argument("--length", type=float, default=0.0,
                    help="motor length incl. nozzle, m (with --diameter enables "
                         "the Lewis L/D correction; use MOTOR length, not a stage "
                         "length padded with interstages/skirts)")
    ps.add_argument("--gross-mass", type=float, default=0.0, help="kg")
    ps.add_argument("--no-avionics", action="store_true",
                    help="stage does not carry guidance avionics (booster)")
    ps.add_argument("--vehicle-gross", type=float, default=0.0,
                    help="vehicle GLOW (kg) for avionics sizing; 0 → stage gross")
    ps.add_argument("--stated-dry", type=float, default=0.0, help="kg")

    args = p.parse_args(argv)

    if args.demo or args.cmd is None:
        _demo()
        return 0

    if args.cmd == "liquid":
        inp = LiquidStageInputs(
            propellant=args.prop, prop_mass_kg=args.prop_mass,
            thrust_n=args.thrust, n_engines=args.engines,
            expansion_ratio=args.expansion, chamber_pressure_pa=args.pc,
            diameter_m=args.diameter, length_m=args.length,
            gross_mass_kg=args.gross_mass, fairing_area_m2=args.fairing_area,
            tank_material=args.tank_material,
            include_avionics=not args.no_avionics,
            vehicle_gross_kg=args.vehicle_gross,
            tank_model=args.tank_model, physics_tank=args.physics_tank,
            lateral_g=args.lateral_g, ullage_pa=args.ullage,
            shell_config=args.shell_config, kappa_e=args.kappa_e,
            stage_role=args.stage_role, delta_v_ms=args.delta_v, isp_s=args.isp)
        est, rep = analyse_liquid(inp, args.stated_dry)
        _print_analysis(est, rep)
    elif args.cmd == "solid":
        inp = SolidStageInputs(
            prop_mass_kg=args.prop_mass, thrust_n=args.thrust,
            chamber_pressure_pa=args.pc, casing=args.casing,
            mass_fraction=args.zeta,
            diameter_m=args.diameter, length_m=args.length,
            gross_mass_kg=args.gross_mass,
            include_avionics=not args.no_avionics,
            vehicle_gross_kg=args.vehicle_gross)
        est, rep = analyse_solid(inp, args.stated_dry)
        _print_analysis(est, rep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
