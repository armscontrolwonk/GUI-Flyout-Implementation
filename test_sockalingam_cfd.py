"""CFD anchor — Sockalingam & Tabiei 2009 (Int. J. Multiphysics 3(3) 293-306)
+ Tabiei & Sockalingam 2011 (J. Aerosp. Eng. 25(2) 273-281), both read from
primary and in the project archive.

FLUENT two-temperature-family nonequilibrium CFD of the IRV-2 sphere-biconic
(R_n = 0.01905 m) with an ISOTHERMAL 294.4 K wall — a cold wall, the same
assumption Sutton-Graves makes, so the flux comparison is apples-to-apples on
wall enthalpy.  Freestream per their Table 3; the comparison uses THEIR
(T∞, P∞) → ρ, not our atmosphere, to isolate the flux model.

What it shows (trajectory point 1: 66.9 km, 6.78 km/s):
  - S-G = 4.40 MW/m² vs the grid-converged FULLY CATALYTIC CFD 6.02 MW/m²
    (ratio 0.73) and vs the NONCATALYTIC case ~4.5 MW/m² (Fig. 4, digitized;
    ratio 0.98).  Physical surfaces are partially catalytic — reality lives
    between the two CFD curves — and S-G rides the low (noncatalytic) edge.
  - Same pattern at points 2-3 (55.8 / 49.3 km): S-G = 0.80-0.81 of the
    fully-catalytic values (2011 Fig. 8, digitized).
  Consistent with the Finke/DKR check (S-G ~11% below DKR in flux at
  7 km/s): Sutton-Graves is the lean member of the method family.  The
  catalycity spread (~35% here) is largest at rarefied altitudes; it narrows
  toward equilibrium at the dense-air peak-heating altitudes that drive the
  survivability verdicts.
"""

import numpy as np

import heating

_R_AIR = 287.05
_RN = 0.01905     # m — IRV-2 nose radius


def _sg(P, T, V):
    rho = P / (_R_AIR * T)
    return heating._stag_flux(np.array([rho]), np.array([V]), _RN)[0] / 1e6


def test_point1_between_catalycity_bracket():
    q = _sg(8.1757, 227.81, 6780.6)          # their Table 3, point 1
    # vs fully catalytic 6.02 MW/m² (grid-converged, stated in both papers)
    assert 0.65 <= q / 6.02 <= 0.85
    # vs noncatalytic ~4.5 MW/m² (Fig. 4, digitized) — S-G sits on this edge
    assert 0.88 <= q / 4.50 <= 1.10


def test_points_2_3_fully_catalytic_ratio():
    # 2011 Fig. 8 fully-catalytic stagnation values ~1100 / ~1650 W/cm²
    # (digitized); S-G runs ~0.8 of the fully-catalytic ceiling.
    for (P, T, V, q_fc) in [(37.362, 258.02, 6788.3, 11.0),
                            (88.118, 270.65, 6785.2, 16.5)]:
        assert 0.70 <= _sg(P, T, V) / q_fc <= 0.90
