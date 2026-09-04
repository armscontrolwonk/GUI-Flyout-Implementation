# References manifest — citation key → paper → where it lives

The canonical paper library is the **Google Drive "Thrusty" folder**
(https://drive.google.com/drive/folders/1IsW0ZoWI1YPSvYHITmo55Fs_1e8AyJ3H).
Policy: **papers live in Drive, not the repo.**  The core physics
citations were uploaded 2026-07-29; the FULL ~107-PDF `data/` corpus was
uploaded 2026-08-17 (verified file-by-file against the repo) and the
repo copies were deleted the same day.  "Repo file" names below are
therefore historical — each file now lives in Drive under the same
name.  Clones do not shrink until the separate, deliberate history
rewrite (the blobs stay in git history until then).

Heating/TPS sources have their own annotated bibliography in
`HEATING_TPS_REFERENCES.md`; the grid-fin corpus is cited inline in
`grid_fin_sizing.py`.  This manifest covers the citation keys the aero and
trajectory code names directly.

## Core citation keys

| Key | Citation | What the code uses it for | Repo file | Drive |
|---|---|---|---|---|
| **AEDC-TDR-64-25** | Clark, E. L. & Trimmer, L. L., *Equations and Charts for the Evaluation of the Hypersonic Aerodynamic Characteristics of Lifting Configurations by the Newtonian Theory*, AEDC, 1964 | Modified-Newtonian K·cos²η component build-up: closed-form swept wedge (§2.1.5), cone frustum (§2.2.5), spherical-segment blunting, composite superposition (Phase 2/2b lifting-body aero in `booster_models.py`) | `data/AD0431848.pdf` | [AD0431848.pdf](https://drive.google.com/file/d/1t1H7-BNiU_iJ7Bc55-EKdtY88FV3BofJ/view) |
| **NASA TR R-127** | Wells, W. R. & Armstrong, W. O., *Tables of Aerodynamic Coefficients Obtained from Developed Newtonian Expressions for Complete and Partial Conic and Spheric Bodies…*, 1962 | The conic/spheric closed forms behind `cd_cone_hypersonic` / `cd_blunted_cone_newtonian` (retired the "Ref (4) Ch. 5" provenance wart) | `data/19630006549.pdf` | [19630006549.pdf](https://drive.google.com/file/d/1qUG3uucFbxKp7OJj_enNLtkJfSJoGN-V/view) |
| **NASA TN D-2942** | Fetterman, D. E., *Favorable Interference Effects on Maximum Lift-Drag Ratios of Half-Cone Delta-Wing Configurations at Mach 6.86*, 1965 | Half-cone + delta-wing measured anchors (L/D Figs. 3–6; Fig. 6b C_N/C_A component anchor; the Phase-3 offset-polar C_N zero-crossing) | **no PDF** (worked from extracted text) | **not yet in Drive** |
| **NASA TN D-2956** | Fetterman, Henderson, Bertram & Johnston, *Studies Relating to the Attainment of High Lift-Drag Ratios at Hypersonic Speeds*, 1965 | Delta-wing ≡ equal-AR/area wedge equivalence; flat-bottom superiority; interference dissipation by ~M 11 | **no PDF** (worked from extracted text) | **not yet in Drive** |
| **Grant & Braun 2010** (AIAA 2010-1212) | Grant, M. J. & Braun, R. D., *Analytic Hypersonic Aerodynamics for Conceptual Design of Entry Vehicles*, 48th AIAA Aerospace Sciences Mtg | Superposition-with-rescaling (Eq. 23); sharp-biconic peak-L/D contour anchors 1.864 / 2.011 (hit to 0.2 % / 0.05 %) | `data/48th-aiaa-aerospace-science.pdf` | [48th-aiaa-aerospace-science.pdf](https://drive.google.com/file/d/1UwoilfWZikna7yOrzENbHHiORD3gQLXF/view) |
| **Corda 1988** (AIAA 88-0369) | Corda, S. & Anderson, J. D., *Viscous Optimized Hypersonic Waveriders Designed from Axisymmetric Flow Fields*, 1988 | Eckert reference-temperature C_f; viscous-waverider L/D ceiling (≈7 @ M6 → ≈6 @ M14); the Fetterman validation case | `data/corda1988.pdf` | [corda1988.pdf](https://drive.google.com/file/d/1MFHLpe8xjBK0jiyYuWjcNfv1A_YM4ngK/view) |
| **Candler S&GS 30** | Candler, G. V. & Leyva, I. A., *CFD Analysis of the Infrared Emission from a Generic Hypersonic Glide Vehicle*, Science & Global Security 30(3), 2022 | Flat-bottom windward anchor (`BODY_FLUX_FRACTION_FLAT` = 0.018, `test_candler_windward_anchor.py`); L/D(α) consistency guard; the Tracy & Wright critique scoping | `data/sgs30candler.pdf` | [sgs30candler.pdf](https://drive.google.com/file/d/15uGsva4JLQTtoCnFy9-FauwNCSn8mbK6/view) |
| **Lobanovskii 1983** | Lobanovskii, Yu. I., *Maximal Lift-Drag Ratio of Wing-Cone and Wing-Half-Cone Combinations…*, Izv. AN SSSR MZhG, 1983 (Springer BF01090577) | The asymmetric-body drag polar as a quadratic trinomial — minimum drag at nonzero lift, the Phase-3 C_L0 offset | `data/BF01090577.pdf` | [BF01090577.pdf](https://drive.google.com/file/d/1eDDwmhE-HbvOyyfu_l-wq3upROrzjrAK/view) |
| **Heybey** | Heybey, W. H., *Newtonian Aerodynamics for General Body Shapes with Several Applications*, NASA TM X-53391, 1966 | Newtonian general-shape background for the Phase-2 build-ups | `data/19660012440.pdf` | [19660012440.pdf](https://drive.google.com/file/d/19A6oGCD2ZrifzVgfCY_m2IqY2MCtzy1T/view) |
| **Tauber-Sutton 1991** | Tauber, M. E. & Sutton, K., *Stagnation-Point Radiative Heating Relations for Earth and Mars Entries*, JSR 28(1), 1991 | Radiative gas heating (zero below 9 km/s — inactive for HGV glide; the Candler-Tauber scoping) | `data/tauber-sutton-1991-stagnation-radiative-heating-earth-mars-jsr-28-1.pdf` | [tauber1991.pdf](https://drive.google.com/file/d/1Ppp6P_SOfPV6pqOtgstbKgsoj48SQWZy/view) |
| **Simon & Blake 1999** (AIAA 99-4258) | Simon, J. M. & Blake, W. B., *Missile Datcom: High Angle of Attack Capabilities*, AFRL, Wright-Patterson AFB, 1999 | The STATIONS the trim-gate moment balance acts through: "the center of pressure of the body at large angles of attack is effectively at the planform centroid", and the two-station form C_m = (x_ac − x_cg)·C_N,potential + (x_c − x_cg)·C_N,viscous (their Eq. 6) with the fin's viscous part at the panel centroid. Describes the same Allen-Perkins / Jorgensen build-up `glider_ld.py` implements. Sources `body_planform_centroid_m` and `fin_centroid_aft_le_m` (`glider_ld.py`) and `_cm()` (`trim_gate.py`) | **no PDF in repo** (read from primary, 2026-09-03) | **not yet in Drive** |
| **Moore, McInville & Hymer 1996** | Moore, F. G., McInville, R. M. & Hymer, T., *Aeroprediction Code for Angles of Attack Above 30 Degrees*, JSR 33(3), May-June 1996 | Confirms the STRUCTURE of both open trim-gate items. Body-alone c.p. is "determined by summing the separately determined linear and nonlinear contributions to the total moment and then dividing by the combined normal force" (their Eq. 3) — the normal-force-weighted two-station mean `trim_gate` uses. For control deflection it confirms the pair (k_W(B), k_B(W)) mirrors the AoA pair (K_W(B), K_B(W)), and that "k_B(W) is still defined by slender-body theory". The numeric constants live in their Ref. 3 (NSWCDD/TR-94/379), not here | **no PDF in repo** (read from primary, 2026-09-04) | **not yet in Drive** |
| **Moore & Moore (AIAA 1.A32074)** | Moore, F. G. & Moore, L. Y., *Approximate Method to Calculate Nonlinear Rolling Moment due to Differential Fin Deflection*, JSR (DOI 10.2514/1.A32074) | Roll-plane: roll driving moment from differential fin deflection, fin-to-fin interference, spanwise c.p. NOT APPLICABLE to Thrusty's 3-DOF longitudinal model; recorded so a future reader knows it was assessed and why it is unused | **no PDF in repo** (read from primary, 2026-09-04) | **not yet in Drive** |
| **Hemsch & Nielsen 1983** | Hemsch, M. J. & Nielsen, J. N., *Equivalent Angle-of-Attack Method for Estimating Nonlinear Aerodynamics of Missile Fins*, JSR 20(4), 1983 | Defines the two factors behind `trim_gate`'s `control_eff`: the Beskin upwash factor K_w (angle-of-attack case) and the fin deflection factor A_ji (Eqs. 11–12), the latter tabulated for slender-body theory vs a/s_m in their Table 1. Establishes that control effectiveness is a FUNCTION of body-radius/semispan, not a constant. Also the equivalent-angle-of-attack method for nonlinear fin behaviour above the linear range. NOT yet implemented — see TODO.md item (c) | **no PDF in repo** (read from primary, 2026-09-04) | **not yet in Drive** |
| **Sooy & Schmidt 2005** | Sooy, T. J. & Schmidt, R. Z., *Aerodynamic Predictions, Comparisons, and Validations Using Missile DATCOM (97) and Aeroprediction 98 (AP98)*, JSR 42(2), March-April 2005 | Bounds the ACCURACY OF THE REFERENCE the trim gate is validated against: DATCOM centre-of-pressure error vs wind tunnel < 2 % of body length at any AoA for body-wing-tail (M1.5, M4.6) and body-tail (M2.0). Also publishes the convention x_cp = −C_m/C_N (their Eq. 4) used by `validation/datcom/compare_datcom.py`. Their body-ALONE case 3 is explicitly inconclusive (biased test data) and is NOT used | **no PDF in repo** (read from primary, 2026-09-03) | **not yet in Drive** |
| **Sutton-Graves 1971** | Sutton, K. & Graves, R. A., *A General Stagnation-Point Convective-Heating Equation for Arbitrary Gas Mixtures*, NASA TR R-376, 1971 | The stagnation convective-heating constant (k = 1.7415e-4) at the root of the heating chain | `data/sutton-graves-1971-stagnation-point-convective-heating-nasa-tr-r-376-ntrs-19720003329.pdf` | [sutton-graves-1971…ntrs-19720003329.pdf](https://drive.google.com/file/d/1wBfCRSnc8R-sijFD3qotEMgiI85LwXdw/view) |

## Supporting material in the Drive Thrusty folder

- **Grant 2012** — Grant, M. J., *Rapid Simultaneous Hypersonic Aerodynamic
  and Trajectory Optimization for Conceptual Design*, PhD thesis, Georgia
  Tech (the long-form of Grant & Braun 2010):
  [book-98619.pdf](https://drive.google.com/file/d/1gtgtjxstwMKQUzF61ixrwARu80LtJFSS/view)
  (also `data/book-98619.pdf`).
- **AST 148 (2024) 109092** — *Hypersonic boost-glide systems: flight
  mechanics and plasma parameters evaluation through aero-thermo-chemical
  CFD* — boost-glide CFD cross-check corpus:
  [1-s2.0-S1270963824002256-main.pdf](https://drive.google.com/file/d/1J036ta02-GbVglShxpFk3ymy3lro5kzM/view)
  (also `data/1-s2.0-S1270963824002256-main.pdf`).
- **tantony-msthesis-final.pdf** — MS thesis (Drive only; not cited by the
  code):
  [link](https://drive.google.com/file/d/1dV-PqVAvTr737ZZJlJvlxTXZVwLv2uO1/view).

## Known gaps (updated 2026-08-17)

1. **Fetterman TN D-2942 and D-2956 have no PDF anywhere** — the Phase-2b/3
   anchors were worked from extracted text that lived only in a session
   scratchpad.  Both are public NASA NTRS documents; downloading them into
   the Drive Thrusty folder closes the gap.  (The last remaining gap.)
2. ~~TR R-127 and Sutton-Graves TR R-376 not mirrored to Drive~~ — CLOSED
   2026-08-17 (both in the full-corpus upload; linked above).
3. ~~The wider `data/` corpus is repo-only~~ — CLOSED 2026-08-17: all 107
   PDFs uploaded to Drive (verified file-by-file), repo copies deleted.
   Remaining: the deliberate history rewrite for the actual clone-size
   win, and optionally sorting the flat Drive upload into topic
   subfolders.

## Environmental documentation (vehicle descriptions)

U.S. NEPA documents for the hypersonic flight-test campaigns. They carry the
only official statements of payload mass and booster configuration used in
the vehicle library. Held in the maintainer's Drive under `LRHW/CPS/EIS`, not
the "Thrusty" paper folder.

| Key | Citation | What the library uses it for | Repo file | Drive |
|---|---|---|---|---|
| **FE-2 EA 2019** | U.S. Navy, Strategic Systems Programs, *Final Environmental Assessment / Overseas Environmental Assessment, Navy Flight Experiment-2 (FE-2)*, December 2019 | §2.5.6: "up to 454 kg (1,000 lb) of tungsten alloy" for the FE-2 payload analysis; Table 2-2 payload materials. Basis of the 450 kg C-HGB mass. | — | `LRHW/CPS/EIS/FE-2_EA.pdf` |
| **FT-3 BA 2020** | U.S. Army RCCTO / SMDC, *Biological Assessment for Hypersonic Flight Test-3 Activities*, 22 September 2020 | §2.2.1 Launch Vehicle Description: FT-3 payload ≈ 350 kg (750 lb), "similar to" FE-2 with 10 % of its tungsten; FT-3 booster stack (Table 3). | — | `LRHW/CPS/EIS/FT-3_BA.pdf` |
| **AHW EA 2011** | U.S. Army SMDC/ARSTRAT, *Advanced Hypersonic Weapon Program Environmental Assessment*, June 2011 | §2.1.2 and Fig. 2.1.2-1: Strategic Target System configuration (Polaris A3 stages 1–2, Orbus 1a stage 3), 30,541 lb total propellant, ≈75,000 lb thrust. Basis of `STARS-1.booster.json` configuration. | — | `LRHW/CPS/EIS/2011-pmrf-…advanced-hypersonic-weapon-program-ea.pdf` |
