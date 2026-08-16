# References manifest — citation key → paper → where it lives

The canonical paper library is the **Google Drive "Thrusty" folder**
(https://drive.google.com/drive/folders/1IsW0ZoWI1YPSvYHITmo55Fs_1e8AyJ3H),
where the core physics citations were uploaded 2026-07-29.  Policy: **new
papers go to Drive, not the repo.**  The PDFs under `data/` predate that
policy and stay for now — deleting them later will NOT shrink clones (the
blobs stay in git history); an actual shrink needs a deliberate history
rewrite.

Heating/TPS sources have their own annotated bibliography in
`HEATING_TPS_REFERENCES.md`; the grid-fin corpus is cited inline in
`grid_fin_sizing.py`.  This manifest covers the citation keys the aero and
trajectory code names directly.

## Core citation keys

| Key | Citation | What the code uses it for | Repo file | Drive |
|---|---|---|---|---|
| **AEDC-TDR-64-25** | Clark, E. L. & Trimmer, L. L., *Equations and Charts for the Evaluation of the Hypersonic Aerodynamic Characteristics of Lifting Configurations by the Newtonian Theory*, AEDC, 1964 | Modified-Newtonian K·cos²η component build-up: closed-form swept wedge (§2.1.5), cone frustum (§2.2.5), spherical-segment blunting, composite superposition (Phase 2/2b lifting-body aero in `booster_models.py`) | `data/AD0431848.pdf` | [AD0431848.pdf](https://drive.google.com/file/d/1t1H7-BNiU_iJ7Bc55-EKdtY88FV3BofJ/view) |
| **NASA TR R-127** | Wells, W. R. & Armstrong, W. O., *Tables of Aerodynamic Coefficients Obtained from Developed Newtonian Expressions for Complete and Partial Conic and Spheric Bodies…*, 1962 | The conic/spheric closed forms behind `cd_cone_hypersonic` / `cd_blunted_cone_newtonian` (retired the "Ref (4) Ch. 5" provenance wart) | `data/19630006549.pdf` | **not yet in Drive** |
| **NASA TN D-2942** | Fetterman, D. E., *Favorable Interference Effects on Maximum Lift-Drag Ratios of Half-Cone Delta-Wing Configurations at Mach 6.86*, 1965 | Half-cone + delta-wing measured anchors (L/D Figs. 3–6; Fig. 6b C_N/C_A component anchor; the Phase-3 offset-polar C_N zero-crossing) | **no PDF** (worked from extracted text) | **not yet in Drive** |
| **NASA TN D-2956** | Fetterman, Henderson, Bertram & Johnston, *Studies Relating to the Attainment of High Lift-Drag Ratios at Hypersonic Speeds*, 1965 | Delta-wing ≡ equal-AR/area wedge equivalence; flat-bottom superiority; interference dissipation by ~M 11 | **no PDF** (worked from extracted text) | **not yet in Drive** |
| **Grant & Braun 2010** (AIAA 2010-1212) | Grant, M. J. & Braun, R. D., *Analytic Hypersonic Aerodynamics for Conceptual Design of Entry Vehicles*, 48th AIAA Aerospace Sciences Mtg | Superposition-with-rescaling (Eq. 23); sharp-biconic peak-L/D contour anchors 1.864 / 2.011 (hit to 0.2 % / 0.05 %) | `data/48th-aiaa-aerospace-science.pdf` | [48th-aiaa-aerospace-science.pdf](https://drive.google.com/file/d/1UwoilfWZikna7yOrzENbHHiORD3gQLXF/view) |
| **Corda 1988** (AIAA 88-0369) | Corda, S. & Anderson, J. D., *Viscous Optimized Hypersonic Waveriders Designed from Axisymmetric Flow Fields*, 1988 | Eckert reference-temperature C_f; viscous-waverider L/D ceiling (≈7 @ M6 → ≈6 @ M14); the Fetterman validation case | `data/corda1988.pdf` | [corda1988.pdf](https://drive.google.com/file/d/1MFHLpe8xjBK0jiyYuWjcNfv1A_YM4ngK/view) |
| **Candler S&GS 30** | Candler, G. V. & Leyva, I. A., *CFD Analysis of the Infrared Emission from a Generic Hypersonic Glide Vehicle*, Science & Global Security 30(3), 2022 | Flat-bottom windward anchor (`BODY_FLUX_FRACTION_FLAT` = 0.018, `test_candler_windward_anchor.py`); L/D(α) consistency guard; the Tracy & Wright critique scoping | `data/sgs30candler.pdf` | [sgs30candler.pdf](https://drive.google.com/file/d/15uGsva4JLQTtoCnFy9-FauwNCSn8mbK6/view) |
| **Lobanovskii 1983** | Lobanovskii, Yu. I., *Maximal Lift-Drag Ratio of Wing-Cone and Wing-Half-Cone Combinations…*, Izv. AN SSSR MZhG, 1983 (Springer BF01090577) | The asymmetric-body drag polar as a quadratic trinomial — minimum drag at nonzero lift, the Phase-3 C_L0 offset | `data/BF01090577.pdf` | [BF01090577.pdf](https://drive.google.com/file/d/1eDDwmhE-HbvOyyfu_l-wq3upROrzjrAK/view) |
| **Heybey** | Heybey, W. H., *Newtonian Aerodynamics for General Body Shapes with Several Applications*, NASA TM X-53391, 1966 | Newtonian general-shape background for the Phase-2 build-ups | `data/19660012440.pdf` | [19660012440.pdf](https://drive.google.com/file/d/19A6oGCD2ZrifzVgfCY_m2IqY2MCtzy1T/view) |
| **Tauber-Sutton 1991** | Tauber, M. E. & Sutton, K., *Stagnation-Point Radiative Heating Relations for Earth and Mars Entries*, JSR 28(1), 1991 | Radiative gas heating (zero below 9 km/s — inactive for HGV glide; the Candler-Tauber scoping) | `data/tauber-sutton-1991-stagnation-radiative-heating-earth-mars-jsr-28-1.pdf` | [tauber1991.pdf](https://drive.google.com/file/d/1Ppp6P_SOfPV6pqOtgstbKgsoj48SQWZy/view) |
| **Sutton-Graves 1971** | Sutton, K. & Graves, R. A., *A General Stagnation-Point Convective-Heating Equation for Arbitrary Gas Mixtures*, NASA TR R-376, 1971 | The stagnation convective-heating constant (k = 1.7415e-4) at the root of the heating chain | `data/sutton-graves-1971-stagnation-point-convective-heating-nasa-tr-r-376-ntrs-19720003329.pdf` | **not yet in Drive** |

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

## Known gaps (loose ends as of 2026-08-16)

1. **Fetterman TN D-2942 and D-2956 have no PDF anywhere** — the Phase-2b/3
   anchors were worked from extracted text that lived only in a session
   scratchpad.  Both are public NASA NTRS documents; downloading them into
   the Drive Thrusty folder closes the gap.
2. **TR R-127** (`data/19630006549.pdf`) and **Sutton-Graves TR R-376** are
   in the repo but not yet mirrored to Drive.
3. The wider corpus under `data/` (~100 PDFs: grid fins, TPS/ablation,
   flight-test, waveriders) remains repo-only; moving it to Drive and
   trimming the repo is the larger, deliberate step (history rewrite needed
   for an actual size win).
