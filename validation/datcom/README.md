# glider_ld vs Digital DATCOM

Cross-validation of the no-separation whole-missile L/D build-up (`glider_ld.py`:
Jorgensen TR R-474 + Allen-Perkins NACA 1048 + Pitts-Nielsen-Kaattari NACA 1307)
against **Digital DATCOM** — the USAF stability-and-control code (AFFDL-TR-79-3032),
public domain, distributed by PDAS (<https://www.pdas.com/datcom.html>).

## Files

- `finless_body.inp` — Digital DATCOM input deck for the finless slender
  reference body (D = 0.5 m, L = 4 m, 1.5 m tangent-ogive nose), M2/3/5,
  α = 0–20°.
- `finless_body.datcom.out` — the DATCOM output for that deck (committed so the
  comparison runs without rebuilding DATCOM).
- `compare_datcom.py` — parses the `.out` and compares L/D-vs-α and C_A0 against
  `glider_ld.whole_missile_LD`.

## Run the comparison

```
python validation/datcom/compare_datcom.py
```

## Regenerate the DATCOM reference (optional)

Digital DATCOM is **not** vendored here (it is a ~51k-line Fortran program).
To regenerate `finless_body.datcom.out`:

1. Download the Digital DATCOM source (`datcom.f`) from PDAS:
   <https://www.pdas.com/datcomdownload.html>.
2. Compile:  `gfortran datcom.f -o datcom.exe -std=legacy -w`
3. Run (it prompts for the input file name):
   `echo finless_body.inp | ./datcom.exe`  → writes `datcom.out`.

Note: Digital DATCOM input cards are fixed **80 columns**; array assignments
that overflow column 80 are silently truncated, so the deck wraps `X(1)=`/`R(1)=`
across continuation lines.

## Result (summary)

| Mach | DATCOM L/D_max @ α | glider_ld L/D_max @ α | gap |
|---|---|---|---|
| 2 | 2.23 @ 16° | 2.13 @ 16° | −5% |
| 3 | 2.71 @ 14° | 2.48 @ 15° | −9% |
| 5 | 3.51 @ 10° | 3.17 @ 12° | −10% |

Zero-lift drag agrees within ~10% and the best-glide AoA matches closely.
`glider_ld` runs slightly conservative (under-predicts L/D — the safe direction
for range). The cross-check drove two sourced fixes in `glider_ld.py`:

1. **Planform area** — the original `A_p = ½Ld` (a cone-only triangle)
   underestimated the planform of a body with a long cylinder; replaced by the
   true nose + cylinder planform.
2. **Crossflow drag coefficient** — the original constant `C_dn = 1.2`
   under-predicted crossflow lift at high Mach (at a M5 best-glide AoA the
   crossflow Mach `M_n = M·sinα ≈ 1`, where the cylinder `C_dn ≈ 2.1`); replaced
   by `C_dn(M_n)` from Gowen & Perkins, NACA TN 2960 Fig. 7.

These cut the worst gap from ~20% to ~10% and flattened its Mach dependence. The
remaining ~10% is a consistent conservative bias (slender-body potential slope vs
DATCOM's fuller body-lift method).
