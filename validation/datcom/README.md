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
| 2 | 2.23 @ 16° | 2.09 @ 16° | −6% |
| 3 | 2.71 @ 14° | 2.36 @ 14° | −13% |
| 5 | 3.51 @ 10° | 2.81 @ 12° | −20% |

Zero-lift drag agrees within ~10% and the best-glide AoA matches closely.
`glider_ld` runs slightly conservative (under-predicts L/D — the safe direction
for range). The cross-check originally exposed a real error — the planform area
used the cone-only triangle `½Ld` instead of the body's true (nose + cylinder)
planform — which is now fixed in `glider_ld.py`. The residual gap grows with
Mach because `glider_ld` holds the crossflow drag coefficient constant
(`C_dn = 1.2`) while the true value rises with crossflow Mach `M·sinα`.
