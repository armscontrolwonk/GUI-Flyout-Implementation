# `estimate_damping()` — design spec (not yet implemented)

A design-time helper that, for a given glide RV, suggests a physically-grounded
phugoid **damping ratio ζ** (the `glider_damping_zeta` knob of the
`damped_glide` mode; see `DAMPED_GLIDE_MEMO.md`). The companion analysis scripts
`damping_topout.py` and `cbo_glide_check.py` already compute the underlying
lift-authority numbers; this spec turns them into a single "your ζ should be
around *n*" estimate. Literature support and citations: `docs/cl_margin_references.md`.

## Why an estimate is possible

ζ is **a property of the guidance, not the airframe**: the open-loop skip
phugoid is essentially undamped, and damping it requires *spare lift* to apply
the feedback. The achievable damping is therefore set by how much lift the
vehicle can pull **above** its max-L/D trim — its **C_L margin**. From the
lift-authority analysis (`damping_topout.py`):

- a bare biconic glides right at its lift margin (Λ = a_L,max/g_eff ≈ 1.1–1.2),
  so ζ tops out around **0.2**;
- a small angle-of-attack / control-surface C_L margin is highly leveraged:
  **+12 % C_L → ζ_max ≈ 0.4, +30 % → ≈ 0.7** (because the trim headroom is tiny).

So the estimate reduces to: *how much C_L margin does this airframe have?*

## The estimate chain

Inputs (from the RV / a few new fields): `beta_kg_m2`, `glider_LD`,
`glider_pullup_g_max`, a **control-surface descriptor** (see tiers), and a
representative glide state (V, h) — taken from a trajectory run or a default
(V ≈ 4 km/s, h ≈ h_eq).

1. **Trim C_L** at the equilibrium glide: lift balances `g_eff = g − V²/r`, so
   `C_L,trim = g_eff·m/(q·S_ref)` — equivalently the C_L the vehicle already
   flies at max-L/D.
2. **Usable extra lift** `ΔC_L = min(` of three bounds `)`:
   - **Control-surface (Newtonian):** `ΔC_L = (S_flap/S_ref)·1.84·[sin²(α+δ) − sin²α]·cosα`
     evaluated at usable deflection **δ ≤ ~10–15°** (separation limit;
     derate ×~0.8 for real-gas above M≈7);
   - **Range-cost knee:** the C_L that costs ≤ ~8 % of L/D, i.e.
     **C_L ≤ ~1.5·C_L,opt** (from `(L/D)/(L/D)_max = 2n/(1+n²)`);
   - **Aerodynamic ceiling:** `C_L,max ≈ 0.4–0.5` (cone/biconic).
3. **C_L margin** `M = (C_L,trim + ΔC_L)/C_L,trim`.
4. **ζ_max** from the lift-authority bound
   `ζ_max ≈ (M·a_L,trim − g_eff)/(2·ω_p·V·γ*)`,
   `ω_p = √(g_eff/H_ρ)`, `γ* = 2·(L/D)·H_ρ·g/V²` (as in `damping_topout.py`).
5. **Report** `ζ ≈ min(0.7, ζ_max)`, with the band it falls in.

## The "at minimum, has / doesn't-have controls" tiers

When detailed control-surface geometry is unknown, fall back to a tier from a
single descriptor:

| Control surfaces | C_L margin | **Suggested ζ** | Basis |
|---|---|---|---|
| **None** (bare biconic / ballistic body) | ~0 | **≈ 0.2** | finless lift-authority floor (`damping_topout.py`) |
| **Small fins / flaps** (few-% area, modest δ) | +10–25 % | **≈ 0.3–0.4** | +12 % C_L → ζ≈0.4 |
| **Substantial elevons / body flaps / lifting fins** (Shuttle / HL-20 / **C-HGB** class) | ≳ +30 % | **≈ 0.5–0.7** | fins *are* the lift reference (Gulan); +30 % → ζ≈0.7 |

The **C-HGB sits in the third tier**: Gulan (2024) references its lift to the
**fin area** — the four fins (span ≈ 1.8× body diameter) are primary lifting/
control surfaces — so it has ample margin and ζ ≈ 0.5–0.7 is appropriate; the
ζ ≈ 0.2 floor applies only to a *finless* body. Choosing a lower ζ for a finned
vehicle is then a **range/energy choice** (spending lift on damping costs ~8 %
L/D per 1.5× C_L), not an airframe limit.

## Default parameters & open items

- **S_flap/S_ref default ≈ 5–8 %** when only "has controls" is known
  (Shuttle elevon ≈ 8 %, body flap ≈ 5 %; `docs/cl_margin_references.md` §4).
  *Open:* the exact C-HGB fin area (Gulan Fig. 24 / its ref [35]) would replace
  this with a vehicle-specific number.
- `H_ρ` clamped 4–12 km; g-limit `glider_pullup_g_max` (default 10) rarely
  binds (aero/C_L-limited, not structure-limited).

## Validation hooks

- The estimate must reproduce the observed behavior: at the C-HGB's β/L-D the
  glide sits at 30–40 km and ζ ≈ 0.4–0.5 reproduces the CBO "majority of glide
  in 30–40 km" (`cbo_glide_check.py`).
- ζ = 0 must remain bit-exact `skip_glide` (already tested in
  `damped_glide_smoke_test.py`).

## Caveats (carried into any UI text)

ζ ≈ 0.7 is the textbook control-design default, not a fit to these vehicles;
real boost-glide vehicles plausibly run lower. The estimate is order-of-~factor-2
(the ζ_max formula is a first-cut), the deflection/real-gas/heating limits cap
the usable margin, and the result is a *suggested starting value*, not a
measurement. Implementation status: **spec only — not wired into the tool.**
