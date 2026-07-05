# `estimate_damping()` — design & spec

A design-time helper + GUI button that suggests a physically-grounded phugoid
**damping ratio ζ** (the `glider_damping_zeta` knob of the `damped_glide` mode;
see `DAMPED_GLIDE_MEMO.md`). Literature support and citations:
`docs/cl_margin_references.md`. The analysis scripts `damping_topout.py` and
`cbo_glide_check.py` prototype the underlying lift-authority numbers.

Implementation: backend in `damping_estimate.py` (tested by
`test_damping_estimate.py`); GUI button + dialog in `thrusty.py`; persistence
fields on `ROParams` in `booster_models.py`.

---

## 1. What the number means — a capability ceiling with modeling uncertainty

ζ is a property of the **guidance**, not the airframe: the open-loop skip
phugoid is essentially undamped, and damping it requires *spare lift* to apply
the feedback. So the estimate's central value is the **achievable ζ ceiling
(ζ_max)** set by the vehicle's lift authority, reported **± a modeling-
uncertainty band**. The band is *epistemic* — how sure we are where the ceiling
is — and it acts as a **capability boundary** on the knob:

- **ζ below the band** → the vehicle comfortably supports it; this is a **free
  design choice** (deliberately bouncier / longer-ranged).
- **ζ inside the band** → you are **at the airframe's authority limit**, where
  "can it actually hold this damping?" is genuinely uncertain.
- **ζ above the band** → you are assuming **more control authority than the lift
  can supply** — not a choice, just unphysical.

Because it is modeling uncertainty, the band **shrinks as the user supplies
data** (control-surface geometry, a flown glide state). That is the desired
property: a *tight* band means "this really is the ceiling"; a *wide* band means
"we are guessing where the ceiling is." The point of this framing (vs. a
"plausible design range") is that it tells the user **when they are migrating
from a vehicle limit to a design choice** — the boundary sharpens exactly as
their knowledge improves. The GUI shades this band on the ζ field so the user
sees where capability ends and choice begins.

If ζ_max lands well above the textbook 0.7, the practical message is "free to
choose any ζ up to ~0.7 without hitting a limit"; if it lands low, "authority-
limited to ~ζ_max."

## 2. Inputs

**Pre-populated from the active RV** (read at button press, via
`effective_ro`): `beta_kg_m2`, `glider_LD`, `diameter_m`, `length_m`,
`nose_radius_m`, `glider_pullup_g_max`.

**User-supplied / editable:**
- **Control surfaces** (the one required input) — `none` / `small` (fins or
  flaps, few-% area) / `substantial` (elevons, body flaps, or lifting fins,
  Shuttle / HL-20 / C-HGB class). Selecting a tier alone returns a band.
- **Advanced (optional, sharpens the estimate):** control-surface area ratio
  `S_flap/S_ref`, usable deflection `δ` (default 12°, capped ~15° by
  separation), and the glide-state overrides below.

**Glide state** (V, altitude): **if a trajectory has been flown, pre-fill the
actual mid-glide V and altitude; otherwise leave blank.** Always editable, with
a **"Restore" button** that resets to the flown values (or to blank if no run).
Blank → the computation falls back to a swept representative range (see §3), so
it still returns an answer with a wider band. The glide-state overrides are
session context, **not** saved on the RV.

## 3. The estimate chain

Anchored at the **equilibrium glide**, derived from the vehicle's own β and L/D
(no trajectory run needed):

1. **Equilibrium altitude** from `ρ_eq = 2·β·g_eff / (V²·(L/D))`,
   `g_eff = g − V²/r`, inverted through the atmosphere model. For the C-HGB this
   self-consistently gives ~31–41 km across V = 3–5 km/s (the verified CBO band).
2. **Speed V:** the flown mid-glide value if available; else **sweep
   V = 3–5 km/s** and report the spread as part of the band. (V is a trajectory
   property, not a vehicle property, so it cannot be derived.)
3. **Control-surface C_L margin M = 1 + ΔC_L/C_L,trim**, with the usable lift
   increment the **minimum** of three bounds:
   - **Newtonian flap** (Grant & Braun; control surface ≈ flat plate):
     `ΔC_L = (S_flap/S_ref)·Cp,max·[sin²(α+δ) − sin²α]·cosα`, `Cp,max ≈ 1.84`,
     derated ×0.85 for real-gas (M > 7), at usable `δ ≤ 15°` (separation limit);
   - **range knee:** `ΔC_L ≤ 0.5·C_L,trim` (flying to ~1.5× C_L,opt costs only
     ~8 % L/D; beyond that range falls off steeply);
   - **aerodynamic ceiling:** `C_L,trim + ΔC_L ≤ C_L,max ≈ 0.45` (cone/biconic).
   Assumed slender-glide-body trim values: `C_L,trim ≈ 0.12`, trim `α ≈ 8°`
   (NASA M=6 biconic trim; TN D-840 / D-4098). These (with δ, V) are the
   dominant uncertainty sources reflected in the band.
4. **ζ_max** from the lift-authority bound (as in `damping_topout.py`):
   `ζ_max ≈ (M−1)·g_eff / (2·ω_p·V·γ*)`, `ω_p = √(g_eff/H_ρ)`,
   `γ* = 2·(L/D)·H_ρ·g/V²`, with a small passive floor (~0.05, atmospheric-drag
   damping). The g-limit `glider_pullup_g_max` caps `a_L,max ≤ n_max·g` (rarely
   binding — aero/C_L-limited, not structure-limited).

**Two modes:**
- **Tier-only (quick):** when no explicit `S_flap/S_ref` is given, return the
  tier's band directly (a wide modeling-uncertainty band):
  **none → 0.10 (0.00–0.20); small → 0.35 (0.25–0.45); substantial →
  0.60 (0.45–0.72).**
- **Computed (advanced):** when `S_flap/S_ref` is supplied, run steps 1–4; the
  band comes from the V-sweep plus a ~×1.4 formula factor (tighter than the
  tier band).

## 4. Output

`EstimateResult(zeta, zeta_lo, zeta_hi, h_eq_km_range, margin_M, s_flap_ratio,
notes)`. The GUI shows e.g. **"ζ ≈ 0.50 (0.38–0.65)"** with a one-line rationale
and an expandable breakdown (trim C_L → M → ζ_max). **Apply** writes the central
value to the ζ knob; the band is shown in the field tooltip and shaded on the
field. ζ is clamped to [0, 1.2]; values ≥ 0.7 are annotated "≥ textbook
well-damped — free to choose up to 0.7."

## 5. Persistence — RV-embedded

Reentry guidance stays on the RV (no separate preset; saved with the
`*.ro.json` and the missile JSON). New `ROParams` fields (serialized in
`ro_to_dict`/`ro_from_dict`):

- `glider_control_surfaces: str = "unknown"`  (`none`/`small`/`substantial`/`unknown`)
- `glider_flap_area_ratio: float = 0.0`  (0 ⇒ use the tier default)
- `glider_flap_deflection_deg: float = 0.0`  (0 ⇒ use the 12° default)

`glider_damping_zeta` (the knob) already exists. Glide-state overrides are
session-side only.

## 6. Caveats (carried into UI text)

ζ ≈ 0.7 is the textbook control default, not a fit to these vehicles; real
boost-glide vehicles plausibly run lower (a range/energy choice). The estimate
is order-of-~factor-2 (the ζ_max formula is a first-cut, and C_L,trim/α/δ are
assumed); the deflection/real-gas/heating limits cap the usable margin. The
result is a **suggested starting value with an honest band**, not a measurement.
