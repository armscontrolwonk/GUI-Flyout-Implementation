# Thrusty — UI Design Spec ("Journal / Minimal")

> **Implementation status.**  V1 (the "global coat") is BUILT: `theme.py`
> tokens, app-wide named-font typography (sans text / mono numbers), `clam`
> ttk restyle, Matplotlib journal rc + ink/accent2 axis mapping, timeline
> tints, brand stripe.  V2 (rail underline controls), V3 (status strip +
> tab underlines), V4 (dialogs) are pending.
>
> **Agreed carve-outs from "solely visual" (2026-07-24):**
> 1. **Semantic colors are exempt from the palette rules.**  The four
>    survival-tier colors (`survivability_report.SURVIVAL_TIERS`: green =
>    experience, blue = design, yellow = beyond, red = fail) are the
>    evidence ladder (METHODS §13.5), not decoration.  They survive the
>    retheme untouched; tier-blue is a verdict, not an accent.
> 2. **Native macOS menu bar and window chrome are kept.**  §4's in-window
>    menu/title bar describes the HTML mockup; Tk menus stay in the system
>    bar.  The 3px brand stripe is retained (top of the content area).
> 3. **The status strip (§6) is a deliberate small re-layout**, scheduled
>    with V3, not V1.
> 4. Plot font SIZES are untouched (the 6-plot grid uses deliberately small
>    type); the journal rc governs colors/spines/grids only.

Implementation reference for the Thrusty desktop app (Python/Tkinter +
Matplotlib).  This documents the *visual system* only; behavior/physics are
unchanged.  Single source of truth in code: `theme.py`.

---

## 1. Design principles

1. **Two fonts, one job each** — sans-serif for all *text*; a monospace face
   for all *numbers/readouts*. Never mix them within a word.
2. **One accent, one signal** — slate for primary/interactive; red-orange
   reserved for the *secondary plot axis* (and nothing else).
3. **Rules, not boxes** — separate with hairlines and whitespace. Controls are
   underlined, not filled boxes. Minimal borders, near-zero corner rounding.
4. **The data is the ink** — plots are light-gridded and monochrome; the curve
   is the darkest thing on screen.
5. **Quiet chrome** — white surfaces, muted gray labels, restrained shadows.

## 2. Color tokens (see theme.py)

| Token | Hex | Use |
|---|---|---|
| `INK` | `#1a1a1a` | Primary text, primary plot series + its (left) axis |
| `SUB` | `#9a9a9a` | Labels, captions, x-axis ticks, muted text |
| `ACCENT` | `#334155` | Interactive: buttons, active tab, chevrons, links, focus |
| `ACCENT2` | `#cf5a2e` | **Secondary plot axis only**: right-axis ticks/title, its dashed series |
| `LINE` | `#ececec` | Hairline separators, faint borders |
| `UNDERLINE` | `#cfcfcf` | Control underlines (V2) |
| `GRID` | `#f3f3f3` | Plot gridlines |
| `BG` | `#ffffff` | All surfaces |
| `RED` | `#9a3535` | Negative deltas, error verdict text |
| `GREEN` | `#3f6b4f` | Positive/OK status (muted) |
| `TINT_KEY` / `TINT_DEBRIS` / `TINT_ZEBRA` | `#eef4ff` / `#fff7e8` / `#fafafa` | Timeline row tints |

## 3. Typography

Sans (text): IBM Plex Sans → Helvetica Neue → system.  Mono (numbers):
IBM Plex Mono → Menlo → Consolas.  Applied by reconfiguring Tk's NAMED fonts
(`TkDefaultFont`, `TkFixedFont`, …) so every widget inherits; per-widget sizes
unchanged in V1.  Install IBM Plex locally to get the intended faces.

## 4–6. Layout, rail, right pane

V2/V3 scope — underline controls, link buttons, status strip, tab underline.
Native menus/title bar kept (carve-out 2).

## 7. Plots

Ink primary (solid) on the left axis; ACCENT2 secondary (dashed) on the right
axis — line color always matches its axis.  Left+bottom spines ink; top/right
hidden (the twin re-shows its right spine in ACCENT2 via
`theme.style_secondary`).  Grid `#f3f3f3` behind data.  Multi-series cycle:
INK, ACCENT, SUB, GREEN, RED (ACCENT2 never enters the cycle).

| Plot | Left (ink) | Right (ACCENT2) |
|---|---|---|
| Altitude vs Time | Altitude (km) | — |
| Speed vs Time | Speed (km/s) | Mach |
| Altitude vs Range | Altitude (km) | — |
| Ground Track | Latitude | (Longitude on x) |
| Pitch, Azimuth vs Time | Elevation (°) | Azimuth (°) |
| Dyn. Pressure, Mach vs Time | q (kPa) | Mach |
| Reentry heating | Flux q̇ (MW/m²) | Integrated load Q (MJ/m²) |

## 8. Status / verdict colors

OK `GREEN`; failure `RED`.  Survival tiers: see carve-out 1 — owned by
`survivability_report.SURVIVAL_TIERS`, exempt.

## Do / Don't

- **Do** keep numbers mono and text sans, always.
- **Do** reserve red-orange strictly for the secondary axis.
- **Don't** reintroduce filled/rounded control boxes, blue *accents* (tier-blue
  is a verdict, exempt), or gradient fills under plot curves.
