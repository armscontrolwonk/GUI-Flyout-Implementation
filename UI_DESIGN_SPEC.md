# Thrusty — UI Design Spec ("Journal / Minimal")

> **Implementation status — V1 tried and PARTIALLY ROLLED BACK (2026-07-24).**
> The ttk/`clam` widget restyle and app-wide retypography were built, judged
> on screen, and reverted: replacing native macOS aqua widgets with clam's
> flat approximations looked worse, and Tk cannot close that gap.  Widgets
> are NATIVE.  What survives (in `theme.py`) is the part the toolkit fully
> delivers: the **plot style** (§7 — ink primary curves, one reserved
> red-orange for every secondary axis, hidden top/right spines, faint grid)
> and the **timeline row tints**.  §§3–6 (typography, rail, chrome, status
> strip) are RETIRED for Tk; they remain here as the design record and would
> apply as written only to a future non-Tk front end.
>
> **Carve-outs (unchanged):** the four survival-tier colors are semantic and
> exempt (`survivability_report.SURVIVAL_TIERS`, METHODS §13.5 — tier-blue is
> a verdict, not an accent); native macOS menu bar and window chrome are
> kept; plot font sizes untouched.

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
