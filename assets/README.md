# assets/

`thrusty.png` — the app icon (Thrusty ruler mascot).  Loaded at startup by
`BoosterFlyoutApp.__init__` via `iconphoto`: on macOS it replaces the stock
Python rocket in the Dock while the app runs; on Windows/Linux it sets the
window/taskbar icon.  Absent file = stock icon, silently.

Wanted: square, transparent-background PNG, ≥512×512 (1024×1024 ideal —
also feeds the .icns build for a Thrusty.app bundle).  DONE 2026-08-17:
the 3-D render is in place, `thrusty.icns` is generated from it (PIL
`Image.save`, 16→1024 px), and the repo now ships **Thrusty.app** — a
wrapper bundle whose launcher runs thrusty.py through an interpreter
copy inside the bundle, so the Dock and menu bar say "Thrusty" (not
"Python") with this icon.  Launch by double-clicking Thrusty.app in the
repo root; running `python3 thrusty.py` by hand still works as before.
