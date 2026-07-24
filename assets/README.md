# assets/

`thrusty.png` — the app icon (Thrusty ruler mascot).  Loaded at startup by
`BoosterFlyoutApp.__init__` via `iconphoto`: on macOS it replaces the stock
Python rocket in the Dock while the app runs; on Windows/Linux it sets the
window/taskbar icon.  Absent file = stock icon, silently.

Wanted: square, transparent-background PNG, ≥512×512 (1024×1024 ideal —
also feeds the .icns build for a Thrusty.app bundle).
