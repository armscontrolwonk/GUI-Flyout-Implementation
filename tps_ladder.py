"""NAS/NRC-2008 boost-glide TPS ladder: glide time -> TPS class.

Reproduces the Conventional Prompt Global Strike (CPGS) reentry-body lineage of
the National Research Council report *U.S. Conventional Prompt Global Strike*
(NRC 2008), pp. 119-121, as distilled in the project approach documents
D-2 (lineage) and D-3 (the ablation->reradiation step).

The report draws ONE TPS materials line, not a per-second ladder.  What changes
with glide time is the BINDING physics, and only the longest tier forces a new
material class:

  glide time  ->  regime          binding F.o.M.        TPS class
  --------------------------------------------------------------------------
  tens of s..~800 s  ABLATION      Q = integral q dt     carbon-phenolic (add
                     -limited      (recession & mass)    thickness) = CONVENTIONAL
       ~1000 s  <-- the step: the binding figure of merit itself changes -->
  ~1000..3600 s  RERADIATION       T_w = (q/eps.sig)^1/4 carbon-carbon (reusable,
                 -limited          (class + oxid. life)  non-ablating) = ADVANCED
  >= ~3600 s     RERADIATION       T_w + oxidation       C/C hot structure + UHTC
  (1 h soak)     -limited          dwell life            leading edges = FRONTIER

Physics (D-glide_heating): peak FLUX is velocity-locked (0.73-0.82 v_c, the same
universal curve for every glider); integrated LOAD Q scales with glide time.
CSM-1 (800 s) and CSM-2 (3,000 s) peak within 0.3% of each other; their loads
differ 4.3x.  So the material step is driven by DURATION (load/soak), not by a
hotter peak.

This module is the seconds->class lookup; it does not fly a trajectory.  Feed it
the reentry/glide-arc duration (Thrusty computes this) or a hypothetical value.
"""

# --- the two boundaries that define the step function (seconds) ---
CROSSOVER_S = 1000.0     # ablation <-> reradiation; the binding F.o.M. switches
HOUR_SOAK_S = 3600.0     # 1 h sustained soak; forces the C/C + UHTC frontier

# --- NRC-2008 CPGS lineage anchors (Figure D-2) ---
# (label, glide_time_s or 'tens', material, tier_key)
NAS_LINEAGE = [
    ("CTM (E2/LETB)",     None, "carbon-phenolic (as designed)",   "conventional"),
    ("SLGSM Mk-500 RB",    300, "carbon-phenolic (stretched)",     "conventional"),
    ("CSM-1 AMaRV",        800, "carbon-phenolic (boost-glide)",   "conventional"),
    ("CSM-2 FALCON",      3000, "carbon-carbon (novel)",           "advanced"),
    ("FALCON HTV-2",      3600, "C/C + UHTC leading edges",        "frontier"),
]

# --- material capability ladder (Figure D-3(b); continuous-use surface T, deg C) ---
MATERIAL_LADDER = [
    ("Ablative carbon-phenolic",        1200, "single-use; recession-limited; tolerates kW/cm^2 for a short soak"),
    ("Reinforced carbon-carbon (RCC)",  1650, "reusable; oxidises rapidly above ~1650 C in atomic oxygen"),
    ("C/SiC (X-38, EXPERT, IXV)",       1700, "qualified for a single re-entry mission"),
    ("C/C hot structure (HTV-2)",       1900, "1900 C surface in flight; warm structure 1090 C for 1 h"),
    ("UHTC ZrB2/HfB2 (sharp LE)",       2200, "operate >2000 C; arc-jet ~2200 C; oxidation life still open"),
]

_TIER = {
    "conventional": dict(
        name="CONVENTIONAL",
        material="carbon-phenolic ablator (stretch thickness)",
        regime="ablation-limited",
        binding="Q = integral q dt  (ablator recession & mass)"),
    "advanced": dict(
        name="ADVANCED",
        material="carbon-carbon (reusable, non-ablating)",
        regime="reradiation-limited",
        binding="T_w = (q/eps.sigma)^1/4  (material class + oxidation life)"),
    "frontier": dict(
        name="FRONTIER",
        material="C/C hot structure + UHTC leading edges",
        regime="reradiation-limited",
        binding="T_w + oxidation dwell life (1 h+ soak)"),
}


def nas_tps_tier(glide_time_s):
    """Place a glide/reentry duration (s) on the NRC-2008 CPGS ladder.

    Returns a dict:
      tier      : 'conventional' | 'advanced' | 'frontier'
      tier_name : 'CONVENTIONAL' | 'ADVANCED' | 'FRONTIER'
      regime    : 'ablation-limited' | 'reradiation-limited'
      binding   : the binding figure of merit for that regime
      material  : the TPS class NRC assigns
      nearest   : (label, seconds) of the closest lineage anchor
      note      : one-line placement summary
    """
    t = float(glide_time_s or 0.0)
    if t < CROSSOVER_S:
        key = "conventional"
    elif t < HOUR_SOAK_S:
        key = "advanced"
    else:
        key = "frontier"
    tier = _TIER[key]

    # nearest lineage anchor with a numeric time
    anchors = [(lbl, s) for (lbl, s, _m, _t) in NAS_LINEAGE if s is not None]
    nearest = min(anchors, key=lambda a: abs(a[1] - t))

    if t < CROSSOVER_S:
        note = (f"{t:.0f} s is ABLATION-limited (below the ~{CROSSOVER_S:.0f} s step): "
                f"carbon-phenolic, sized by recession/mass.")
    elif t < HOUR_SOAK_S:
        note = (f"{t:.0f} s is RERADIATION-limited (above the ~{CROSSOVER_S:.0f} s step): "
                f"needs a reusable non-ablating class (carbon-carbon).")
    else:
        note = (f"{t:.0f} s exceeds the {HOUR_SOAK_S:.0f} s (1 h) soak frontier: "
                f"C/C hot structure with UHTC leading edges.")

    return dict(tier=key, tier_name=tier["name"], regime=tier["regime"],
                binding=tier["binding"], material=tier["material"],
                nearest=nearest, note=note)


def format_ladder(glide_time_s=None):
    """Plain-text NRC-2008 ladder, with the current glide time placed on it if
    given.  Feeds the GUI 'Heating Survivability' panel; testable without a GUI."""
    lines = ["TPS-class ladder — NRC 2008 (CPGS reentry-body lineage, pp. 119-121)", ""]
    cur = nas_tps_tier(glide_time_s) if glide_time_s else None
    for lbl, s, mat, tier in NAS_LINEAGE:
        secs = "~tens of s" if s is None else f"{s:>5.0f} s"
        mark = ""
        if cur and s is not None and cur["nearest"][0] == lbl:
            mark = "  <- nearest anchor"
        lines.append(f"  {secs:>11}   {_TIER[tier]['name']:<12} {mat}{mark}")
    lines.append("")
    lines.append(f"  the step: ablation <-> reradiation at ~{CROSSOVER_S:.0f} s "
                 f"(binding F.o.M. switches Q -> T_w)")
    if cur:
        lines += ["",
                  f"This vehicle  ({glide_time_s:.0f} s glide):",
                  f"  regime   : {cur['regime']}",
                  f"  binding  : {cur['binding']}",
                  f"  TPS class: {cur['tier_name']} — {cur['material']}",
                  f"  {cur['note']}"]
    return "\n".join(lines)
