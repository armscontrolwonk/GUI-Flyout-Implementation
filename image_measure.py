"""Pure measurement logic for the image dimensioning tool (no Tk, no display).

Everything the risk register in IMAGE_DIMENSION_TOOL_DESIGN.md turns on lives
here, so it is unit-tested without a GUI: scale from an anchor, the pixel
quantum, the resolution floor (R4), per-measurement provenance, the clocking
correction (R1), anchor-free quantities (R2), the convention conversions (R5),
and the notes stamp (decision 1: no persistence, a text stamp is all that
survives).  The Tk dialog is a thin shell over this.

Governing rule (design doc): the image PROPOSES, the human COMMITS.  Nothing
here writes a field; it returns proposed values with their provenance and
flags, and the caller (a human clicking Accept) decides.
"""

import math

RESOLUTION_FLOOR_PX = 5.0        # R4: a feature smaller than this is refused
CLOCKING_X_ROLL_DEG = 45.0       # R1: worst-case ×-roll for a 4-fin set


def _dist(p1, p2):
    return math.hypot(float(p2[0]) - float(p1[0]), float(p2[1]) - float(p1[1]))


class Scale:
    """Metres-per-pixel from an anchor: two ORIGINAL-image pixel points a known
    real distance apart.  Pixel coordinates must be in the source image's own
    resolution (not display pixels) — the caller maps display→source first, so
    the quantum reflects the real image (R4)."""

    def __init__(self, p1, p2, real_distance_m, anchor_note=""):
        px = _dist(p1, p2)
        if px <= 0.0:
            raise ValueError("scale anchor points coincide (zero pixel span)")
        if float(real_distance_m) <= 0.0:
            raise ValueError("scale distance must be positive")
        self.anchor_pixels = px
        self.real_distance_m = float(real_distance_m)
        self.m_per_px = self.real_distance_m / px
        self.anchor_note = str(anchor_note or "")

    def pixel_quantum_m(self):
        """1 pixel in metres — the finest distinction the image can make."""
        return self.m_per_px

    def quantum_str(self):
        q = self.m_per_px
        return f"1 px = {q * 100:.2g} cm" if q < 1.0 else f"1 px = {q:.3g} m"

    def measure(self, p1, p2):
        """(metres, pixel_span) for a two-point measurement.  The floor is NOT
        applied here — Measurement records it as a flag so the caller can show
        or refuse it."""
        px = _dist(p1, p2)
        return px * self.m_per_px, px

    def below_floor(self, pixel_span):
        return float(pixel_span) < RESOLUTION_FLOOR_PX


# R1: the clocking choices offered at a clocking-sensitive prompt.  Ordered
# (label shown → key passed to Measurement); default is the first (in-plane, no
# correction) so the tool never silently inflates a span — the correction is
# OFFERED, never inferred.
CLOCKING_OPTIONS = (
    ("in-plane (fin edge-on, true span)", "in_plane"),
    ("×-rolled 45° (apply cos45 correction)", "x_rolled"),
    ("unknown (no correction; may under-read)", "unknown"),
)


def clocking_correction(measured_m, clocking):
    """R1: a span measured side-on under-reads if the feature is rolled out of
    the image plane.  ×-rolled (4-fin "×") divides by cos45°; 'in_plane' and
    'unknown' pass through (unknown is flagged elsewhere, never auto-corrected —
    the correction is OFFERED, not inferred)."""
    if clocking == "x_rolled":
        return float(measured_m) / math.cos(math.radians(CLOCKING_X_ROLL_DEG))
    return float(measured_m)


# Convention conversions (R5): the number CLICKED off a side view vs the number
# the data model STORES.  Each entry is (label shown in the prompt, click→store
# callable).  Keeps a silent factor-of-2 from living between the eye and the DB.
def _identity(x):
    return x


def _twice(x):
    return 2.0 * x


def _half(x):
    return 0.5 * x


CONVENTIONS = {
    "ro_diameter":  ("base diameter (tip-to-tip across the base)", _identity),
    "ro_length":    ("length (nose tip to base)", _identity),
    "ro_nose_r":    ("nose-tip RADIUS = half the clicked blunt-tip width", _half),
    "wedge_depth":  ("side-view base DEPTH (stored as ⌀ = the depth)", _identity),
    "half_cone_depth": ("side-view depth = ⌀/2 → stored ⌀ = 2×", _twice),
    "wedge_span":   ("plan-view span (tip to tip) — needs a PLAN view", _identity),
    "fore_cone_length": ("fore-cone length (nose tip to the cone break)", _identity),
    "break_diameter": ("diameter at the fore/aft cone break (across)", _identity),
    "wing_root":    ("ONE wing/fin panel's root chord", _identity),
    "wing_span":    ("ONE wing/fin panel's exposed span (root to tip)", _identity),
    # booster
    "stage_diameter": ("stage body diameter (across)", _identity),
    "stage_length":   ("stage length (top to bottom of the stage)", _identity),
    "fairing_diameter": ("fairing base diameter (across)", _identity),
    "fairing_length": ("fairing length (base to nose tip)", _identity),
    "fin_span":     ("ONE fin's exposed span (root to tip)", _identity),
    "fin_root":     ("ONE fin's root chord (leading to trailing edge at root)", _identity),
    "fin_tip":      ("ONE fin's tip chord", _identity),
    "strapon_diameter": ("ONE strap-on's diameter (across)", _identity),
    "strapon_length": ("ONE strap-on's length", _identity),
    "overall_length": ("overall length (tip to base) — cross-check only, "
                       "never stored", _identity),
}


# ── Length closure (booster) ────────────────────────────────────────────────
# The measured stage lengths (+ fairing) should tile the vehicle's overall
# length.  A mismatch is INFORMATION — a wrong claimed total (the AUR 10.2 m
# case), a mis-clicked invisible joint (R3), a real gap — so the tool WARNS
# and reports, never normalizes: auto-correcting the segments to force
# closure would launder the disagreement into the data.
OVERALL_LEN_CHECK_FIELD = "overall_len_check"    # never written to the editor
_SEGMENT_CONVENTIONS = ("stage_length", "fairing_length")


def closure_segments(prompts):
    """The prompt fields whose accepted values should tile the overall
    length (stage lengths + fairing length)."""
    return [p["field"] for p in prompts
            if p.get("convention") in _SEGMENT_CONVENTIONS]


def length_closure(accepted, prompts, total_m):
    """Compare sum(measured length segments) against the overall length.
    total_m comes from the check-only measurement or from a scale anchor the
    user declared to BE the overall length.  Returns None when no total is
    available (nothing to check against); otherwise a dict with the running
    sum, the missing segments, and — once every declared segment is
    measured — the signed error."""
    if not total_m or float(total_m) <= 0.0:
        return None
    segs = closure_segments(prompts)
    if not segs:
        return None
    missing = [f for f in segs if f not in (accepted or {})]
    s = sum(float(accepted[f]) for f in segs if f in (accepted or {}))
    out = dict(sum_m=s, total_m=float(total_m), missing=missing,
               complete=not missing)
    if not missing:
        out["delta_m"] = s - float(total_m)
        out["rel"] = (s - float(total_m)) / float(total_m)
    return out


def closure_note(c):
    """One-line human reading of a length_closure() result."""
    if c is None:
        return ""
    if not c["complete"]:
        return ("length closure pending — unmeasured: "
                + ", ".join(c["missing"]))
    return (f"length closure: segments sum {c['sum_m']:.4g} m vs total "
            f"{c['total_m']:.4g} m ({c['rel']:+.1%})")


# ── Cross-view consistency (Phase B multi-view) ─────────────────────────────
# The overall length appears in BOTH the side and plan views, so measuring it
# twice audits the two views' independent scale anchors for free: if they
# disagree, one anchor is wrong — and the span (which only the plan view can
# give) would inherit that error.  Check-only, like the closure total: the
# plan-view length is never stored.
PLAN_LEN_CHECK_FIELD = "plan_len_check"


def view_consistency(accepted):
    """Compare the stored (side-view) length with the plan-view check
    measurement.  None until both exist; then the signed disagreement,
    relative to the side view."""
    a = accepted or {}
    side = a.get("_len_var")
    plan = a.get(PLAN_LEN_CHECK_FIELD)
    if not side or not plan or float(side) <= 0.0:
        return None
    return dict(side_m=float(side), plan_m=float(plan),
                rel=(float(plan) - float(side)) / float(side))


def view_consistency_note(vc):
    if vc is None:
        return ""
    return (f"view cross-check: length side {vc['side_m']:.4g} m vs plan "
            f"{vc['plan_m']:.4g} m ({vc['rel']:+.1%})")


class Measurement:
    """One proposed field value with its full provenance and flags.  Immutable
    record; the caller reads .value_m and .flags to present Accept/Edit/Skip."""

    def __init__(self, field, click_m, pixel_span, scale, *, view="side",
                 convention=None, clocking="in_plane", flags=None):
        label, convert = (CONVENTIONS.get(convention, (None, _identity))
                          if convention else (None, _identity))
        v = clocking_correction(click_m, clocking)
        self.field = field
        self.view = view
        self.convention = convention
        self.convention_label = label
        self.clocking = clocking
        self.pixel_span = float(pixel_span)
        self.pixel_quantum_m = scale.pixel_quantum_m()
        self.value_m = convert(v)
        self.flags = list(flags or [])
        if scale.below_floor(pixel_span):
            self.flags.append(
                f"below the {RESOLUTION_FLOOR_PX:.0f} px resolution floor "
                f"({pixel_span:.1f} px) — refused")
            self.refused = True
        else:
            self.refused = False
        if clocking == "x_rolled":
            self.flags.append("×-roll cos45° correction applied")
        elif clocking == "unknown":
            self.flags.append("clocking unknown — no roll correction, span may under-read")

    def quantum_str(self):
        q = self.pixel_quantum_m
        return f"1 px = {q * 100:.2g} cm" if q < 1.0 else f"1 px = {q:.3g} m"


# ── Angle measurement (3 clicks: vertex + two rays) ─────────────────────────
# Angles are the premier anchor-free quantity (R2): they need NO scale and
# survive a wrong anchor completely.  Their failure mode is different —
# non-uniform stretch (a figure resized in one axis) and perspective tilt
# corrupt every angle silently — so every measurable angle ships WITH an
# independent cross-check: an identity twin derived from measured lengths
# (tan θ = (⌀/2)/L for the cone flank; tan Λ = (c_r − c_t)/s_e for sweep) and,
# on symmetric bodies, a two-flank symmetry check that doubles as a
# perspective/tilt detector.  Warn-only, like the length closure.

def angle_between_deg(vertex, p1, p2):
    """Angle at `vertex` between rays vertex→p1 and vertex→p2, in degrees
    (0–180).  Raises ValueError on a degenerate (zero-length) ray."""
    ax, ay = float(p1[0]) - float(vertex[0]), float(p1[1]) - float(vertex[1])
    bx, by = float(p2[0]) - float(vertex[0]), float(p2[1]) - float(vertex[1])
    na, nb = math.hypot(ax, ay), math.hypot(bx, by)
    if na <= 0.0 or nb <= 0.0:
        raise ValueError("angle ray has zero length")
    c = max(-1.0, min(1.0, (ax * bx + ay * by) / (na * nb)))
    return math.degrees(math.acos(c))


class AngleMeasurement:
    """One proposed ANGLE value (degrees).  Anchor-free: needs no scale and
    has no pixel quantum; the resolution guard is on RAY LENGTH instead — a
    short ray makes the angle noisy, so rays under the floor are refused.

    `complement=True` stores 90° − (the clicked angle): used for wing/fin LE
    sweep, which is measured between two REAL edges meeting at the root-LE
    corner — the LE and the ROOT CHORD — and stored as Λ (from spanwise).
    Measuring against a physical edge instead of an imagined 'spanwise' ray
    removes the swap that reads a slender delta's 77° sweep as its 13°
    complement.  `raw_deg` keeps the clicked value for a transparent read-out.
    """

    def __init__(self, field, vertex, p1, p2, complement=False):
        self.field = field
        self.hand_entered = False
        self.view = "angle"                    # not tied to a scale/view
        self.complement = bool(complement)
        self.flags = ["angle — anchor-free (no scale involved)"]
        rays = [math.hypot(float(p[0]) - float(vertex[0]),
                           float(p[1]) - float(vertex[1])) for p in (p1, p2)]
        if min(rays) < 2.0 * RESOLUTION_FLOOR_PX:
            self.refused = True
            self.raw_deg = self.value_deg = None
            self.flags.append(
                f"ray shorter than {2.0 * RESOLUTION_FLOOR_PX:.0f} px — angle "
                "too noisy, refused")
        else:
            self.refused = False
            self.raw_deg = angle_between_deg(vertex, p1, p2)
            self.value_deg = (90.0 - self.raw_deg if complement
                              else self.raw_deg)
            if complement:
                self.flags.append(
                    f"LE↔root {self.raw_deg:.1f}° → sweep Λ = "
                    f"{self.value_deg:.1f}° (90° − LE↔root)")


def sweep_from_planform(root_chord_m, span_exposed_m, tip_chord_m=0.0):
    """The identity twin of a measured LE sweep: with a straight trailing
    edge, tan Λ = (c_r − c_t)/s_e (delta wing: c_t = 0 → tan Λ = c_r/s_e).
    Anchor-free too — a ratio of two same-scale lengths."""
    s = float(span_exposed_m)
    if s <= 0.0:
        return None
    return math.degrees(math.atan2(float(root_chord_m) - float(tip_chord_m), s))


def cone_half_angle_from_lengths(diameter_m, length_m):
    """Identity twin of a measured cone flank angle: tan θ = (⌀/2)/L."""
    L = float(length_m)
    if L <= 0.0:
        return None
    return math.degrees(math.atan2(0.5 * float(diameter_m), L))


ANGLE_CHECK_REL = 0.05          # warn beyond ±5% — stretch/tilt/mis-click


def angle_check_note(measured_deg, derived_deg, what):
    """Warn-only comparator between a measured angle and its derived-from-
    lengths twin.  Disagreement on a clean orthographic image is impossible,
    so it specifically diagnoses non-uniform stretch, perspective tilt, or a
    mis-click — never auto-corrected.  One failure mode is decoded by name:
    measured + derived ≈ 90° means the angle was taken from the BODY AXIS
    instead of spanwise (on a nose-up image, clicking down the root is the
    natural wrong motion) — the note states the complement to use, but never
    writes it."""
    if derived_deg is None:
        return ""
    m, d = float(measured_deg), float(derived_deg)
    rel = (m - d) / d if d else 0.0
    if abs(rel) <= ANGLE_CHECK_REL:
        verdict = "agrees"
    elif abs((m + d) - 90.0) <= 3.0:
        verdict = (f"DISAGREES, and measured + derived ≈ 90°: the two "
                   f"reference edges were swapped (the second ray should run "
                   f"along the ROOT CHORD, not spanwise) — the complement "
                   f"{90.0 - m:.1f}° matches the planform (re-measure, or Type "
                   f"value it)")
    else:
        verdict = "DISAGREES — image stretch/tilt or a mis-click?"
    return (f"{what}: measured {m:.1f}° vs derived-from-lengths {d:.1f}° "
            f"({rel:+.1%}) — {verdict}")


def symmetry_note(upper_deg, lower_deg):
    """Two-flank symmetry check on an axisymmetric body: the flanks must be
    equal; asymmetry impeaches the IMAGE (tilt / perspective), lengths and
    all — the R9 screening upgraded to a measurement."""
    u, d = float(upper_deg), float(lower_deg)
    mean = 0.5 * (u + d)
    if mean <= 0.0:
        return ""
    rel = abs(u - d) / mean
    verdict = ("symmetric — no tilt detected" if rel <= ANGLE_CHECK_REL
               else "ASYMMETRIC — image tilted or perspective-distorted; "
                    "treat ALL measurements from this view with suspicion")
    return (f"flank symmetry: upper {u:.1f}° vs lower {d:.1f}° "
            f"({rel:.1%} apart) — {verdict}")


# Check-only flank fields (axisymmetric): never stored; they exist to audit
# the image and the ⌀/L pair.
FLANK_UPPER_FIELD = "flank_upper_deg"
FLANK_LOWER_FIELD = "flank_lower_deg"


def ro_angle_prompts(body_form="axisymmetric"):
    """Angle checklist for the RO editor.  The stored target is the wing LE
    sweep (degrees — load-bearing since the Phase-2b wing-body composite
    derives the exposed planform from it); the cone flank angles are
    CHECK-ONLY audits (symmetry + identity vs the ⌀/L pair).  Wing sweep is
    offered whenever the form can carry a wing, same rule as ro_prompts."""
    p = []
    if body_form != "wedge":
        p.append(dict(field="_wing_sweep_var", angle=True, unit="deg",
                      complement=True,
                      label="ANGLE: click the wing-root LE corner (vertex), "
                      "then a point along the LEADING EDGE, then a point along "
                      "the ROOT CHORD toward the tail — both real edges; the "
                      "tool stores sweep Λ = 90° − that opening (a slender "
                      "delta reads a small 10–20° opening → large Λ)",
                      view="side"))
    if body_form == "axisymmetric":
        p.append(dict(field=FLANK_UPPER_FIELD, angle=True, unit="deg",
                      label="ANGLE (check only): nose tip (vertex), then a "
                      "point along the UPPER flank, then a point along the "
                      "body axis — upper half-angle", view="side"))
        p.append(dict(field=FLANK_LOWER_FIELD, angle=True, unit="deg",
                      label="ANGLE (check only): nose tip (vertex), then a "
                      "point along the LOWER flank, then a point along the "
                      "body axis — lower half-angle", view="side"))
    return p


def booster_angle_prompts(has_fins=False):
    """Angle checklist for the booster editor: the fin LE sweep (degrees)."""
    if not has_fins:
        return []
    return [dict(field="fin_sweep", angle=True, unit="deg", complement=True,
                 label="ANGLE: click the fin-root LE corner (vertex), then a "
                 "point along the LEADING EDGE, then a point along the ROOT "
                 "CHORD toward the tail — both real edges; stores sweep Λ = "
                 "90° − that opening", view="side")]


class HandEntry:
    """A checklist value the user TYPED instead of clicked — for dimensions
    already known precisely (a published diameter, the scale-anchor length).
    Recorded as its own kind so the provenance stamp never claims it came off
    the image: no pixel quantum, no view, flagged entered-by-hand."""

    def __init__(self, field, value_m):
        self.field = field
        self.value_m = float(value_m)
        self.refused = False
        self.hand_entered = True
        self.flags = ["entered by hand — not measured from the image"]


# ── R8: the mixed-provenance delta view ─────────────────────────────────────
# Apply must NEVER silently overwrite: before anything is written, the user
# sees field → current editor value → proposed value → Δ%.  A large delta is
# a FINDING (a hand-entered number contradicted by the image, or vice versa),
# surfaced rather than silently resolved either way.
DELTA_WARN_REL = 0.05            # highlight |Δ| beyond ±5%


def apply_deltas(accepted, current):
    """Rows for the apply-preview table.  `accepted` is {field: proposed};
    `current` maps each WRITABLE field to its present editor value (None when
    blank/unparseable).  Fields absent from `current` — the check-only
    cross-checks (overall length, plan length, flank angles) — are excluded:
    they are never written.  Rows sort biggest |Δ| first (findings on top),
    fields with no prior value ("new") last."""
    rows = []
    for f, new in (accepted or {}).items():
        if f not in current:
            continue
        old = current[f]
        rel = None
        if old is not None and float(old) != 0.0:
            rel = (float(new) - float(old)) / float(old)
        rows.append(dict(field=f, old=(None if old is None else float(old)),
                         new=float(new), rel=rel))
    return sorted(rows, key=lambda r: (r["rel"] is None,
                                       -(abs(r["rel"]) if r["rel"] is not None
                                         else 0.0)))


# Anchor-free quantities (R2): what survives a WRONG scale anchor.  Absolute
# lengths inherit the anchor's error 1:1; ratios and angles cancel it.
ANCHOR_FREE = (
    "fineness ratio (length / diameter)",
    "cone / wedge half-angle",
    "taper and sweep angles",
    "any ratio of two lengths measured at the same scale",
)


def anchor_free_note():
    return ("Anchor-free (immune to a wrong scale): "
            + "; ".join(ANCHOR_FREE)
            + ".  Absolute lengths inherit the scale anchor's error 1:1.")


def provenance_stamp(measurements, scale, date_str, view_note=""):
    """The text stamp appended to the object's notes on Apply (decision 1: the
    audit trail without the bytes).  date_str is passed in for determinism.
    Hand-entered values are listed SEPARATELY — the stamp never claims a typed
    number was measured off the image."""
    accepted = [m for m in measurements if not getattr(m, "refused", False)]
    measured = sorted({m.field for m in accepted
                       if not getattr(m, "hand_entered", False)})
    entered = sorted({m.field for m in accepted
                      if getattr(m, "hand_entered", False)})
    bits = [f"[{date_str}] dimensional draft from image",
            ", ".join(measured) or "(none)"]
    if entered:
        bits.append("entered by hand (not measured): " + ", ".join(entered))
    views = sorted({getattr(m, "view", "side") for m in accepted
                    if not getattr(m, "hand_entered", False)} - {"angle"})
    if len(views) > 1:
        bits.append("views: " + "+".join(views))
    bits += [(scale.anchor_note or "scale set"), scale.quantum_str()]
    if view_note:
        bits.append(view_note)
    return " · ".join(bits)


# The prompt checklist (R5 conventions embedded).  Generated from the declared
# reentry-object topology; each prompt names the field var, the instruction,
# the view it needs, and the convention that converts click→stored value.
def ro_prompts(body_form="axisymmetric", biconic=False):
    """Ordered prompts for the reentry-object editor, generated from the
    declared topology: body form (shape dropdown) and biconic (its Shape
    entry — body-of-revolution only).  Wing prompts are ALWAYS offered for
    forms that can carry a wing (every form except the wedge, whose body IS
    the lifting surface): the wings are visible in the image whether or not
    Maneuvering is ticked yet, skip is structural, and APPLY enables the
    Maneuvering section when wing geometry was measured (the same
    measured-it-so-show-it rule as the booster's fairing/fin sections)."""
    p = []
    if body_form == "wedge":
        p.append(dict(field="_len_var", label="Click the two ends of the LENGTH "
                      "(nose tip to base)", view="side", convention="ro_length"))
        p.append(dict(field="_dia_var", label="Click the side-view BASE DEPTH "
                      "(top to bottom at the base)", view="side",
                      convention="wedge_depth"))
        p.append(dict(field="_body_span_var", label="Click the PLAN-view SPAN "
                      "(tip to tip) — requires a top/plan view", view="plan",
                      convention="wedge_span"))
        # Cross-view audit: length in the plan view too (check-only, never
        # stored) — if the two views' scales disagree on it, one anchor is
        # wrong and the span would inherit the error.
        p.append(dict(field=PLAN_LEN_CHECK_FIELD, label="Click the LENGTH in "
                      "the PLAN view — cross-checks the two views' scales; "
                      "not stored", view="plan", convention="ro_length"))
    elif body_form == "half_cone":
        p.append(dict(field="_len_var", label="Click the two ends of the LENGTH",
                      view="side", convention="ro_length"))
        p.append(dict(field="_dia_var", label="Click the side-view DEPTH "
                      "(the flat cut is at the axis; stored ⌀ = 2×)", view="side",
                      convention="half_cone_depth"))
    else:                                       # axisymmetric (cone / biconic)
        p.append(dict(field="_len_var", label="Click the two ends of the LENGTH",
                      view="side", convention="ro_length"))
        p.append(dict(field="_dia_var", label="Click across the BASE DIAMETER",
                      view="side", convention="ro_diameter"))
        p.append(dict(field="_nose_var", label="Click across the blunt NOSE TIP "
                      "(radius = half the tip width; often below the floor)",
                      view="side", convention="ro_nose_r"))
        if biconic:                             # declared: the biconic checkbox
            p.append(dict(field="_fore_len_var", label="Click the FORE-CONE "
                          "length (nose tip to the cone break)", view="side",
                          convention="fore_cone_length"))
            p.append(dict(field="_break_dia_var", label="Click across the BREAK "
                          "diameter (where the fore cone meets the aft frustum)",
                          view="side", convention="break_diameter"))
    if body_form != "wedge":
        # S and AR are DERIVED from this planform by the editor (single
        # source of truth = wing_geometry), so the tool measures the
        # planform, never the area.  LE sweep is an angle — the 3-click
        # angle prompt covers it (ro_angle_prompts).
        p.append(dict(field="_wing_root_var", label="Click ONE wing/fin panel's "
                      "ROOT chord (leading to trailing edge at the body)",
                      view="side", convention="wing_root"))
        # R1: an exposed span foreshortens when the panel set is ×-rolled.
        p.append(dict(field="_wing_span_var", label="Click ONE wing/fin panel's "
                      "exposed SPAN (root to tip)", view="side",
                      convention="wing_span", clocking_sensitive=True))
    return p


def booster_prompts(n_stages=1, has_fairing=False, has_fins=False,
                    n_fins=0, n_strapons=0):
    """Ordered prompts for the booster editor, generated from the topology the
    editor ALREADY declares (stage count, fairing on/off, fins on/off + count,
    strap-on count).  Repeated features (fins, strap-ons) are measured ONCE and
    the model replicates them to the declared count — so a prompt asks for ONE
    instance and the label states the count assumption (R1/design 'measure one,
    declare count')."""
    p = []
    for i in range(1, max(1, int(n_stages)) + 1):
        p.append(dict(field=f"stage{i}_len",
                      label=f"Click STAGE {i} length (top to bottom)",
                      view="side", convention="stage_length"))
        p.append(dict(field=f"stage{i}_dia",
                      label=f"Click STAGE {i} diameter (across)",
                      view="side", convention="stage_diameter"))
    if has_fairing:
        p.append(dict(field="fairing_len", label="Click FAIRING length "
                      "(base to nose tip)", view="side", convention="fairing_length"))
        p.append(dict(field="fairing_dia", label="Click FAIRING base diameter",
                      view="side", convention="fairing_diameter"))
    if has_fins:
        note = (f" — measure ONE; the model replicates to the {int(n_fins)} "
                "declared fins, assumed identical") if n_fins else ""
        # R1: span is the dimension a ×-rolled fin set foreshortens side-on;
        # the prompt flags itself so the dialog offers the cos correction.
        p.append(dict(field="fin_span", label="Click ONE fin's exposed SPAN"
                      + note, view="side", convention="fin_span",
                      clocking_sensitive=True))
        p.append(dict(field="fin_root", label="Click ONE fin's ROOT chord",
                      view="side", convention="fin_root"))
        p.append(dict(field="fin_tip", label="Click ONE fin's TIP chord",
                      view="side", convention="fin_tip"))
    if int(n_strapons) > 0:
        note = (f" — measure ONE; replicated to the {int(n_strapons)} declared "
                "strap-ons, assumed identical")
        p.append(dict(field="strapon_dia", label="Click ONE strap-on's DIAMETER"
                      + note, view="side", convention="strapon_diameter"))
        p.append(dict(field="strapon_len", label="Click ONE strap-on's LENGTH",
                      view="side", convention="strapon_length"))
    # Closure cross-check: overall length is never stored (no such editor
    # field — the model derives it from the parts), but measuring it lets the
    # dialog check that the stage (+ fairing) lengths tile it.  Warn-only.
    p.append(dict(field=OVERALL_LEN_CHECK_FIELD,
                  label="Click the OVERALL length (tip to base) — cross-check "
                  "only, not stored", view="side", convention="overall_length"))
    return p
