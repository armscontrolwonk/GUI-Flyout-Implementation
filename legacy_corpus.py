"""Every legacy file shape the loaders have ever accepted, as dicts.

This is the corpus behind `test_legacy_upgrade.py`: each entry exercises one
compatibility branch that older Thrusty files depend on.  It is data, not a
test, so the golden-output capture and the test can both import it.
"""

# A minimal modern booster, the base every legacy variant mutates.
MODERN_BOOSTER = dict(
    name="Base", mass_initial=5000.0, mass_propellant=3500.0, mass_final=1500.0,
    diameter_m=0.9, length_m=11.0, burn_time_s=70.0, isp_s=230.0,
    guidance="pitch_program", burnout_angle_deg=45.0,
)

MODERN_RO = dict(name="Obj", mass_kg=500.0, beta_kg_m2=8000.0, shape="cone",
                 diameter_m=0.6, length_m=1.8)


def _b(**over):
    d = dict(MODERN_BOOSTER)
    d.update(over)
    return d


def _r(**over):
    d = dict(MODERN_RO)
    d.update(over)
    return d


# name -> raw dict, as it would sit on disk in some past version of Thrusty.
LEGACY_BOOSTERS = {
    "modern_control": _b(),
    # guidance vocabulary
    "guidance_gravity_turn": _b(guidance="gravity_turn"),
    "guidance_loft": _b(guidance="loft", loft_angle_deg=38.0,
                        loft_angle_rate_deg_s=2.5),
    "guidance_loft_multistage": dict(_b(guidance="loft", loft_angle_deg=40.0,
                                        loft_angle_rate_deg_s=2.0),
                                     stage2=_b(name="S2", guidance="loft")),
    # burnout angle under its old key
    "loft_angle_deg_only": {k: v for k, v in _b(loft_angle_deg=33.0).items()
                            if k != "burnout_angle_deg"},
    # 'rv' era field names
    "rv_named_loadout": _b(num_rvs=3, rv_mass_kg=250.0, rv_separates=True,
                           payload_kg=750.0),
    # payload baked into the stage masses (Scud class)
    "payload_body_baked": _b(mass_initial=5897.0, mass_final=2198.0,
                             payload_kg=1000.0, ro_separates=False),
    "payload_stack_only": _b(mass_initial=6000.0, payload_kg=1000.0,
                             ro_separates=True),
    "payload_body_baked_rv_key": _b(mass_initial=5897.0, mass_final=2198.0,
                                    payload_kg=1000.0, rv_separates=False),
    # nose geometry given as a ratio
    "nose_ld_ratio": _b(nose_ld_ratio=3.0, shroud_nose_ld_ratio=2.0),
    # mass_final absent (derived)
    "no_mass_final": {k: v for k, v in _b().items() if k != "mass_final"},
    # embedded reentry object under both keys
    "embedded_ro": _b(ro=_r()),
    "embedded_rv": _b(rv=_r(glider_LD=1.8, glider_enabled=True)),
    # reentry hardware inline on the booster (the oldest form)
    "inline_rv_fields": _b(rv_beta_kg_m2=9000.0, rv_mass_kg=600.0,
                           rv_shape="cone", rv_diameter_m=0.55,
                           rv_length_m=1.6, rv_separates=True),
    "inline_ro_fields": _b(ro_beta_kg_m2=9000.0, ro_mass_kg=600.0,
                           ro_shape="cone", ro_diameter_m=0.55,
                           ro_length_m=1.6, glider_enabled=True,
                           glider_LD=2.2, glider_guidance="constant_bank",
                           glider_pullup_g_max=12.0,
                           glider_terminal_dive=True,
                           glider_terminal_alt_km=30.0),
    "inline_fields_no_beta": _b(ro_mass_kg=600.0, ro_shape="cone"),
    # a multi-stage legacy stack
    "two_stage_legacy": dict(_b(guidance="gravity_turn", num_rvs=1,
                                rv_mass_kg=400.0, payload_kg=400.0,
                                rv_separates=True),
                             stage2=_b(name="S2", mass_initial=1200.0,
                                       mass_propellant=900.0, mass_final=300.0)),
}

LEGACY_ROS = {
    "modern_control": _r(),
    # retired glide-law names
    "glide_constant_bank": _r(glider_guidance="constant_bank", glider_LD=2.0),
    "glide_azimuth_command": _r(glider_guidance="azimuth_command", glider_LD=2.0),
    "glide_skip_to_equilibrium": _r(glider_guidance="skip_to_equilibrium",
                                    glider_LD=2.0),
    "glide_absent": {k: v for k, v in _r().items() if k != "glider_guidance"},
    # separation tokens that predate the two-token vocabulary
    "sep_separating_rv": _r(separation_mode="separating_rv"),
    "sep_non_separating": _r(separation_mode="non_separating"),
    # body form vocabulary
    "body_form_unknown": _r(body_form="lifting_brick"),
    "body_form_absent": {k: v for k, v in _r().items() if k != "body_form"},
    # maneuvering capability declared implicitly (pre-split files)
    "cap_from_ld": _r(glider_LD=1.8),
    "cap_from_enabled": _r(glider_enabled=True),
    "cap_from_wing": _r(wing_area_m2=0.35),
    "cap_none": _r(),
    "cap_explicit_false": _r(glider_LD=1.8, maneuvering=False),
    # pre-split hardware/plan mixture: an object file carrying plan keys
    "carries_plan_keys": _r(glider_LD=2.0, glider_enabled=True,
                            glider_guidance="damped_glide",
                            glider_bank_schedule=[[0.0, 100.0, 15.0]],
                            glider_beta_entry_kg_m2=7.0,
                            glider_pullup_g_max=12.0,
                            separation_mode="body"),
}
