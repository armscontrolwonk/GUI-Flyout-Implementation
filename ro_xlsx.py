"""
ro_xlsx.py — XLSX import/export for Thrusty reentry objects (ROParams).

The reentry-object counterpart to missile_xlsx.py: edit a reentry object in a
familiar spreadsheet grid instead of hand-editing ro.json.  Reuses that
module's low-level cell writers/readers so the two stay visually consistent.

Sheet layout
------------
  Sheet 1 "RO"        — every ROParams field, fields-as-rows, values in col D
  Sheet 2 "Reference" — read-only TPS material catalog + emissivity guidance

Public API
----------
  export_ro_xlsx(path, rv)      -> None
  import_ro_xlsx(path)          -> ROParams
  make_blank_ro_template(path)  -> None

Google Sheets: no dedicated integration is needed — in Google Sheets use
File > Download > Microsoft Excel (.xlsx), then Load Reentry Object from XLSX.  A native
Sheets read/write is a possible future enhancement.
"""

from __future__ import annotations

from missile_xlsx import (
    _xl, _section, _label, _inputs, _dropdown, _yn,
    _rnum, _rint, _rstr, _rbool,
    _NOSE_OPTS, _NOSE_LABEL, _NOSE_KEY,
)

# ---------------------------------------------------------------------------
# Row registry (1-based).  Writer and reader share it so the mapping never
# drifts.  All values live in column D (4); labels/units/notes in B/C/I.
# ---------------------------------------------------------------------------
_R: dict[str, int] = {
    'name':        4,
    # geometry & mass
    'mass':        7,
    'beta':        8,
    'shape':       9,
    'diam':       10,
    'length':     11,
    'nose_rn':    12,
    'sep':        13,
    # maneuvering / glider
    'g_on':       16,
    'g_ld':       17,
    'g_gmax':     18,
    'g_betaS':    19,
    'g_guid':     20,
    'g_zeta':     21,
    'g_skip':     22,
    'g_aero':     23,
    'g_tdive':    24,
    'g_talt':     25,
    # TPS
    'emiss':      28,
    'nose_mat':   29,
    'body_mat':   30,
    'body_thk':   31,
    'struct_mat': 32,
    'struct_lim': 33,
    # custom nose material
    'cn_label':   36,
    'cn_abl':     37,
    'cn_lim':     38,
    'cn_dens':    39,
    'cn_heff':    40,
    # custom body material
    'cb_label':   43,
    'cb_abl':     44,
    'cb_lim':     45,
    'cb_dens':    46,
    'cb_heff':    47,
    # provenance
    'source':     50,
    'notes':      51,
}

_VAL_COL = 4   # column D

_SEP_OPTS   = ['separating_rv', 'body']
_YESNO_OPTS = ['YES', 'NO']
_CUSTOM     = 'custom'          # material-cell sentinel for a bespoke material


def _material_opts():
    """Dropdown options for a material cell: blank, every catalog key, custom."""
    import heating
    return [''] + list(heating.TPS_MATERIALS.keys()) + [_CUSTOM]


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------
def _build_ro_sheet(ws, rv) -> None:
    ws.column_dimensions['A'].width = 2
    ws.column_dimensions['B'].width = 30
    ws.column_dimensions['C'].width = 10
    ws.column_dimensions['D'].width = 26
    ws.column_dimensions['I'].width = 40

    ws.merge_cells('B1:D1')
    t = ws.cell(row=1, column=2, value='Thrusty — Reentry Object')
    from openpyxl.styles import Font
    t.font = Font(bold=True, size=13)

    mat_opts = _material_opts()

    def put(rk, value, fmt='General'):
        _inputs(ws, _R[rk], [_VAL_COL], [value], fmt=fmt)

    # Identity
    _section(ws, 3, 'Identity')
    _label(ws, _R['name'], 'Name')
    put('name', rv.name)

    # Geometry & mass
    _section(ws, 6, 'Geometry & mass')
    _label(ws, _R['mass'],   'Mass', 'kg')
    _label(ws, _R['beta'],   'Ballistic coeff β', 'kg/m²',
           'm/(Cd·A); scales with mass at fixed Cd·A')
    _label(ws, _R['shape'],  'Nose shape')
    _label(ws, _R['diam'],   'Base diameter', 'm')
    _label(ws, _R['length'], 'Length', 'm')
    _label(ws, _R['nose_rn'],'Nose-tip radius', 'm')
    _label(ws, _R['sep'],    'Separation mode', '', 'separating_rv or body')
    put('mass', rv.mass_kg); put('beta', rv.beta_kg_m2)
    put('diam', rv.diameter_m); put('length', rv.length_m)
    put('nose_rn', rv.nose_radius_m)
    _dropdown(ws, _R['shape'], _VAL_COL, _NOSE_OPTS)
    ws.cell(row=_R['shape'], column=_VAL_COL, value=_NOSE_LABEL.get(rv.shape, 'Cone'))
    _dropdown(ws, _R['sep'], _VAL_COL, _SEP_OPTS)
    ws.cell(row=_R['sep'], column=_VAL_COL, value=(rv.separation_mode or 'separating_rv'))

    # Maneuvering / glider
    _section(ws, 15, 'Maneuvering (glider / HGV)')
    _label(ws, _R['g_on'],    'Glider enabled')
    _label(ws, _R['g_ld'],    'Lift/drag L/D')
    _label(ws, _R['g_gmax'],  'Pull-up g-limit', 'g')
    _label(ws, _R['g_betaS'], 'Re-entry βₛ', 'kg/m²', '0 = Tracy')
    _label(ws, _R['g_guid'],  'Guidance mode', '', 'e.g. damped_glide, skip_glide')
    _label(ws, _R['g_zeta'],  'Damping ratio ζ', '', 'damped_glide only')
    _label(ws, _R['g_skip'],  'Skip count')
    _label(ws, _R['g_aero'],  'Aero model', '', 'e.g. polar')
    _label(ws, _R['g_tdive'], 'Terminal dive')
    _label(ws, _R['g_talt'],  'Terminal alt', 'km')
    _dropdown(ws, _R['g_on'], _VAL_COL, _YESNO_OPTS)
    ws.cell(row=_R['g_on'], column=_VAL_COL, value=_yn(rv.glider_enabled))
    put('g_ld', rv.glider_LD); put('g_gmax', rv.glider_pullup_g_max)
    put('g_betaS', rv.glider_beta_entry_kg_m2)
    put('g_guid', rv.glider_guidance); put('g_zeta', rv.glider_damping_zeta)
    put('g_skip', rv.glider_skip_count)
    put('g_aero', getattr(rv, 'glider_aero_model', 'polar'))
    _dropdown(ws, _R['g_tdive'], _VAL_COL, _YESNO_OPTS)
    ws.cell(row=_R['g_tdive'], column=_VAL_COL, value=_yn(rv.glider_terminal_dive))
    put('g_talt', rv.glider_terminal_alt_km)

    # TPS materials
    _section(ws, 27, 'Thermal protection (TPS)')
    _label(ws, _R['emiss'],     'Emissivity', '', '0.85 typical (0.75–0.90)')
    _label(ws, _R['nose_mat'],  'Nose material')
    _label(ws, _R['body_mat'],  'Body material')
    _label(ws, _R['body_thk'],  'Body layer thickness', 'm', '0 = auto')
    _label(ws, _R['struct_mat'],'Structure material')
    _label(ws, _R['struct_lim'],'Structure limit', 'K')
    put('emiss', rv.emissivity); put('body_thk', rv.body_tps_thickness_m)
    put('struct_lim', rv.structure_limit_K)
    import heating
    for rk, key in (('nose_mat', rv.nose_tps_material),
                    ('body_mat', rv.body_tps_material),
                    ('struct_mat', rv.structure_material)):
        _dropdown(ws, _R[rk], _VAL_COL, mat_opts)
        cell_val = key
        if key == heating.CUSTOM_NOSE_KEY or key == heating.CUSTOM_BODY_KEY:
            cell_val = _CUSTOM
        ws.cell(row=_R[rk], column=_VAL_COL, value=cell_val)

    # Custom material sub-blocks
    def _put_custom(prefix, cust):
        cust = cust or {}
        _inputs(ws, _R[prefix + '_label'], [_VAL_COL], [cust.get('label', '')])
        _dropdown(ws, _R[prefix + '_abl'], _VAL_COL, _YESNO_OPTS)
        ws.cell(row=_R[prefix + '_abl'], column=_VAL_COL,
                value=_yn(bool(cust.get('is_ablator', False))))
        _inputs(ws, _R[prefix + '_lim'],  [_VAL_COL],
                [cust.get('continuous_K') or cust.get('peak_K') or ''])
        _inputs(ws, _R[prefix + '_dens'], [_VAL_COL], [cust.get('density_kg_m3', '')])
        _inputs(ws, _R[prefix + '_heff'], [_VAL_COL], [cust.get('H_eff_MJ_kg', '')])

    _section(ws, 35, 'Custom nose material  (only if Nose material = custom)')
    for rk, lb, un in (('cn_label', 'Name', ''), ('cn_abl', 'Ablator', ''),
                       ('cn_lim', 'Temp. limit', 'K'), ('cn_dens', 'Density', 'kg/m³'),
                       ('cn_heff', 'Heat of ablation', 'MJ/kg')):
        _label(ws, _R[rk], lb, un)
    _put_custom('cn', rv.nose_tps_custom)

    _section(ws, 42, 'Custom body material  (only if Body material = custom)')
    for rk, lb, un in (('cb_label', 'Name', ''), ('cb_abl', 'Ablator', ''),
                       ('cb_lim', 'Temp. limit', 'K'), ('cb_dens', 'Density', 'kg/m³'),
                       ('cb_heff', 'Heat of ablation', 'MJ/kg')):
        _label(ws, _R[rk], lb, un)
    _put_custom('cb', rv.body_tps_custom)

    # Provenance
    _section(ws, 49, 'Provenance')
    _label(ws, _R['source'], 'Source', '', 'short citation')
    _label(ws, _R['notes'],  'Notes', '', 'free-form; confidence, assumptions')
    _inputs(ws, _R['source'], [_VAL_COL], [rv.source])
    _inputs(ws, _R['notes'],  [_VAL_COL], [rv.notes])


def _build_ro_reference_sheet(ws) -> None:
    import heating
    ws.column_dimensions['A'].width = 22
    ws.column_dimensions['B'].width = 30
    ws.column_dimensions['C'].width = 10
    ws.column_dimensions['D'].width = 12
    ws.column_dimensions['E'].width = 10
    hdr = ['key', 'label', 'group', 'limit K', 'ablator?']
    for j, h in enumerate(hdr, start=1):
        from openpyxl.styles import Font
        c = ws.cell(row=1, column=j, value=h); c.font = Font(bold=True)
    r = 2
    for key, m in heating.TPS_MATERIALS.items():
        ws.cell(row=r, column=1, value=key)
        ws.cell(row=r, column=2, value=m.get('label', ''))
        ws.cell(row=r, column=3, value=m.get('group', ''))
        ws.cell(row=r, column=4, value=m.get('continuous_K') or m.get('peak_K'))
        ws.cell(row=r, column=5, value='yes' if m.get('is_ablator') else 'no')
        r += 1
    ws.cell(row=r + 1, column=1, value='Emissivity typical: 0.85  (range 0.75–0.90)')


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------
def _read_material(ws, rk, prefix, sentinel):
    """Return (material_key, custom_dict_or_None) for a material cell."""
    val = _rstr(ws, _R[rk], _VAL_COL)
    if val.lower() == _CUSTOM:
        lim = _rnum(ws, _R[prefix + '_lim'], _VAL_COL, 0.0)
        label = _rstr(ws, _R[prefix + '_label'], _VAL_COL, 'Custom material')
        if lim <= 0 and not label:
            return '', None
        cust = {
            'label': label or 'Custom material',
            'is_ablator': _rbool(ws, _R[prefix + '_abl'], _VAL_COL),
            'continuous_K': lim, 'peak_K': lim,
            'density_kg_m3': _rnum(ws, _R[prefix + '_dens'], _VAL_COL, 0.0) or None,
            'H_eff_MJ_kg': _rnum(ws, _R[prefix + '_heff'], _VAL_COL, 0.0) or None,
        }
        return sentinel, cust
    return val, None


def import_ro_xlsx(path: str):
    """Read an RV XLSX and return an ROParams."""
    import heating
    from missile_models import ROParams
    xl = _xl()
    wb = xl.load_workbook(path, data_only=True)
    # New workbooks use sheet "RO"; accept the legacy "RV" name too.
    ws = wb['RO'] if 'RO' in wb.sheetnames else wb['RV']

    nose_key, nose_cust = _read_material(ws, 'nose_mat', 'cn', heating.CUSTOM_NOSE_KEY)
    body_key, body_cust = _read_material(ws, 'body_mat', 'cb', heating.CUSTOM_BODY_KEY)
    struct_key = _rstr(ws, _R['struct_mat'], _VAL_COL)
    if struct_key.lower() == _CUSTOM:
        struct_key = ''      # custom not supported for the structure slot

    return ROParams(
        name=_rstr(ws, _R['name'], _VAL_COL, 'Unnamed'),
        mass_kg=_rnum(ws, _R['mass'], _VAL_COL),
        beta_kg_m2=_rnum(ws, _R['beta'], _VAL_COL),
        shape=_NOSE_KEY.get(_rstr(ws, _R['shape'], _VAL_COL), 'cone') or 'cone',
        diameter_m=_rnum(ws, _R['diam'], _VAL_COL),
        length_m=_rnum(ws, _R['length'], _VAL_COL),
        nose_radius_m=_rnum(ws, _R['nose_rn'], _VAL_COL),
        separation_mode=_rstr(ws, _R['sep'], _VAL_COL, 'separating_rv') or 'separating_rv',
        glider_enabled=_rbool(ws, _R['g_on'], _VAL_COL),
        glider_LD=_rnum(ws, _R['g_ld'], _VAL_COL),
        glider_pullup_g_max=_rnum(ws, _R['g_gmax'], _VAL_COL, 10.0),
        glider_beta_entry_kg_m2=_rnum(ws, _R['g_betaS'], _VAL_COL),
        glider_guidance=_rstr(ws, _R['g_guid'], _VAL_COL, 'equilibrium_glide') or 'equilibrium_glide',
        glider_damping_zeta=_rnum(ws, _R['g_zeta'], _VAL_COL, 0.7),
        glider_skip_count=_rint(ws, _R['g_skip'], _VAL_COL, 1),
        glider_aero_model=_rstr(ws, _R['g_aero'], _VAL_COL, 'polar') or 'polar',
        glider_terminal_dive=_rbool(ws, _R['g_tdive'], _VAL_COL),
        glider_terminal_alt_km=_rnum(ws, _R['g_talt'], _VAL_COL, 30.0),
        emissivity=_rnum(ws, _R['emiss'], _VAL_COL, 0.85),
        nose_tps_material=nose_key,
        body_tps_material=body_key,
        body_tps_thickness_m=_rnum(ws, _R['body_thk'], _VAL_COL),
        structure_material=struct_key,
        structure_limit_K=_rnum(ws, _R['struct_lim'], _VAL_COL),
        nose_tps_custom=nose_cust,
        body_tps_custom=body_cust,
        source=_rstr(ws, _R['source'], _VAL_COL),
        notes=_rstr(ws, _R['notes'], _VAL_COL),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def export_ro_xlsx(path: str, rv) -> None:
    """Write an ROParams to a fully filled-in XLSX at path."""
    xl = _xl()
    wb = xl.Workbook()
    ws = wb.active
    ws.title = 'RO'
    _build_ro_sheet(ws, rv)
    _build_ro_reference_sheet(wb.create_sheet('Reference'))
    wb.save(path)


def make_blank_ro_template(path: str) -> None:
    """Write a blank RV template for the user to fill in from scratch."""
    from missile_models import ROParams
    export_ro_xlsx(path, ROParams(name='', mass_kg=0, beta_kg_m2=0))
