"""Deterministic geometry/bookkeeping tests for src/main.py.

These never load the FAIRChem model — they exercise only the pure ASE geometry
and energy-reference helpers, so they run fast on CPU. Run with the venv python:

    ./.env.local/Scripts/python.exe tests/test_geometry.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
import main  # noqa: E402


BOND = 1.6  # Ang, generous C-O / O-H bonding cutoff


def _neighbours_within(mol, i, cutoff=BOND):
    d = np.linalg.norm(mol.positions - mol.positions[i], axis=1)
    return [j for j in range(len(mol)) if j != i and d[j] < cutoff]


def test_glycolate_is_glycol_minus_hydroxyl_H():
    glyc = main.buildGlycol()
    glate = main.buildGlycolate()

    # one fewer atom, and it is an H (composition C2O2H5)
    assert len(glate) == len(glyc) - 1, len(glate)
    syms = glate.get_chemical_symbols()
    assert syms.count('C') == 2 and syms.count('O') == 2 and syms.count('H') == 5, syms

    # the removed atom is the O1 hydroxyl H: anchor O1 (index 2) now has exactly
    # one bonded neighbour (C1), down from two (C1 + hydroxyl H) in glycol.
    assert _neighbours_within(glyc, 2) == [0, 8], _neighbours_within(glyc, 2)
    assert _neighbours_within(glate, 2) == [0], _neighbours_within(glate, 2)


def test_orient_single_neighbour_points_bond_up():
    # The alkoxide case: orienting with one neighbour must align the
    # anchor->neighbour bond to +z (so the O sits lowest, lone pairs down).
    oriented = main.orientAnchorDown(main.buildGlycolate(), anchor=2, neighbours=(0,))
    v = oriented.positions[0] - oriented.positions[2]
    v = v / np.linalg.norm(v)
    assert np.allclose(v, [0, 0, 1], atol=1e-6), v


def test_adatom_facet_adds_one_undercoordinated_zn():
    flat, _, flatTop, _ = main.buildFacet('Zn(002)')
    slab, fixedIdx, slabTop, sites = main.buildFacet('Zn(002)-adatom')

    assert len(slab) == len(flat) + 1, (len(slab), len(flat))

    # adatom is the unique, highest atom and is free to relax
    z = slab.positions[:, 2]
    assert int(z.argmax()) == len(slab) - 1, z.argmax()
    assert (z == z.max()).sum() == 1
    assert z.max() > flatTop
    assert abs(slabTop - z.max()) < 1e-9
    assert (len(slab) - 1) not in set(fixedIdx.tolist())

    assert [s[0] for s in sites] == ['adatom'], sites


def test_vacancy_facet_removes_one_top_layer_atom():
    flat, _, _, _ = main.buildFacet('Zn(002)')
    slab, _, _, sites = main.buildFacet('Zn(002)-vacancy')

    assert len(slab) == len(flat) - 1, (len(slab), len(flat))

    def n_top(a):
        zr = np.round(a.positions[:, 2], 1)
        return int((zr == zr.max()).sum())

    assert n_top(slab) == n_top(flat) - 1, (n_top(slab), n_top(flat))
    assert [s[0] for s in sites] == ['vacancy'], sites


def test_glycolate_reference_is_glycol_minus_half_H2():
    eGas = {'water': -14.0, 'glycol': -52.0}
    eH2 = -6.0

    # glycolate: dissociative reference E_gas(glycol) - 1/2 E(H2)
    glate_ref = main.refEnergy(main.ADSORBATES['glycolate'], 'glycolate', eGas, eH2)
    assert abs(glate_ref - (-52.0 - 0.5 * -6.0)) < 1e-12, glate_ref

    # water/glycol: their own gas energy
    assert main.refEnergy(main.ADSORBATES['water'], 'water', eGas, eH2) == -14.0
    assert main.refEnergy(main.ADSORBATES['glycol'], 'glycol', eGas, eH2) == -52.0


def test_place_adsorbate_anchor_at_height_on_defect_facet():
    slab, fixedIdx, slabTop, sites = main.buildFacet('Zn(002)-adatom')
    spec = main.ADSORBATES['glycolate']
    system, nSlab = main.placeAdsorbate(slab, slabTop, fixedIdx, spec, sites[0][1], height=2.0)

    anchor = nSlab + spec['anchor']
    assert abs(system.positions[anchor, 2] - (slabTop + 2.0)) < 1e-9, system.positions[anchor, 2]
    assert nSlab == len(slab)


def run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    failures = 0
    for t in tests:
        try:
            t()
            print(f'PASS  {t.__name__}')
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f'FAIL  {t.__name__}: {type(exc).__name__}: {exc}')
    print(f'\n{len(tests) - failures}/{len(tests)} passed')
    return failures


if __name__ == '__main__':
    sys.exit(1 if run() else 0)
