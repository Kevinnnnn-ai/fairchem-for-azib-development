from pathlib import Path
import numpy as np
from ase import Atoms
from ase.build import hcp0001
from ase.constraints import FixAtoms
from ase.io import write
from ase.optimize import LBFGS
from fairchem.core import pretrained_mlip, FAIRChemCalculator

# Zn(100) prismatic facet is optional — only present in newer ASE.
try:
    from ase.build import hcp10m10
except ImportError:
    hcp10m10 = None


# ---------------------------------------------------------------------------
# Competitive adsorption: ethylene glycol (PEG monomer) vs H2O on Zn facets.
#
# A single molecule's adsorption energy in vacuum is not meaningful on its own:
# at the real interface every surface site is already taken by water. The
# question that maps to PEG's accepted "leveling agent" mechanism is whether the
# additive can OUT-BIND water for a site. So the key quantity here is
#     dE = E_ads(glycol) - E_ads(water)        (on the same facet)
# dE < 0  ->  glycol binds more strongly than water and can displace it
# dE > 0  ->  water wins; glycol cannot displace interfacial water on a flat
#             terrace, so its action must lie elsewhere (EDL, potential, steps,
#             charged/deprotonated species).
# Because dE is a difference of adsorption energies computed with the SAME
# calculator and references, much of the model's systematic error cancels — so
# the comparison is far more trustworthy than any single absolute number.
# ---------------------------------------------------------------------------


def getRunDir(prefix='compete-glycol-water'):
    runDirRoot = Path(__file__).resolve().parent.parent / 'stdout' / 'runs'
    runDirRoot.mkdir(parents=True, exist_ok=True)
    i = 1
    while (runDirRoot / f'{prefix}-{i}').exists():
        i += 1

    runDir = runDirRoot / f'{prefix}-{i}'
    runDir.mkdir()
    return runDir


def buildGlycol():
    # Ethylene glycol, HO-CH2-CH2-OH (C2H6O2), flat starting geometry.
    # Binding oxygen is O1 (index 2); its bonded neighbours are C1 (0) and the
    # hydroxyl H (8), used to orient it O-down.
    return Atoms(
        'C2O2H6',
        positions=[
            (0.000, 0.000, 0.000),    # 0 C1
            (1.520, 0.000, 0.000),    # 1 C2
            (-0.473, 1.339, 0.000),   # 2 O1 (on C1)  <- binding O
            (1.993, -1.339, 0.000),   # 3 O2 (on C2)
            (-0.363, -0.514, 0.890),  # 4 H on C1
            (-0.363, -0.514, -0.890), # 5 H on C1
            (1.883, 0.514, 0.890),    # 6 H on C2
            (1.883, 0.514, -0.890),   # 7 H on C2
            (0.289, 1.923, 0.000),    # 8 H on O1 (hydroxyl)
            (2.953, -1.315, 0.000),   # 9 H on O2 (hydroxyl)
        ],
    )


def buildWater():
    # H2O, O-down reference adsorbate. O is index 0 (the binding atom); its two
    # H's (indices 1, 2) are used to orient it O-down. O-H 0.96 Ang, angle 104.5.
    return Atoms(
        'OH2',
        positions=[
            (0.0000, 0.0000, 0.0),   # 0 O  <- binding atom
            (0.7575, 0.5865, 0.0),   # 1 H
            (-0.7575, 0.5865, 0.0),  # 2 H
        ],
    )


# Each adsorbate: how to build it, its binding (anchor) atom, and the atoms
# bonded to that anchor (used to point its lone pairs at the surface).
ADSORBATES = {
    'water':  {'build': buildWater,  'anchor': 0, 'neighbours': (1, 2)},
    'glycol': {'build': buildGlycol, 'anchor': 2, 'neighbours': (0, 8)},
}


def orientAnchorDown(mol, anchor, neighbours):
    # Rotate the molecule so the bisector of the anchor's bonds points to +z.
    # For an sp3 O that puts the lone pairs roughly -z (down, at the slab) with
    # the rest of the molecule above the O — the geometry an O-donor adopts when
    # coordinating a surface metal atom.
    m = mol.copy()
    aPos = m.positions[anchor]
    bis = np.zeros(3)
    for nb in neighbours:
        v = m.positions[nb] - aPos
        bis += v / np.linalg.norm(v)
    bis /= np.linalg.norm(bis)
    m.rotate(bis, (0, 0, 1), center=aPos)
    return m


def buildFacet(name):
    # Return (slab, fixedIdx, slabTop, sites). Fix the bottom half of the atomic
    # layers (hold the bulk) and relax the top half + adsorbate.
    if name == 'Zn(002)':
        slab = hcp0001('Zn', size=(3, 3, 4), vacuum=8.0, periodic=True)
    elif name == 'Zn(100)':
        if hcp10m10 is None:
            raise RuntimeError('ase.build.hcp10m10 unavailable in this ASE')
        # Prismatic (10-10) surface; hcp10m10 needs an even 2nd size index and
        # corrugates into 8 z-levels (4 layers x 2 sub-rows). (3,4,4) gives a
        # ~8x10 Ang cell (enough lateral vacuum for glycol) and 6 top-layer atoms.
        slab = hcp10m10('Zn', size=(3, 4, 4), vacuum=8.0, periodic=True)
    else:
        raise ValueError(f'unknown facet {name}')

    zRound = np.round(slab.positions[:, 2], 1)
    levels = np.unique(zRound)
    nFix = len(levels) // 2
    fixedIdx = np.where(np.isin(zRound, levels[:nFix]))[0]
    slabTop = slab.positions[:, 2].max()

    # atop / bridge / hollow built from the top-layer atom nearest the cell
    # centre, so sites stay inside the cell away from periodic-image edges.
    top = slab.positions[zRound == levels[-1]]
    centerXY = slab.cell[:2, :2].sum(axis=0) / 2.0
    A = top[np.argmin(np.linalg.norm(top[:, :2] - centerXY, axis=1))]
    nn = top[np.argsort(np.linalg.norm(top[:, :2] - A[:2], axis=1))]
    sites = [('atop', A[:2])]
    if len(nn) >= 2:
        sites.append(('bridge', (A[:2] + nn[1][:2]) / 2.0))
    if len(nn) >= 3:
        sites.append(('hollow', (A[:2] + nn[1][:2] + nn[2][:2]) / 3.0))
    return slab, fixedIdx, slabTop, sites


def placeAdsorbate(slab, slabTop, fixedIdx, spec, siteXY, height=2.0):
    # Build an O-down adsorbate with its anchor O placed `height` Ang directly
    # above the site, return the combined constrained system and slab size.
    ads = orientAnchorDown(spec['build'](), spec['anchor'], spec['neighbours'])
    a = spec['anchor']
    ads.positions[:, :2] += siteXY - ads.positions[a, :2]
    ads.positions[:, 2] += (slabTop + height) - ads.positions[a, 2]
    system = slab.copy()
    system += ads
    system.set_constraint(FixAtoms(indices=fixedIdx))
    return system, len(slab)


def relax(atoms, logPath, trajPath=None):
    atoms.calc = calc
    opt = LBFGS(
        atoms,
        logfile=str(logPath),
        trajectory=(str(trajPath) if trajPath else None),
    )
    opt.run(fmax=0.05, steps=300)
    return atoms


def slug(s):
    return s.replace('(', '').replace(')', '')


runDir = getRunDir()
print(f'Saving run output to {runDir}')

FACETS = ['Zn(002)', 'Zn(100)']

# ---------------------------------------------------------------------------
# Geometry first (no calculator) so any geometry bug fails fast. Build every
# facet and write the initial adsorbate placements for inspection.
# ---------------------------------------------------------------------------
facetData = {}
for fname in FACETS:
    try:
        slab, fixedIdx, slabTop, sites = buildFacet(fname)
    except Exception as exc:                       # noqa: BLE001 - skip unbuildable facet
        print(f'  [skip {fname}] {exc}')
        continue
    facetData[fname] = {'slab': slab, 'fixedIdx': fixedIdx,
                        'slabTop': slabTop, 'sites': sites}
    print(
        f'{fname}: {len(slab)} Zn, {len(np.unique(np.round(slab.positions[:,2],1)))} '
        f'layers; fixing {len(fixedIdx)}, relaxing {len(slab)-len(fixedIdx)} + adsorbate; '
        f'sites={[s[0] for s in sites]}'
    )
    for aname, spec in ADSORBATES.items():
        system, nSlab = placeAdsorbate(slab, slabTop, fixedIdx, spec, sites[0][1])
        write(runDir / f'init_{slug(fname)}_{aname}.xyz', system)
        low = system.get_chemical_symbols()[nSlab + int(system.positions[nSlab:, 2].argmin())]
        print(f'    init {aname:6s}: lowest atom {low} at gap '
              f'{system.positions[nSlab:,2].min()-slabTop:.2f} Ang')

if not facetData:
    raise RuntimeError('no facets could be built')

# ---------------------------------------------------------------------------
# Load the model.
# ---------------------------------------------------------------------------
predictor = pretrained_mlip.get_predict_unit('uma-s-1p2', device='cuda')
calc = FAIRChemCalculator(predictor, task_name='oc20')

# Gas-phase references (facet-independent), computed once.
eGas = {}
for aname, spec in ADSORBATES.items():
    mol = spec['build']()
    mol.set_cell([20.0, 20.0, 20.0])
    mol.center()
    mol.pbc = True
    relax(mol, runDir / f'ref_gas_{aname}.log')
    eGas[aname] = mol.get_potential_energy()
    write(runDir / f'ref_gas_{aname}.xyz', mol)
print('Gas references (eV): ' + ', '.join(f'{k}={v:.3f}' for k, v in eGas.items()))

# ---------------------------------------------------------------------------
# Per facet: relax the clean slab once, then scan each adsorbate over the sites
# and keep the strongest (most negative E_ads). E_ads = E(slab+ads) - E(slab) - E_gas.
# ---------------------------------------------------------------------------
results = {}
for fname, fd in facetData.items():
    clean = fd['slab'].copy()
    clean.set_constraint(FixAtoms(indices=fd['fixedIdx']))
    relax(clean, runDir / f'ref_slab_{slug(fname)}.log', runDir / f'ref_slab_{slug(fname)}.traj')
    eSlab = clean.get_potential_energy()
    write(runDir / f'ref_slab_{slug(fname)}.xyz', clean)
    print(f'\n{fname}: E_slab = {eSlab:.3f} eV')

    best = {}
    for aname, spec in ADSORBATES.items():
        scan = []
        for sname, xy in fd['sites']:
            system, nSlab = placeAdsorbate(fd['slab'], fd['slabTop'], fd['fixedIdx'], spec, xy)
            tag = f'{slug(fname)}_{aname}_{sname}'
            relax(system, runDir / f'opt_{tag}.log', runDir / f'opt_{tag}.traj')
            write(runDir / f'final_{tag}.xyz', system)
            eAds = system.get_potential_energy() - eSlab - eGas[aname]
            oZn = np.linalg.norm(
                system.positions[:nSlab] - system.positions[nSlab + spec['anchor']], axis=1).min()
            gap = system.positions[nSlab:, 2].min() - fd['slabTop']
            scan.append({'site': sname, 'eAds': eAds, 'oZn': oZn, 'gap': gap, 'atoms': system})
            print(f'    {aname:6s} {sname:7s}: E_ads = {eAds:+.3f} eV  O-Zn = {oZn:.2f}  gap = {gap:.2f}')
        b = min(scan, key=lambda r: r['eAds'])
        best[aname] = b
        write(runDir / f'best_{slug(fname)}_{aname}.xyz', b['atoms'])
        print(f'  best {aname}: {b["site"]}  E_ads = {b["eAds"]:+.3f} eV  O-Zn = {b["oZn"]:.2f} Ang')

    dE = best['glycol']['eAds'] - best['water']['eAds']
    results[fname] = {'eSlab': eSlab, 'best': best, 'dE': dE}

# ---------------------------------------------------------------------------
# Summary: the competitive quantity dE = E_ads(glycol) - E_ads(water) per facet.
# ---------------------------------------------------------------------------
lines = [
    'Competitive adsorption: ethylene glycol vs H2O on Zn facets',
    'dE = E_ads(glycol) - E_ads(water);  dE < 0 => glycol can displace water',
    f'Gas refs (eV): ' + ', '.join(f'{k}={v:.3f}' for k, v in eGas.items()),
    '',
    f'{"facet":10s} {"E_ads(water)":>13s} {"E_ads(glycol)":>14s} {"dE":>8s}   verdict',
]
for fname, r in results.items():
    w = r['best']['water']['eAds']
    g = r['best']['glycol']['eAds']
    verdict = 'glycol displaces water' if r['dE'] < 0 else 'water wins (glycol cannot displace)'
    lines.append(f'{fname:10s} {w:>13.3f} {g:>14.3f} {r["dE"]:>+8.3f}   {verdict}')

lines += [
    '',
    'Note: neutral molecules, vacuum, no applied potential/explicit electrolyte.',
    'Treat as relative mechanistic screening, not absolute interfacial energetics.',
    'Trust dE (a difference) far more than the individual absolute E_ads values.',
]
summary = '\n'.join(lines)
(runDir / 'summary.txt').write_text(summary)
print('\n' + summary)
print(f'\nDone. Output written to {runDir}')
