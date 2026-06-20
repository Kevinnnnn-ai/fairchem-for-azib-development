from pathlib import Path
import numpy as np
from ase import Atoms
from ase.build import hcp0001
from ase.io import write
from ase.optimize import LBFGS
from fairchem.core import pretrained_mlip, FAIRChemCalculator



def getRunDir(prefix='peg1-zn002'):
    runDirRoot = Path(__file__).resolve().parent.parent / 'stdout' / 'runs'
    runDirRoot.mkdir(parents=True, exist_ok=True)
    i = 1
    while (runDirRoot / f'{prefix}-{i}').exists():
        i += 1

    runDir = runDirRoot / f'{prefix}-{i}'
    runDir.mkdir()
    return runDir



runDir = getRunDir()
print(f'Saving run output to {runDir}')

predictor = pretrained_mlip.get_predict_unit('uma-s-1p2', device='cuda')
calc = FAIRChemCalculator(predictor, task_name='oc20')

# Zn(002) slab = the basal plane of hcp Zn (the (002) reflection is the
# second order of the basal (001)/(0001) plane). This is the thermodynamically
# stable, dendrite-free facet that PEG promotes during AZIB electrodeposition,
# so it is the conventional electrode model for a PEG/Zn interface study.
# hcp0001 builds it from Zn's real measured lattice constants (a=2.66 Ang,
# c/a=1.856), so no estimated constant is needed. size=(3, 3, 4) -> a 3x3
# surface supercell, 4 atomic layers.

slab = hcp0001('Zn', size=(3, 3, 4), vacuum=8.0, periodic=True)

# PEG-1 = monoethylene glycol (ethylene glycol), HO-CH2-CH2-OH (C2H6O2).
# Not available in ASE's molecule() database, so build it from explicit
# coordinates (approximate tetrahedral geometry; the optimiser relaxes it).

adsorbate = Atoms(
    'C2O2H6',
    positions=[
        (0.000, 0.000, 0.000),    # C1
        (1.520, 0.000, 0.000),    # C2
        (-0.473, 1.339, 0.000),   # O1 (on C1)
        (1.993, -1.339, 0.000),   # O2 (on C2)
        (-0.363, -0.514, 0.890),  # H on C1
        (-0.363, -0.514, -0.890), # H on C1
        (1.883, 0.514, 0.890),    # H on C2
        (1.883, 0.514, -0.890),   # H on C2
        (0.289, 1.923, 0.000),    # H on O1 (hydroxyl)
        (2.953, -1.315, 0.000),   # H on O2 (hydroxyl)
    ],
)

# Centre PEG-1 laterally on the slab and sit it `height` above the top atoms,
# rather than at a corner site, so it stays within the cell (outlined) borders.

height = 2.0
centerXY = slab.cell[:2, :2].sum(axis=0) / 2.0
topZ = slab.positions[:, 2].max()

ads = adsorbate.copy()
ads.positions[:, :2] += centerXY - ads.positions[:, :2].mean(axis=0)
ads.positions[:, 2] += topZ + height - ads.positions[:, 2].mean()
slab.extend(ads)

# Sanity check in fractional coordinates, which is correct for the
# non-orthogonal hexagonal surface cell: every adsorbate atom must lie inside
# the cell parallelogram (0 <= a, b <= 1).

adsFrac = slab.cell.scaled_positions(slab.positions[-len(adsorbate):])[:, :2]
if not (adsFrac.min() >= 0.0 and adsFrac.max() <= 1.0):
    raise ValueError('PEG-1 extends beyond the slab cell borders')

print(
    f'PEG-1 centred at ({centerXY[0]:.2f}, {centerXY[1]:.2f}) Ang; '
    f'fractional a[{adsFrac[:, 0].min():.2f}, {adsFrac[:, 0].max():.2f}] '
    f'b[{adsFrac[:, 1].min():.2f}, {adsFrac[:, 1].max():.2f}] within the cell'
)

slab.calc = calc

# Save the starting structure before relaxation.
write(runDir / 'initial.xyz', slab)

# Set up LBFGS dynamics object, logging the optimisation into the run directory.
opt = LBFGS(
    slab,
    logfile=str(runDir / 'opt.log'),
    trajectory=str(runDir / 'opt.traj'),
)

# No step cap: run until convergence (fmax <= 0.05 eV/Ang) or manual termination.
opt.run(fmax=0.05)

# Save the relaxed structure.
write(runDir / 'final.xyz', slab)
print(f'Done. Output written to {runDir}')