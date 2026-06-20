from ase import Atoms
from ase.build import fcc100, add_adsorbate
from ase.optimize import LBFGS
from fairchem.core import pretrained_mlip, FAIRChemCalculator



predictor = pretrained_mlip.get_predict_unit('uma-s-1p2', device='cuda')
calc = FAIRChemCalculator(predictor, task_name='oc20')

# Zn(100) slab.
# Note: Zn is natively hcp, so ASE has no built-in fcc lattice constant for it.
# We model the cubic (100) face explicitly; a ~= a_hcp * sqrt(2) ~= 3.77 Ang keeps
# the nearest-neighbour distance close to bulk Zn. Adjust `a` if you have a
# preferred reference value.

slab = fcc100('Zn', (3, 3, 3), a=3.77, vacuum=8, periodic=True)

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

add_adsorbate(slab, adsorbate, 2.0, 'bridge')

slab.calc = calc

# Set up LBFGS dynamics object
opt = LBFGS(slab)
opt.run(0.05, 100)