# Hunting a real binding state: glycolate deprotonation + defect sites on Zn

**Date:** 2026-06-21
**Status:** approved (user delegated all technical decisions — "just do")
**Builds on:** `compete-glycol-water-1` (see `stdout/artifacts/`), memories
`peg1-zn002-weak-binding`, `azib-research-direction`, `uma-oc20-no-dispersion-underbinds`.

## Problem

Every adsorption run so far shows neutral ethylene glycol and water only weakly
*physisorbing* on flat Zn — floating ~4 Å on Zn(002), ~2.9 Å on Zn(100), no Zn–O
chemical bond anywhere (global min Zn–O = 2.85 Å). That is the physically correct
regime for neutral closed-shell molecules on bare metal at the PZC in vacuum, but it
leaves the user's question open: **is there any state in this model where the additive
actually chemisorbs?**

Two physically-motivated channels are known to produce real binding and are missing
from the current flat-terrace / neutral-molecule setup:

1. **Deprotonation → alkoxide chemisorption.** Removing a hydroxyl H gives an
   alkoxide O with an unsatisfied valence that forms a genuine Zn–O bond (~1.9–2.1 Å).
2. **Under-coordinated surface sites.** Adatoms (nucleation/dendrite tips) and
   vacancies expose low-coordination Zn that binds adsorbates far more strongly than
   an ideal terrace.

## Goal

Extend `src/main.py`'s scan so a single run answers: *does glycol chemisorb when it
deprotonates, and/or when it meets an under-coordinated Zn site, where the neutral
molecule on the flat terrace does not?* Keep the existing competitive `dE` framing
intact.

## Key design decisions

### 1. Glycolate modeled as dissociative adsorption, neutral, referenced to ½H₂

The deprotonated species is, in isolation, a charged anion or an open-shell radical —
neither of which the `oc20` task head (neutral, closed-shell, RPBE-flavored) handles
cleanly. We avoid the problem entirely with a Hess's-law / dissociative-adsorption
reference. The *combined* slab+fragment system stays neutral (the alkoxide's dangling
electron pairs into the metal — exactly an OC20-style O-on-metal bond), and the
stripped proton is accounted for as ½H₂ gas:

```
glycol(gas) + slab  ->  glycolate(O-bound to Zn) + ½ H₂(gas)

E_chem = E(slab + glycolate) + ½·E(H₂) − E(slab) − E_gas(glycol)
       = E(slab + glycolate) − E(slab) − [ E_gas(glycol) − ½·E(H₂) ]
```

So glycolate plugs into the existing `E_ads = E(system) − E_slab − E_ref` machinery
with `E_ref(glycolate) = E_gas(glycol) − ½·E(H₂)`. No isolated charged/radical species
is ever computed; everything stays on `oc20` and comparable to the water/glycol
numbers. Bonus: the ½H₂ term is literally the hydrogen-evolution side reaction, so
`E_chem < 0` flags glycol deprotonation as a possible HER promoter on that facet.

**Interpretation metrics added to the summary:**
- `dE_displace = E_ads(glycol) − E_ads(water)` — existing competitive metric (per facet).
- `dE_deprot   = E_ads(glycolate) − E_ads(glycol)` — does deprotonation deepen binding?
- per-(facet,adsorbate): best `E_ads`, min O–Zn distance, and a
  chemisorbed (<2.4 Å) / physisorbed (2.4–3.2 Å) / floating (>3.2 Å) label.

### 2. Defect sites = adatom + vacancy on Zn(002)

Built by trivial, robust edits to a flat `hcp0001` slab (no fragile stepped-surface
construction):
- **adatom:** append one Zn at a top-layer hollow xy, ~2.0 Å above the surface; it
  becomes the highest atom and the adsorption site sits atop it. Mimics a growing
  nucleation/dendrite tip (the most AZIB-relevant defect).
- **vacancy:** remove one top-layer atom near the cell centre; the adsorption site is
  the vacancy pocket, exposing under-coordinated neighbours and the second layer.

Zn(002) is chosen because it is the otherwise-inert basal plane — showing that the
*same* facet binds once a defect or deprotonation is introduced is the cleanest result.
The clean-slab reference for each defect facet is the relaxed defected slab *without*
adsorbate, so `E_ads` isolates adsorbate binding to the defect.

### 3. Scope (one self-contained run)

| facet           | adsorbates                | sites                  |
|-----------------|---------------------------|------------------------|
| Zn(002)-flat    | water, glycol, glycolate  | atop / bridge / hollow |
| Zn(100)-flat    | water, glycol, glycolate  | atop / bridge / hollow |
| Zn(002)-adatom  | water, glycol, glycolate  | adatom top             |
| Zn(002)-vacancy | water, glycol, glycolate  | vacancy pocket         |

≈ (9 + 9 + 3 + 3) = 24 relaxations + 4 slab refs + 3 gas refs (water, glycol, H₂).
Flat facets keep the 3-site scan; defect facets use their single defect site. Run
prefix: `glycolate-defect-1`.

## Architecture / changes to `src/main.py`

Small, additive changes that reuse the existing scan loop:

- `buildGlycolate()` — `buildGlycol()` with the O1 hydroxyl H (index 8) removed →
  `C2O2H5`; anchor stays O1 (index 2), neighbours `(0,)` (only C1). `orientAnchorDown`
  already works with a single neighbour (aligns C1→O1 to +z ⇒ alkoxide O-down).
- `ADSORBATES['glycolate']` — `{build, anchor:2, neighbours:(0,), ref: lambda eGas,eH2: eGas['glycol'] - 0.5*eH2}`.
  water/glycol use the default `ref = eGas[name]`.
- `buildFacet(name)` extended to accept `'Zn(002)-adatom'` and `'Zn(002)-vacancy'`,
  returning the same `(slab, fixedIdx, slabTop, sites)` tuple with a single defect
  `site` and `slabTop` set to the adatom height (adatom) or the remaining top (vacancy).
- Gas-reference section also relaxes an H₂ molecule → `eH2`.
- `E_ads` line uses `spec['ref'](eGas, eH2)` when present, else `eGas[aname]`.
- Summary extended with `dE_deprot`, the chemisorption/physisorption/floating label,
  and a per-facet verdict. Keep the existing caveat block; add a note that glycolate
  is a dissociative (½H₂-referenced) energy, not a molecular adsorption energy.

## Testing

The calculator-dependent relaxations are not unit-testable, but every *geometry and
bookkeeping* helper is deterministic and will be unit-tested without loading the model
(`tests/test_geometry.py`, plain asserts or pytest if available):

- `buildGlycolate()` has composition C2O2H5, exactly one fewer H than `buildGlycol()`,
  and the removed atom is the O1 hydroxyl H (O1 retains a single bonded neighbour).
- `orientAnchorDown` with a single neighbour puts the anchor O at the minimum z of the
  molecule (O-down).
- `buildFacet('Zn(002)-adatom')` has exactly one more atom than flat, the adatom is the
  unique highest atom, and it is *not* in `fixedIdx`.
- `buildFacet('Zn(002)-vacancy')` has exactly one fewer top-layer atom than flat.
- `E_ref(glycolate) == eGas['glycol'] − 0.5·eH2` (reference-energy bookkeeping).
- `placeAdsorbate` puts the anchor atom at `slabTop + height` for a defect facet.

The existing "geometry first, no calculator" fail-fast block at the top of the script
is retained (and now also writes the defect/glycolate initial placements).

## Out of scope (deliberately deferred)

- True charged anion (`omol` head), explicit co-adsorbed H / full HER pathway, applied
  potential / EDL, explicit solvent or water clusters, stepped (hkl) surfaces, Zn(100)
  defects, DFT cross-check. These are follow-ups; this run isolates the two highest-value
  binding channels with the smallest, safest code change.
