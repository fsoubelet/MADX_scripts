# AC Dipole MAD-X Scripts

## Contents

This directory contains various MAD-X masks for a study involving AC dipole tracking, investigating the amplitude detuning induced by sextupoles in the LHC.
Various masks have all nonlinearity sources but sextupoles turned off (crossing angles, separation bumps, octupoles, errors etc) so that the only possible non-linearities (and thus amplitude detuning) come from sextupoles.

The directory includes the following files:
- `ac_kick_tracking_template.mask`: template for tracking under AC dipole motion.
- `amplitude_offset_tracking_template.mask`: template for tracking free oscillations with a given initial amplitude offset.
- `amplitude_offset_tracking_driven_tune_template.mask`: template for tracking free oscillations with a given initial amplitude offset, but after matching to driven tune from the AC dipole templates.
- `template.mask`: general template to adapt.

## Usage

The templates are meant to be called from my own python script, but needs `afs` access as they will call files from there.
The `DetlaQx` and `DetlaQy` values, corresponding to the AC dipole kicks, should be set manually.
The rest of parameters are to be given to the python script and will be replaced in the templates.