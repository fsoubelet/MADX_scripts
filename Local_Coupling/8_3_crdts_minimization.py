"""
This script is used to start a batch simulations looking at the coupling RDTs and CRDTs calculation in
the presence of the colinearity knob in an IR (here IR1) with also a small value of the coupling knob and
some magnet tilt errors in the IRs.

The measurement is simulated by tracking a particle and defining BPMs as observation points. The BPMs
are taken from the model directory, which should be made ahead of time with omc3's model_creator as shown
below.

To create the appropriate model through Python:
```python
from pathlib import Path
from omc3.model_creator import create_instance_and_model
create_instance_and_model(
    accel="lhc",
    type="nominal",
    year="2018",
    beam=1,
    energy=6.5,
    nat_tunes=[62.31, 60.32],
    modifiers=[Path("/afs/cern.ch/eng/lhc/optics/runII/2018/PROTON/opticsfile.22")],
    outputdir=Path("/afs/cern.ch/work/f/fesoubel/lhc_model"),  # or wherever
)
```

Need to provide at HTCondor submission time with `job_submitter`:
- DPSI_MEAN -> mean value of the DPSI tilt distribution (this one isn't narrow)
"""
import json
from pathlib import Path
from typing import Tuple, Union

import cpymad
import numpy as np
import pyhdtoolkit
import tfs
from cpymad.madx import Madx, TwissFailed
from loguru import logger
from omc3.hole_in_one import hole_in_one_entrypoint as hole_in_one
from omc3.tbt_converter import converter_entrypoint as tbt_converter
from omc3.utils import logging_tools
from pydantic import BaseModel
from pyhdtoolkit.cpymadtools import errors, matching, orbit, special, track
from pyhdtoolkit.utils import defaults
from scipy.optimize import Bounds, minimize

# ----- Setup ----- #

defaults.config_logger(level="DEBUG")
LOGGER = logging_tools.get_logger(__name__)  # to get omc3 logging

PATHS = {
    "optics2018": Path("/afs/cern.ch/eng/lhc/optics/runII/2018"),
    "local": Path("/Users/felixsoubelet/cernbox/OMC/MADX_scripts/Local_Coupling"),
    "htc_outputdir": Path("Outputdata"),
}


# ----- Utilities ----- #


def fullpath(filepath: Path) -> str:
    return str(filepath.absolute())


class Results(BaseModel):
    tilt_mean: float = None
    kqsx3_l1_value: float = None
    kqsx3_r1_value: float = None
    fxy_l1_value: float = None
    fxy_r1_value: float = None

    def to_json(self, filepath: Union[str, Path]) -> None:
        logger.debug(f"Exporting results structure to '{Path(filepath).absolute()}'")
        with Path(filepath).open("w") as results_file:
            json.dump(self.dict(), results_file)


def generate_errors(tilt_mean: float = 0.0, location: str = "afs", opticsfile: str = "opticsfile.22") -> None:
    """
    Get an LHC setup, implement triplet errors (DPSI) and export them to disk to be loaded later on. This
    is used to ensure that all steps ran during the scipy optimization process have the exact same error
    distribution.

    Args:
        tilt_mean (float): mean value of the dpsi tilt distribution when applying to quadrupoles.
        location (str): where the scripts are running, which dictates where to get the lhc sequence and
            opticsfile. Can be 'local' and 'afs', defaults to 'afs'.
        opticsfile (str): name of the optics configuration file to use. Defaults to 'opticsfile.22'.

    Returns:
        None, will write the errors table to disk.
    """
    with Madx(stdout=False) as madx:
        # ----- Init ----- #
        logger.info(f"Running with a mean tilt of {tilt_mean:.1E}")
        madx.option(echo=False, warn=False)
        madx.option(rand="best", randid=np.random.randint(1, 11))  # random number generator
        madx.eoption(seed=np.random.randint(1, 999999999))  # not using default seed

        # ----- Machine ----- #
        logger.debug("Calling optics")
        if location == "afs":
            madx.call(fullpath(PATHS["optics2018"] / "lhc_as-built.seq"))
            madx.call(fullpath(PATHS["optics2018"] / "PROTON" / opticsfile))
        elif location == "local":
            madx.call(fullpath(PATHS["local"] / "sequences" / "lhc_as-built.seq"))
            madx.call(fullpath(PATHS["local"] / "optics" / opticsfile))
        else:
            logger.error("Unknown parameter location, exiting")
            raise ValueError("The 'location' parameter should be either 'afs' or 'local'.")

        # ----- Setup ----- #
        special.re_cycle_sequence(madx, sequence="lhcb1", start="MSIA.EXIT.B1")  # same re-cycling as model
        _ = orbit.setup_lhc_orbit(madx, scheme="flat")
        special.make_lhc_beams(madx, energy=6500, emittance=3.75e-6)
        madx.use(sequence="lhcb1")
        # matching.match_tunes_and_chromaticities(madx, "lhc", "lhcb1", 62.31, 60.32, 2.0, 2.0, calls=200)

        # ----- Errors ----- #
        logger.info("Applying misalignments to IR quads 1 to 6")
        errors.misalign_lhc_ir_quadrupoles(  # 5 percent deviation
            madx,
            ip=1,
            beam=1,
            quadrupoles=[1, 2, 3, 4, 5, 6],
            sides="RL",
            dpsi=f"{tilt_mean} * TGAUSS(2.5)",  # f"{tilt_mean} + {abs(tilt_mean) * 0.05} * TGAUSS(2.5)",
        )

        logger.info("Exporting errors to disk at 'errors.tfs'")
        madx.command.select(flag="error", full=True)
        madx.command.esave(file="errors.tfs")

    errs = tfs.read("errors.tfs", index="NAME")
    logger.info(f"Assigned errors:\n {errs.DPSI[errs.DPSI != 0]}")


# ----- Simulation ----- #


def make_simulation(
    correctors: Tuple[float, float],
    lhc_model_dir: str = "/afs/cern.ch/work/f/fesoubel/htcondor_results/8.3_minimization_study/lhc_model",
    coupling_knob: float = 2e-3,
    location: str = "afs",
    opticsfile: str = "opticsfile.22",
) -> float:
    """
    Get a complete LHC setup, implement coupling knob, implement colinearity knob in IR, perform tracking
    and do data analysis on the output. It returns the mean of CRDTs at the inner BPMs locations.

    Args:
        correctors (Tuple[float, float]): setting value of the left and right kqsx3 correctors.
        lhc_model_dir (str): location of a single model dir for all simulations to tap into.
        coupling_knob (float): value of the coupling knob.
        location (str): where the scripts are running, which dictates where to get the lhc sequence and
            opticsfile. Can be 'local' and 'afs', defaults to 'afs'.
        opticsfile (str): name of the optics configuration file to use. Defaults to 'opticsfile.22'.

    Returns:
        None, the outputs are analysis results from `omc3` and will be written to disk.
    """
    kqsx3_left, kqsx3_right = correctors
    logger.info(f"LEFT:\t {kqsx3_left}")
    logger.info(f"RIGHT:\t {kqsx3_right}")

    logger.debug("Loading observation BPMs from model")
    model_twiss_path = Path(lhc_model_dir).resolve() / "twiss.dat"
    observation_bpms = tfs.read(model_twiss_path).NAME.tolist()

    with Madx(stdout=False) as madx:
        # ----- Init ----- #
        logger.info(f"Running with corrector settings of {correctors}")
        madx.option(echo=False, warn=False)

        # ----- Machine ----- #
        logger.debug("Calling optics")
        if location == "afs":
            madx.call(fullpath(PATHS["optics2018"] / "lhc_as-built.seq"))
            madx.call(fullpath(PATHS["optics2018"] / "PROTON" / opticsfile))
        elif location == "local":
            madx.call(fullpath(PATHS["local"] / "sequences" / "lhc_as-built.seq"))
            madx.call(fullpath(PATHS["local"] / "optics" / opticsfile))
        else:
            logger.error("Unknown parameter location, exiting")
            raise ValueError("The 'location' parameter should be either 'afs' or 'local'.")

        # ----- Setup ----- #
        special.re_cycle_sequence(madx, sequence="lhcb1", start="MSIA.EXIT.B1")  # same re-cycling as model
        _ = orbit.setup_lhc_orbit(madx, scheme="flat")
        special.make_lhc_beams(madx, energy=6500, emittance=3.75e-6)
        madx.use(sequence="lhcb1")

        # ----- Load Errors ----- #
        logger.info("Loading triplet errors from file")
        madx.command.readtable(file="errors.tfs", table="errors")
        madx.command.seterr(table="errors")

        # ----- Knobs Settings ----- #
        logger.info(f"Applying coupling and colinearity knobs settings")
        matching.match_tunes_and_chromaticities(madx, "lhc", "lhcb1", 62.27, 60.36, 2.0, 2.0)  # avoid flip
        special.apply_lhc_coupling_knob(madx, coupling_knob=coupling_knob, beam=1, telescopic_squeeze=True)
        with madx.batch():
            madx.globals.update({"KQSX3.L1": kqsx3_left * 1e-4, "KQSX3.R1": kqsx3_right * 1e-4})
        matching.match_tunes_and_chromaticities(madx, "lhc", "lhcb1", 62.31, 60.32, 2.0, 2.0)  # working point

        # ----- Tracking ----- #
        special.make_lhc_thin(madx, sequence="lhcb1", slicefactor=4)
        logger.info("Doing tracking")
        _ = track.track_single_particle(
            madx,
            sequence="lhcb1",
            nturns=1023,
            initial_coordinates=(2e-4, 0, 2e-4, 0, 0, 0),
            observation_points=observation_bpms,
            FILE="Outputdata/track",  # MAD-X will append "one" to this name since we give ONETABLE
            RECLOSS=True,  # Create an internal table recording lost particles
            ONEPASS=True,  # Do not search closed orbit and give coordinates relatively to the reference orbit
            DUMP=True,  # Write to file
            ONETABLE=True,  # Gather all observation points data into a single table (and file if DUMP set)
        )
    assert Path("Outputdata/trackone").is_file()

    logger.info("Doing 'trackone' file conversion with omc3's 'tbt_converter'")
    tbt_converter(
        files=[Path("Outputdata/trackone")],
        tbt_datatype="trackone",
        outputdir="Outputdata",  # will create 'Outputdata/trackone.sdds'
        drop_elements=["LHCB1MSIA.EXIT.B1_P_"],  # start of machine point added by MAD-X
    )
    Path("stats.txt").unlink()

    logger.info("Performing harmonic and optics analysis of tracking data with omc3's 'hole_in_one'")
    hole_in_one(
        harpy=True,
        files=[Path("Outputdata/trackone.sdds")],
        tbt_datatype="lhc",
        turns=[0, 1023],
        autotunes="transverse",
        to_write=["lin", "spectra", "full_spectra", "bpm_summary"],  # in `lin_files` subfolder of outputdir
        turn_bits=15,  # it's clean simulation data so we don't need too much here, let's speed up
        optics=True,
        accel="lhc",
        year="2018",
        beam=1,
        energy=6.5,
        model_dir=Path(lhc_model_dir),
        compensation="none",
        three_bpm_method=True,
        nonlinear=["crdt"],  # only interested in the CRDTs here
        outputdir=Path("Outputdata/measured_optics"),
    )

    logger.info("Querying CRDTs data from results on disk")
    F_XY = tfs.read("Outputdata/measured_optics/crdt/skew_quadrupole/F_XY.tfs", index="NAME")
    result = F_XY[F_XY.index.str.contains("BPMSW.1[RL]1.B1")].AMP.mean()
    logger.success(f"Resulting mean amplitude of CRDT:\t {result}")
    return result


def optimize_crdts(dpsi_mean: float):
    """
    Generate an error distribution, run scipy's optimization, run with the optimized parameters and
    return the results.
    """
    logger.info("Generating errors for simulations")
    generate_errors(tilt_mean=dpsi_mean)

    colin_bounds = Bounds([-20, 20], [-20, 20])  # range of values for the left & right KQSX3 correctors
    attempts: int = 0
    results = None

    # Loop a few times to try and get a decent error generation that allows us to run for the dpsi_mean value
    while results is not None and attempts < 20:
        try:
            optimized_correctors = minimize(
                make_simulation, x0=[-8, -3], method="trust-constr", bounds=colin_bounds
            )
            logger.success(f"Optimized colinearity knob setting: {optimized_correctors.x}")

            logger.info("Running simulation with optimized setting")
            make_simulation(optimized_correctors.x)

            logger.info("Exporting final results")
            F_XY = tfs.read("Outputdata/measured_optics/crdt/skew_quadrupole/F_XY.tfs", index="NAME")
            results = Results(
                tilt_mean=dpsi_mean,
                kqsx3_l1_value=optimized_correctors.x[0],
                kqsx3_r1_value=optimized_correctors.x[1],
                fxy_l1_value=F_XY[F_XY.index.str.contains("BPMSW.1L1.B1")].AMP.to_numpy()[0],
                fxy_r1_value=F_XY[F_XY.index.str.contains("BPMSW.1R1.B1")].AMP.to_numpy()[0],
            )
        except TwissFailed:
            logger.error("Failed configuration!")
        attempts += 1
    return results or Results(tilt_mean=dpsi_mean)


# ----- Running ----- #


if __name__ == "__main__":
    with Madx(stdout=False) as mad:
        logger.critical(
            f"Using: pyhdtoolkit {pyhdtoolkit.__version__} | cpymad {cpymad.__version__}  | {mad.version}"
        )

    simulation_results = optimize_crdts(dpsi_mean=1)  # %(DPSI_MEAN)s,
    simulation_results.to_json(PATHS["htc_outputdir"] / "results.json")
