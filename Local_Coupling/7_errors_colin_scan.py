"""
The previous scripts of step 7 show strange results at higher DPSI values, possibly because MAD-X
struggles with the discrepancy between degrees of freedom and constraints when matching.
This script implements a given high DPSI in Q1 to Q6 quads around an IP (from the previous scripts) and
does a scan of independent MQSX3.[RL]{IP} settings (provide all values for each at the commandline through
PyLHC's 'job_submitter') to see if MAD-X was struggling by itself, or if the conditions are indeed too hard.

This is a very fast script that only does some matchings as the hardest operation, it should be flavor
'espresso', and 'microcentury' at worst if AFS is slow today.
"""
import json
import multiprocessing
import sys
from pathlib import Path
from typing import List, Sequence, Tuple, Union

import cpymad
import numpy as np
import pandas as pd
import pyhdtoolkit
from cpymad.madx import Madx
from joblib import Parallel, delayed
from loguru import logger
from pydantic import BaseModel
from pyhdtoolkit.cpymadtools import errors, matching, orbit, special
from pyhdtoolkit.utils import defaults

# ----- Setup ----- #

PATHS = {
    "optics2018": Path("/afs/cern.ch/eng/lhc/optics/runII/2018"),
    "local": Path("/Users/felixsoubelet/cernbox/OMC/MADX_scripts/Local_Coupling"),
    "htc_outputdir": Path("Outputdata"),
}
logger.add(
    PATHS["htc_outputdir"] / "full_pylog.log",
    format=defaults.LOGURU_FORMAT,
    enqueue=True,
    level="DEBUG",
)
defaults.config_logger(level="INFO")

# ----- Utilities ----- #


class Results(BaseModel):
    tilt_mean: float
    kqsx3_l1_value: float
    kqsx3_r1_value: float
    beta12_value: float
    beta21_value: float

    def to_json(self, filepath: Union[str, Path]) -> None:
        logger.debug(f"Exporting results structure to '{Path(filepath).absolute()}'")
        with Path(filepath).open("w") as results_file:
            json.dump(self.dict(), results_file)

    @classmethod
    def from_json(cls, json_file: Union[str, Path]):
        """
        Load a Pokemon instance's data from disk, saved in the JSON format.
        Args:
                json_file (Union[Path, str]): PosixPath object or string with the save file location.
        """
        logger.info(f"Loading JSON data from file at '{Path(json_file).absolute()}'")
        return cls.parse_file(json_file, content_type="application/json")


def fullpath(filepath: Path) -> str:
    return str(filepath.absolute())


# ----- Simulation ----- #


def simulate(
    kqsx3_right: float, kqsx3_left: float, tilt_mean: float = 0.0, quadrupoles: List[int] = [1, 2, 3, 4, 5, 6]
) -> Results:
    """
    Get a complete LHC setup, implement coupling at IP with tilt errors, scan the colinearity knob
    settings and gets the resulting cross-term ripken values at IP.
    This is a check of the determined correction (by MAD-X matching) of coupling at IP that I don't trust
    for higher tilt values (empirically, > 0.5mrad mean for all magnets). The colinearity knob elements are
    powered individually.

    Args:
        kqsx3_right (float): the powering value of the right corrector (RQSX3.R[IP]).
        kqsx3_left (float): the powering value of the left corrector (RQSX3.L[IP]).
        tilt_mean (float): mean value of the dpsi tilt distribution when applying to quadrupoles. To be
            provided throught htcondor submitter if running in CERN batch.
        quadrupoles (List[int]) the list of quadrupoles to apply errors to. Defaults to Q1 to A6, to be
            provided throught htcondor submitter if running in CERN batch.

    Returns:
        A Results object with the run parameters and the Ripken beta12 and beta21 at IP.
    """
    with Madx() as madx:
        # ----- Init ----- #
        logger.info(f"Running with a mean tilt of {tilt_mean:.1E}")
        madx.option(echo=False, warn=False)
        madx.option(rand="best", randid=np.random.randint(1, 11))  # random number generator
        madx.eoption(seed=np.random.randint(1, 999999999))  # not using default seed

        # ----- Machine ----- #
        logger.debug("Calling optics")
        # madx.call(fullpath(PATHS["optics2018"] / "lhc_as-built.seq"))  # afs
        # madx.call(fullpath(PATHS["optics2018"] / "PROTON" / "opticsfile.22"))  # afs
        # madx.call(fullpath(PATHS["local"] / "sequences" / "lhc_as-built.seq"))  # local testing
        # madx.call(fullpath(PATHS["local"] / "optics" / "opticsfile.22"))  # local testing

        # ----- Setup ----- #
        special.re_cycle_sequence(madx, sequence="lhcb1", start="IP3")
        orbit_scheme = orbit.setup_lhc_orbit(madx, scheme="flat")
        special.make_lhc_beams(madx, energy=6500, emittance=3.75e-6)
        madx.use(sequence="lhcb1")
        # if no slicing, tunes need to be very apart to avoid a tune flip when applying the knobs,
        matching.match_tunes_and_chromaticities(madx, "lhc", "lhcb1", 62.27, 60.36, 2.0, 2.0, calls=200)

        # ----- Errors ----- #
        logger.info("Applying misalignments to IR quads 1 to 6")
        errors.misalign_lhc_ir_quadrupoles(  # 5 percent deviation
            madx,
            ip=1,
            beam=1,
            quadrupoles=quadrupoles,
            sides="RL",
            dpsi=f"{tilt_mean} + {abs(tilt_mean) * 0.05} * TGAUSS(2.5)",
        )

        # ----- Colin Settings ----- #
        logger.info(f"Applying corrector settings: KQSX3.R1 = {kqsx3_right} and KQSX3.L1 = {kqsx3_left}")
        with madx.batch():
            madx.globals.update({"kqsx3.r1": kqsx3_right, "kqsx3.l1": kqsx3_left})
        matching.match_tunes_and_chromaticities(madx, "lhc", "lhcb1", 62.31, 60.32, 2.0, 2.0, calls=200)
        madx.twiss(ripken=True)
        twiss_df = madx.table.twiss.dframe().copy().set_index("name", drop=True)

    return Results(
        tilt_mean=tilt_mean,
        kqsx3_r1_value=kqsx3_right,
        kqsx3_l1_value=kqsx3_left,
        beta12_value=twiss_df.loc[twiss_df.index == "ip1:1"].beta12[0],
        beta21_value=twiss_df.loc[twiss_df.index == "ip1:1"].beta21[0],
    )


# ----- Running ----- #


if __name__ == "__main__":
    with Madx(stdout=False) as mad:
        logger.critical(
            f"Using: pyhdtoolkit {pyhdtoolkit.__version__} | cpymad {cpymad.__version__}  | {mad.version}"
        )
    # simulation_results = simulate(  # afs run
    #     kqsx3_right=%(KQSX3_RIGHT)s,
    #     kqsx3_left=%(KQSX3_LEFT)s,
    #     tilt_mean=%(DPSI_MEAN)s,
    #     quadrupoles=[1, 2, 3, 4, 5, 6],
    # )
    # simulation_results = simulate(  # local testing
    #     kqsx3_right=20e-4,
    #     kqsx3_left=20e-4,
    #     tilt_mean=1e-3,
    #     quadrupoles=[1, 2, 3, 4, 5, 6],
    # )
    simulation_results.to_json(PATHS["htc_outputdir"] / "results.json")
