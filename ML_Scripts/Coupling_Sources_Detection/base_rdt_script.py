"""
This script starts a batch of simulations of given errors that introduce coupling, and returns the true
errors introduced as well as the result of a TWISS call + calculated coupling RDTs from 'optics_functions'.

For this script, the errors are distributed with a 'value + 0.05 * value * TGAUSS(2.5)' command,
with the standard value being provided at the commandline (or by the htcondor_submitter). This means all
errors will be closely distributed around the same value, which given the phase advances in the IR should
be a 'worst case' scenario.

Seeds run concurrently through joblib's threading backend. If using HTCondor, make sure to request enough
CPUs when increasing the number of seeds, or your jobs will run out of memory.
"""
import multiprocessing
from dataclasses import dataclass
from pathlib import Path
from typing import List

import cpymad
import numpy as np
import pandas as pd
import pyhdtoolkit
import tfs
from cpymad.madx import Madx
from joblib import Parallel, delayed
from loguru import logger
from optics_functions.coupling import coupling_via_cmatrix
from pyhdtoolkit.cpymadtools import matching, special, twiss, utils
from pyhdtoolkit.utils import defaults

# ----- Setup ----- #

# fmt: off
MONITOR_TWISS_COLUMNS = ["name", "s", "betx", "alfx", "bety", "alfy", "mux", "muy", "dx", "dy", "dpx", "dpy",
                         "x", "y", "ddx", "ddy", "k1l", "k1sl", "k2l", "k3l", "k4l", "wx", "wy", "phix",
                         "phiy", "dmux", "dmuy", "keyword", "dbx", "dby", "r11", "r12", "r21", "r22"]
# fmt: on
KQSX_KNOBS = [f"kqsx3.{side}{ip}" for side in ("r", "l") for ip in (1, 2, 5, 8)]
PATHS = {
    "optics2018": Path("/afs/cern.ch/eng/lhc/optics/runII/2018"),
    "local": Path("/Users/felixsoubelet/cernbox/OMC/MADX_scripts/Local_Coupling"),
    "htc_outputdir": Path("Outputdata"),
}

defaults.config_logger(level="INFO", enqueue=True)  # goes to stdout
logger.add(
    PATHS["htc_outputdir"] / "full_pylog.log",
    format=defaults.LOGURU_FORMAT,
    enqueue=True,
    level="DEBUG",
)

# ----- Utilities ----- #


@dataclass
class ScenarioResult:
    twiss_rdts: tfs.TfsDataFrame
    error_table: tfs.TfsDataFrame


def fullpath(filepath: Path) -> str:
    """Necessary for AFS paths."""
    return str(filepath.absolute())


def call_paths(madx: Madx, location: str = "afs", opticsfile: str = "opticsfile.22") -> None:
    """Call optics from the given location, either 'afs' or 'local'."""
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


def get_bpms_twiss_and_rdts(madx: Madx) -> tfs.TfsDataFrame:
    """
    Run a TWISS for all BPMs on the currently active sequence, compute RDTs through CMatrix approach and
    return the  aggregated results.
    """
    twiss_tfs = twiss.get_pattern_twiss(  # need chromatic flag as we're dealing with coupling
        madx, patterns=["^BPM.*B[12]$"], columns=MONITOR_TWISS_COLUMNS, chrom=True
    )
    twiss_tfs.columns = [col.upper() for col in twiss_tfs.columns]  # optics_functions needs capitalized
    # names
    twiss_tfs[["F1001REAL", "F1001IMAG", "F1010REAL", "F1010IMAG"]] = coupling_via_cmatrix(
        twiss_tfs, output=["rdts"], complex_columns=False
    )
    return twiss_tfs


# ----- Simulation ----- #


def make_simulation(
    kqsx3: float = 0.0, location: str = "afs", opticsfile: str = "opticsfile.22"
) -> ScenarioResult:
    """
    Get a complete LHC setup, implement coupling sources as given errors (for now, powering values of the
    various  MQSX3s). The coupling RDTs are calculated from a Twiss call at monitor elements throughout
    the machine, through a CMatrix approach.

    Args:
        kqsx3 (float): powering value for IR skew quadrupolar correctors. To be provided throught htcondor
            submitter if running in CERN batch. Defaults to 0. TODO: implement as a TGAUSS?
        location (str): where the scripts are running, which dictates where to get the lhc sequence and
            opticsfile. Can be 'local' and 'afs', defaults to 'afs'.
        opticsfile (str): name of the optics configuration file to use. Defaults to 'opticsfile.22'.

    Returns:
        A custom dataclass holding both the twiss result including coupling RDTs and the assigned errors
        table.
    """
    with Madx(command_log=fullpath(PATHS["htc_outputdir"] / "cpymad_commands.log")) as madx:
        # ----- Init ----- #
        logger.info(f"Running with KQSX3 value of {kqsx3:.1E}")
        madx.option(echo=False, warn=False)
        madx.option(rand="best", randid=np.random.randint(1, 11))  # random number generator
        madx.eoption(seed=np.random.randint(1, 999999999))  # not using default seed

        # ----- Machine Setup ----- #
        call_paths(madx, location, opticsfile)
        special.re_cycle_sequence(madx, sequence="lhcb1", start="MSIA.EXIT.B1")  # same re-cycling as model
        special.make_lhc_beams(madx, energy=7000, emittance=3.75e-6)
        madx.command.use(sequence="lhcb1")
        matching.match_tunes_and_chromaticities(madx, "lhc", "lhcb1", 62.31, 60.32, 2.0, 2.0, calls=200)

        # ----- Power MQSX3s, Twiss and RDTs ----- #
        logger.info(f"Powering MQSX3s")  # start with small values so we don't need the coupling knobs atm
        with madx.batch():
            madx.globals.update({knob: kqsx3 for knob in KQSX_KNOBS})  # TODO: implement as TGAUSS errors?
        twiss_rdts = get_bpms_twiss_and_rdts(madx)
        # known_errors = utils.get_table_tfs(madx, table_name="")  # TODO: implement as dpsis so we get a table?
        return ScenarioResult(twiss_rdts, tfs.TfsDataFrame())


def gather_batches(kqsx3: float = 0.0, batches: int = 50):
    """
    Parallelize batches of different runs.

    Args:
        kqsx3 (float): powering value for IR skew quadrupolar correctors. To be provided throught htcondor
            submitter if running in CERN batch. Defaults to 0. TODO: implement as a TGAUSS?
        batches (int): the number of batches to run.
    """
    # Using Joblib's threading backend as computation happens in MAD-X who releases the GIL
    # Also because cpymad itself uses theads and a multiprocessing backend would refuse that
    n_threads = int(multiprocessing.cpu_count() / 2)  # to ease the memory stress on HTCondor nodes

    # ----- Run simulations concurrently ----- #
    logger.info(f"Computing using Joblib's 'threading' backing, with {n_threads} threads")
    results: List[ScenarioResult] = Parallel(n_jobs=n_threads, backend="threading")(
        delayed(make_simulation)(kqsx3) for _ in range(seeds)
    )

    # ----- Aggregate ----- #
    logger.info("Aggregating results from all seeds")
    return  # TODO: aggregated df?


# ----- Running ----- #


if __name__ == "__main__":
    with Madx(stdout=False) as mad:
        logger.critical(
            f"Using: pyhdtoolkit {pyhdtoolkit.__version__} | cpymad {cpymad.__version__}  | {mad.version}"
        )
    training_set = gather_batches()
    # TODO: write to disk (hdf5?)
