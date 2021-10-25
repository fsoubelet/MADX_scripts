"""
This script starts a batch of simulations of given errors that introduce coupling, and returns the true
errors introduced as well as the calculated coupling RDTs from 'optics_functions'.

For this script, the errors are distributed with a prodived standard deviation according to the MAD-X
'value * TGAUSS(2.5)' command, with the standard deviation value being provided at the commandline (or by the
htcondor_submitter). Errors are distributed to all quadrupoles.

Seeds run concurrently through joblib's threading backend. If using HTCondor, make sure to request enough
CPUs when increasing the number of seeds, or your jobs will run out of memory.

NOTE: this script requires pyhdtoolkit >= 0.15.1


Need to provide at HTCondor submission time with `job_submitter`:
- DPSI_STD -> standard deviation of the DPSI tilt distribution (this one isn't narrow)
- SEEDS -> the number of different scenarios gathered in this run
"""
import logging
import multiprocessing
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cpymad
import numpy as np
import pyhdtoolkit
import tfs
from cpymad.madx import Madx
from joblib import Parallel, delayed
from loguru import logger
from optics_functions.coupling import coupling_via_cmatrix
from pyhdtoolkit.cpymadtools import errors, lhc, matching, twiss, utils
from pyhdtoolkit.cpymadtools.constants import MONITOR_TWISS_COLUMNS
from pyhdtoolkit.utils import defaults

# ----- Setup ----- #


PATHS = {
    "optics2018": Path("/afs/cern.ch/eng/lhc/optics/runII/2018"),
    "local": Path("/Users/felixsoubelet/cernbox/OMC/MADX_scripts/Local_Coupling"),
    "htc_outputdir": Path("Outputdata"),
}

defaults.config_logger(level="WARNING", enqueue=True)  # goes to stdout
logger.add(  # only kicks in on htcondor or if you create 'Outputdata'
    PATHS["htc_outputdir"] / "full_pylog.log",
    format=defaults.LOGURU_FORMAT,
    enqueue=True,
    level="DEBUG",
)


# ----- Utilities ----- #


@dataclass
class ScenarioResult:
    tilt: float
    coupling_rdts: tfs.TfsDataFrame
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


def get_bpms_coupling_rdts(madx: Madx) -> tfs.TfsDataFrame:
    """
    Run a TWISS for all BPMs on the currently active sequence, compute RDTs through CMatrix approach and
    return the  aggregated results.
    """
    twiss_tfs = twiss.get_pattern_twiss(  # need chromatic flag as we're dealing with coupling
        madx, patterns=["^BPM.*B[12]$"], columns=MONITOR_TWISS_COLUMNS, chrom=True
    )
    twiss_tfs.columns = twiss_tfs.columns.str.upper()  # optics_functions needs capitalized names
    twiss_tfs.NAME = twiss_tfs.NAME.str.upper()
    return coupling_via_cmatrix(twiss_tfs, complex_columns=False, output=["rdts"])


def apply_dpsi_to_all_quads(madx: Madx, tilt_std: float = 0.0) -> None:
    """Apply a the provided stdev as a TGAUSS(2.5) to all LHC quads in the machine."""
    madx.command.select(flag="error", clear=True)
    madx.command.select(flag="error", pattern=r"^MQ\.")
    madx.command.ealign(dpsi=f"{tilt_std} * TGAUSS(2.5)")
    madx.command.etable(table="quad_errors")  # save errors in table
    madx.select(flag="error", clear=True)  # cleanup the flag


# ----- Simulation ----- #


def make_simulation(
    tilt_std: float = 0.0,
    location: str = "afs",
    opticsfile: str = "opticsfile.22",
) -> ScenarioResult:
    """
    Get a complete LHC setup, implement coupling sources as tilt errors in the desired IR quadrupoles. The
    coupling RDTs are calculated from a Twiss call at monitor elements throughout the machine, through a
    CMatrix approach.

    Args:
        tilt_std (float): standard dev of the dpsi tilt distribution when applying to quadrupoles. To be
            provided throught htcondor submitter if running in CERN batch.
        location (str): where the scripts are running, which dictates where to get the lhc sequence and
            opticsfile. Can be 'local' and 'afs', defaults to 'afs'.
        opticsfile (str): name of the optics configuration file to use. Defaults to 'opticsfile.22'.

    Returns:
        A custom dataclass holding both the twiss result including coupling RDTs and the assigned errors
        table.
    """
    try:
        with Madx(stdout=False) as madx:
            # ----- Init ----- #
            logger.info(f"Running with a mean tilt of {tilt_std:.1E}")
            madx.option(echo=False, warn=False)
            madx.option(rand="best", randid=np.random.randint(1, 11))  # random number generator
            madx.eoption(seed=np.random.randint(1, 999999999))  # not using default seed

            # ----- Machine Setup ----- #
            call_paths(madx, location, opticsfile)
            lhc.re_cycle_sequence(madx, sequence="lhcb1", start="MSIA.EXIT.B1")  # same re-cycling as model
            lhc.make_lhc_beams(madx, energy=7000, emittance=3.75e-6)
            madx.command.use(sequence="lhcb1")
            # Tunes are matched from opticsfile and no modification was made so
            # matching.match_tunes_and_chromaticities(madx, "lhc", "lhcb1", 62.31, 60.32, 2.0, 2.0, calls=200)

            # ----- Introduce Errors, Twiss and RDTs ----- #
            logger.info(
                f"Introducing tilts in all quadrupoles"
            )  # small values so we don't need coupling knobs
            apply_dpsi_to_all_quads(madx, tilt_std=tilt_std)
            coupling_rdts = get_bpms_coupling_rdts(madx)
            known_errors = utils.get_table_tfs(madx, table_name="quad_errors").set_index("NAME")
        return ScenarioResult(tilt_std, coupling_rdts, known_errors)
    except:
        return 1


def gather_batches(tilt_std: float = 0.0, n_batches: int = 50) -> None:
    """
    Parallelize batches of different runs, output data into 'Outputdata' for HTCondor.

    Args:
        tilt_std (float): standard dev of the dpsi tilt distribution when applying to quadrupoles. To be
            provided throught htcondor submitter if running in CERN batch.
        n_batches (int): the number of batches to run.
    """
    # Using Joblib's threading backend as computation happens in MAD-X who releases the GIL
    # Also because cpymad itself uses theads and a multiprocessing backend would refuse that
    n_threads = int(multiprocessing.cpu_count())  # / 2)  # to ease the memory stress on HTCondor nodes

    # ----- Run simulations concurrently ----- #
    logger.info(f"Computing using Joblib's 'threading' backing, with {n_threads} threads")
    results: List[ScenarioResult] = Parallel(n_jobs=n_threads, backend="threading", verbose=10)(
        delayed(make_simulation)(tilt_std) for _ in range(n_batches)
    )
    results = [res for res in results if isinstance(res, ScenarioResult)]

    logger.info("Stacking input data to a single dimensional array")
    ml_inputs = [np.hstack(res.coupling_rdts.to_numpy()) for res in results]
    ml_outputs = [res.error_table.DPSI.to_numpy() for res in results]

    logging.info("Writing training dataset to disk")
    np.savez("Outputdata/inputs.npz", ml_inputs)
    np.savez("Outputdata/outputs.npz", ml_outputs)


# ----- Running ----- #


if __name__ == "__main__":
    with Madx(stdout=False) as mad:
        logger.critical(
            f"Using: pyhdtoolkit {pyhdtoolkit.__version__} | cpymad {cpymad.__version__}  | {mad.version}"
        )
    # gather_batches(tilt_std=%(DPSI_STD)s, n_batches=%(SEEDS)s)
