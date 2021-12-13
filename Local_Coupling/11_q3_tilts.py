"""
To be ran on the optics server or a similarly powerful machine.
This script starts a batch of simulations in which the MQSXs are powered as desired, and various
values of Q3 tilts are implemented in IR1. From these configurations coupling RDTs are calculated according
to the CMatrix approach and returned along the relevant simulation parameters.

The MQSX powerings should be realistic values: check for 2018 ATS run operationnal parameters in LSA or look
at the determined corrections from the 2021 beam tests in the OMC logbook.

Simulations run concurrently through joblib's threading backend. If using HTCondor, make sure to request enough
CPUs when increasing the number of seeds, or your jobs will run out of memory.

NOTE: This script requires pyhdtoolkit >= 0.15.1 and click >= 8.0
NOTE: This simulation script does not care for optics balancing after the Q3 tilt, only about a parameter scan
to find which (if) Q3 tilt can cancel out the coupling RDTs at the Q3 or MQSX3 locations.

Example call:
```bash
python <path/to/script.py> --q3_tilt_limit 1e-3 --n_tilts 101 --kqsx3_left 0.0011 --kqsx3_right 0.0006 --outputdir </path/to/output/dir>
```
"""
import multiprocessing
import pickle
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, List, Sequence

import click
import cpymad
import numpy as np
import pandas as pd
import pyhdtoolkit
import tfs
from cpymad.madx import Madx
from joblib import Parallel, delayed
from loguru import logger
from optics_functions.coupling import coupling_via_cmatrix
from pyhdtoolkit.cpymadtools import errors, lhc, matching, twiss
from pyhdtoolkit.utils import defaults

# ----- Setup ----- #

RUN_LOCATION = "afs" if "cern.ch" in str(Path.home()) else ("local" if "Users" in str(Path.home()) else None)
PATHS = {
    "optics2018": Path("/afs/cern.ch/eng/lhc/optics/runII/2018"),
    "local": Path.home() / "cernbox" / "OMC" / "MADX_scripts" / "Local_Coupling",
    "htc_outputdir": Path("Outputdata"),
}
TWISS_COLUMNS: List[str] = ["name", "s", "x", "y", "betx", "bety", "alfx", "alfy", "k1l", "k1sl", "r11", "r12", "r21", "r22"]

BEAM_TEST_SETTINGS: Dict[str, Dict[str, float]] = {  # determined in 2021 beam tests
    "IP1": {"kqsx3.l1": 0.0011, "kqsx3.r1": 0.0006},
    "IP2": {"kqsx3.l2": -0.0014, "kqsx3.r2": -0.0014},
    "IP5": {"kqsx3.l5": 0.0007, "kqsx3.r5": 0.0007},
    "IP8": {"kqsx3.l8": -0.0005, "kqsx3.r8": -0.0005},
}

defaults.config_logger(level="WARNING", enqueue=True)  # goes to stdout


# ----- Utilities ----- #


@dataclass
class ScenarioResult:
    q3_tilt: float
    kqsx3_left: float
    kqsx3_right: float
    coupling_rdts: tfs.TfsDataFrame


def fullpath(filepath: Path) -> str:
    """Necessary for AFS paths."""
    return str(filepath.absolute())


def call_paths(madx: Madx, opticsfile: str = "opticsfile.22") -> None:
    """Call optics from the appropriate location, either 'afs' or 'local'."""
    logger.debug("Calling optics")
    if RUN_LOCATION == "afs":
        madx.call(fullpath(PATHS["optics2018"] / "lhc_as-built.seq"))
        madx.call(fullpath(PATHS["optics2018"] / "PROTON" / opticsfile))
    elif RUN_LOCATION == "local":
        madx.call(fullpath(PATHS["local"] / "sequences" / "lhc_as-built.seq"))
        madx.call(fullpath(PATHS["local"] / "optics" / opticsfile))
    else:
        logger.error("Unknown runtime location, exiting")
        raise ValueError("The 'RUN_LOCATION' variable should be either 'afs' or 'local'.")


def get_coupling_rdts(madx: Madx) -> pd.DataFrame:
    """
    Run a TWISS for all BPMs on the currently active sequence, compute RDTs through CMatrix approach and
    return the  aggregated results. Remember that you need a TWISS call before running this function.
    """
    twiss_tfs = twiss.get_pattern_twiss(  # need chromatic flag as we're dealing with coupling
        madx, columns=TWISS_COLUMNS, chrom=True
    )
    twiss_tfs.columns = twiss_tfs.columns.str.upper()  # optics_functions needs capitalized names
    twiss_tfs.NAME = twiss_tfs.NAME.str.upper()
    twiss_tfs[["F1001", "F1010"]] = coupling_via_cmatrix(twiss_tfs, output=["rdts"])
    return twiss_tfs


# ----- Simulation ----- #


def make_simulation(
    q3_tilt: float = 0.0,
    kqsx3_left: float = 0.0,
    kqsx3_right: float = 0.0,
    opticsfile: str = "opticsfile.22",
) -> ScenarioResult:
    """
    Get a complete LHC setup, implement MQSX powering in the IR. The Q3 quadrupoles are then tilted by
    the provided amount, and the coupling RDTs are calculated throughout the machine after a TWISS call,
    through a CMatrix approach.

    Args:
        q3_tilt (float): the tilt angle (DPSI) of the Q3 quadrupoles to implement, as directly given to
            the EALIGN command.
        kqsx3_left (float): the KQSX3 value for the MQSX3.L corrector.
        kqsx3_right (float): the KQSX3 value for the MQSX3.R corrector.
        opticsfile (str): name of the optics configuration file to use. Defaults to 'opticsfile.22'.

    Returns:
        A dataclass holding the Q3 tilt value, the kqsx3 settings and the twiss + calculated coupling RDTs.
    """
    try:
        with Madx(stdout=False) as madx:
            # ----- Init ----- #
            logger.info(f"Running with a Q3 tilt of {q3_tilt:.1E}")
            madx.option(echo=False, warn=False)

            # ----- Machine Setup ----- #
            call_paths(madx, opticsfile)
            lhc.re_cycle_sequence(madx, sequence="lhcb1", start="MSIA.EXIT.B1")  # same re-cycling as model
            lhc.make_lhc_beams(madx, energy=7000, emittance=3.75e-6)
            madx.command.use(sequence="lhcb1")
            # settings from the opticsfile have us already matched
            # matching.match_tunes_and_chromaticities(madx, "lhc", "lhcb1", 62.31, 60.32, 2.0, 2.0, calls=200)

            # ----- Introduce MQSX3 Settings ----- #
            logger.debug(f"Powering MQSX3s")  # small values so we don't need coupling knobs
            with madx.batch():
                madx.globals.update({"kqsx3.l1": kqsx3_left, "kqsx3.r1": kqsx3_right})

            # ----- Q3 Tilt ----- #
            logger.debug(f"Tilting Q3s by {q3_tilt:.1E}")
            errors.misalign_lhc_ir_quadrupoles(  # calls EALIGN on Q3(RL) with DPSI=q3_tilt
                madx, ips=[1], beam=1, quadrupoles=[3], sides="RL", dpsi=f"{q3_tilt}"
            )

            # ----- Twiss and Coupling RDTs ----- #
            # madx.command.twiss(chrom=True)  # need the chromatic flag here
            coupling_rdts = get_coupling_rdts(madx)
        return ScenarioResult(q3_tilt, kqsx3_left, kqsx3_right, coupling_rdts)
    except Exception as e:  # Make sure we return something and don't crash, will filter out later
        return 1


def gather_batches(
    q3_tilts: Sequence[float],
    kqsx3_left: float = 0.0,
    kqsx3_right: float = 0.0,
) -> List[ScenarioResult]:
    """
    Parallelize different runs (different Q3 tilt values).

    Args:
        q3_tilts (Sequence[float]): the different values of Q3 tilts to run with.
        kqsx3_left (float): the KQSX3 value for the MQSX3.L corrector.
        kqsx3_right (float): the KQSX3 value for the MQSX3.R corrector.

    Returns:
        A list with all the ScenarioResult from different runs.
    """
    # Using Joblib's threading backend as computation happens in MAD-X who releases the GIL
    # Also because cpymad itself uses theads and a multiprocessing backend would refuse that
    n_threads = (
        int(multiprocessing.cpu_count() / 2) if RUN_LOCATION == "local" else int(multiprocessing.cpu_count())
    )

    logger.debug("Making partial callable func before parallelising")
    simulation = partial(make_simulation, kqsx3_left=kqsx3_left, kqsx3_right=kqsx3_right)

    # ----- Run simulations concurrently ----- #
    logger.info(f"Computing using Joblib's 'threading' backing, with {n_threads} threads")
    # Setting verbose >= 50 so that joblib streams progress to stdout and not stderr
    # Lots of output -> pipe and filter every 100 iterations with 'python ... 2>/dev/null/ | rg "00 tasks"'
    results: List[ScenarioResult] = Parallel(n_jobs=n_threads, backend="threading", verbose=50)(
        delayed(simulation)(q3_tilt) for q3_tilt in q3_tilts
    )
    logger.debug("Filtering out crashed results")  # only keep proper ScenarioResult objects
    return [res for res in results if isinstance(res, ScenarioResult)]


# ----- Running ----- #


@click.command()
@click.option(
    "--q3_tilt_limit",
    type=click.FloatRange(min=0),
    required=True,
    default=1e-5,
    show_default=True,
    help="High and low bounds of the Q3 tilt values linspace to simulate for."
    "The parameter space will be a linspace going from [-limit, limit] in --n_tilts steps.",
)
@click.option(
    "--n_tilts",
    type=click.IntRange(min=0),
    required=True,
    default=5,
    show_default=True,
    help="Number of Q3 tilt values to generate in the parameter space.",
)
@click.option(
    "--kqsx3_left",
    type=float,
    required=True,
    default=0.0,
    show_default=True,
    help="The powering value of the left MQSX corrector.",
)
@click.option(
    "--kqsx3_right",
    type=float,
    required=True,
    default=0.0,
    show_default=True,
    help="The powering value of the right MQSX corrector.",
)
@click.option(
    "--outputdir",
    type=click.Path(resolve_path=True, path_type=Path),
    required=True,
    help="Output directory in which to write the resulting data.",
)
def main(q3_tilt_limit: float, n_tilts: int, kqsx3_left: float, kqsx3_right: float, outputdir: Path) -> None:
    """
    Create a linear parameter space of Q3 tilt values from the provided limit and number of tilts in the space,
    then parallelise simulations and gather the results. The data is written to disk at the provided `outputdir`
    location.
    """
    with Madx(stdout=False) as mad:
        logger.critical(
            f"Using: pyhdtoolkit {pyhdtoolkit.__version__} | cpymad {cpymad.__version__}  | {mad.version}"
        )

    # ----- Simulations ----- #
    q3_tilts = np.linspace(-q3_tilt_limit, q3_tilt_limit, n_tilts)
    results: List[ScenarioResult] = gather_batches(q3_tilts, kqsx3_left, kqsx3_right)

    # ----- Write Results to Disk ----- #
    logger.info(f"Writing results to '{outputdir}'")
    with (outputdir / "q3_tilts_simulation_results.pkl").open("wb") as file:
        pickle.dump(results, file)


if __name__ == "__main__":
    main()
