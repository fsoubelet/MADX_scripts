"""
This script starts a batch of seeds for simulations of DPSI in Q1 to Q6 quads around an IP, and
asks MAD-X to attempt coupling correction at IP through the Ripken parameters beta12 and beta21.

For this script, the errors are distributed with a 'std * TGAUSS(2.5)' command, with the standard
deviation being provided at the commandline. This means all errors will be distributed around 0
according to a truncated gaussian distribution with the stdev given at the commandline.

Seeds run concurrently through joblib's threading backend. Make sure to request enough CPUs on
HTCondor when increasing the number of seeds, or your jobs will run out of memory.
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
from cpymad.madx import Madx, TwissFailed
from joblib import Parallel, delayed
from loguru import logger
from pydantic import BaseModel
from pyhdtoolkit.cpymadtools import errors, matching, orbit, special
from pyhdtoolkit.cpymadtools.constants import DEFAULT_TWISS_COLUMNS
from pyhdtoolkit.utils import defaults

# ----- Setup ----- #

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


class Results(BaseModel):
    tilt_mean: float
    tilt_std: float
    r11_value: float
    r11_std: float
    r12_value: float
    r12_std: float
    r21_value: float
    r21_std: float
    r22_value: float
    r22_std: float
    beta12_value: float
    beta12_std: float
    beta21_value: float
    beta21_std: float
    kqsx3_l1_value: float
    kqsx3_l1_std: float
    kqsx3_r1_value: float
    kqsx3_r1_std: float

    def to_json(self, filepath: Union[str, Path]) -> None:
        logger.debug(f"Exporting results structure to '{Path(filepath).absolute()}'")
        with Path(filepath).open("w") as results_file:
            json.dump(self.dict(), results_file)

    @classmethod
    def from_json(cls, json_file: Union[str, Path]):
        """
        Load a Result instance's data from disk, saved in the JSON format.
        Args:
            json_file (Union[Path, str]): PosixPath object or string with the save file location.
        """
        logger.info(f"Loading JSON data from file at '{Path(json_file).absolute()}'")
        return cls.parse_file(json_file, content_type="application/json")


def fullpath(filepath: Path) -> str:
    return str(filepath.absolute())


# ----- Simulation ----- #


def make_simulation(
    tilt_stdev: float = 0.0, quadrupoles=None, tolerance: float = 1e-4
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get a complete LHC setup, implement coupling at IP with tilt errors and attempt correction by
    matching cross-term Ripken parameters to 0 at IP using skew quadrupole correctors (MQSX3).
    Both simulation parameters the results are stored in a structure and exported to JSON.

    Args:
        tilt_stdev (float): mean value of the dpsi tilt distribution when applying to quadrupoles.
            To be provided throught htcondor submitter if running in CERN batch.
        quadrupoles (List[int]) the list of quadrupoles to apply errors to. Defaults to Q1 to Q6,
            to be provided throught htcondor submitter if running in CERN batch.
        tolerance (float): value above which to consider MAD-X has messed up when matching and
            restart the simulation. Defaults to 1e-4.

    Returns:
        A tuple of two relevant dataframes, the first one with the assigned errors and the second
        one with the twiss result after correction.
    """
    quadrupoles = [1, 2, 3, 4, 5, 6] if quadrupoles is None else quadrupoles
    beta12, beta21 = 1, 1  # for initial checks

    while beta12 > tolerance or beta21 > tolerance:
        with Madx(command_log=fullpath(PATHS["htc_outputdir"] / "cpymad_commands.log")) as madx:
            # ----- Init ----- #
            logger.info(f"Running with a tilt stdev of {tilt_stdev:.1E}")
            madx.option(echo=False, warn=False)
            madx.option(rand="best", randid=np.random.randint(1, 11))  # random number generator
            madx.eoption(seed=np.random.randint(1, 999999999))  # not using default seed

            # ----- Machine ----- #
            logger.debug("Calling optics")
            # madx.call(fullpath(PATHS["optics2018"] / "lhc_as-built.seq"))  # afs
            # madx.call(fullpath(PATHS["optics2018"] / "PROTON" / "opticsfile.22"))  # afs
            # madx.call(fullpath(PATHS["local"] / "sequences" / "lhc_as-built.seq"))  # local
            # madx.call(fullpath(PATHS["local"] / "optics" / "opticsfile.22"))  # local

            # ----- Setup ----- #
            special.re_cycle_sequence(madx, sequence="lhcb1", start="IP3")
            _ = orbit.setup_lhc_orbit(madx, scheme="flat")
            special.make_lhc_beams(madx, energy=6500, emittance=3.75e-6)
            madx.command.use(sequence="lhcb1")
            matching.match_tunes_and_chromaticities(madx, "lhc", "lhcb1", 62.31, 60.32, 2.0, 2.0, calls=200)

            # ----- Errors ----- #
            logger.info(f"Applying misalignments to IR quads {quadrupoles[0]} to {quadrupoles[-1]}")
            errors.misalign_lhc_ir_quadrupoles(  # requires pyhdtoolkit >= 0.9.0
                madx,
                ip=1,
                beam=1,
                quadrupoles=quadrupoles,
                sides="RL",
                dpsi=f"{tilt_stdev} * TGAUSS(2.5)",
            )
            tilt_errors = (  # just get the dpsi column, save memory
                madx.table.ir_quads_errors.dframe().set_index("name", drop=True).loc[:, ["dpsi"]]
            )

            # ----- Correction ----- #
            special.match_no_coupling_through_ripkens(  # requires pyhdtoolkit >= 0.9.2
                madx, sequence="lhcb1", location="IP1", vary_knobs=["KQSX3.R1", "KQSX3.L1"]
            )
            try:
                twiss_df = madx.twiss(ripken=True).dframe().copy()
                twiss_df["k1s"] = twiss_df.k1sl / twiss_df.l
                twiss_df = twiss_df.loc[:, DEFAULT_TWISS_COLUMNS + ["k1s"]]  # save memory
                twiss_df = twiss_df.set_index("name", drop=True)
                beta12, beta21 = twiss_df.beta12["ip1"], twiss_df.beta21["ip1"]
            except TwissFailed:  # MAD-X giga-fucked internally
                beta12, beta21 = 1, 1  # force these values so we restard the simulation
    return tilt_errors, twiss_df


def gather_simulated_seeds(tilt_stdev: float = 0.0, quadrupoles=None, seeds: int = 50) -> Results:
    """
    Simulate a setup through many different seeds, aggregating the results and exporting the final
    structure. Parameters and the results are stored in a structure and exported to JSON.

    Args:
        tilt_stdev (float): mean value of the dpsi tilt distribution when applying to quadrupoles.
            To be provided throught htcondor submitter if running in CERN batch.
        quadrupoles (List[int]) the list of quadrupoles to apply errors to. Defaults to Q1 to A6,
            to be provided throught htcondor submitter if running in CERN batch.
        seeds (int): the number of runs with different seeds to do for each tilt value.
    """
    # Using Joblib's threading backend as computation happens in MAD-X who releases the GIL
    # Also because cpymad itself uses theads and a multiprocessing backend would refuse that
    quadrupoles = [1, 2, 3, 4, 5, 6] if quadrupoles is None else quadrupoles
    n_threads = int(multiprocessing.cpu_count() / 2)  # to ease the memory stress on HTCondor nodes

    # ----- Run simulations concurrently ----- #
    logger.info(f"Computing using Joblib's 'threading' backing, with {n_threads} threads")
    tilt_errors, corrected_twisses = zip(
        *Parallel(n_jobs=n_threads, backend="threading")(
            delayed(make_simulation)(tilt_stdev, quadrupoles) for _ in range(seeds)
        )
    )

    # ----- Aggregate ----- #
    logger.info("Aggregating results from all seeds")
    all_errors = pd.concat(tilt_errors)  # concatenating all errors for this tilt's runs
    all_results = pd.concat(corrected_twisses)  # concatenating all twisses for this tilt's runs

    return Results(
        tilt_mean=tilt_stdev,
        tilt_std=all_errors.dpsi.std(),
        r11_value=all_results.r11.mean(),
        r11_std=all_results.r11.std(),
        r12_value=all_results.r12.mean(),
        r12_std=all_results.r12.std(),
        r21_value=all_results.r21.mean(),
        r21_std=all_results.r21.std(),
        r22_value=all_results.r22.mean(),
        r22_std=all_results.r22.std(),
        beta12_value=all_results.beta12.mean(),
        beta12_std=all_results.beta12.std(),
        beta21_value=all_results.beta21.mean(),
        beta21_std=all_results.beta21.std(),
        kqsx3_l1_value=all_results.loc[all_results.index == "mqsx.3l1:1"].k1s.mean(),
        kqsx3_l1_std=all_results.loc[all_results.index == "mqsx.3l1:1"].k1s.std(),
        kqsx3_r1_value=all_results.loc[all_results.index == "mqsx.3r1:1"].k1s.mean(),
        kqsx3_r1_std=all_results.loc[all_results.index == "mqsx.3r1:1"].k1s.std(),
    )


# ----- Running ----- #


if __name__ == "__main__":
    with Madx(stdout=False) as mad:
        logger.critical(
            f"Using: pyhdtoolkit {pyhdtoolkit.__version__} | cpymad {cpymad.__version__}  | {mad.version}"
        )
    # simulation_results = gather_simulated_seeds(  # afs run
    #     tilt_stdev=%(DPSI_MEAN)s,
    #     quadrupoles=[1, 2, 3, 4, 5, 6],
    #     seeds=%(SEEDS)s,
    # )
    # simulation_results = gather_simulated_seeds(  # local testing
    #     tilt_mean=5e-4,
    #     quadrupoles=[1, 2, 3, 4, 5, 6],
    #     seeds=50,
    # )
    simulation_results.to_json(PATHS["htc_outputdir"] / "results.json")
