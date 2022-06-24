"""
This script starts a batch of simulations of given errors that introduce coupling, and returns the true
errors introduced as well as the calculated coupling RDTs from 'optics_functions'.

For this script, the errors are distributed with a prodived standard deviation according to the MAD-X
'value * TGAUSS(2.5)' command, with the standard deviation value being provided at the commandline (or by the
htcondor_submitter). Errors are distributed to all IR quadrupoles for IRs 1, 2, 5 and 8 (the ones with IP
points).

Seeds run concurrently through joblib's threading backend. If using HTCondor, make sure to request enough
CPUs when increasing the number of seeds, or your jobs will run out of memory.

NOTE: this script requires pyhdtoolkit >= 0.17.0 and click >= 8.0

The lists of np.ndarrays are saved in .npz format, and can be loaded back with:
```python
with np.load("inputs.npz") as data:
    ml_inputs = list(data.values())[0]
```
"""
import multiprocessing
import pickle
import time

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union
from uuid import uuid4

import click
import numpy as np
import pandas as pd
import tfs

from cpymad.madx import Madx
from joblib import Parallel, delayed
from loguru import logger
from optics_functions.coupling import coupling_via_cmatrix
from rich.traceback import install as install_rich_traceback

from pyhdtoolkit.cpymadtools import errors, lhc, matching, twiss, utils
from pyhdtoolkit.cpymadtools.constants import MONITOR_TWISS_COLUMNS
from pyhdtoolkit.utils import defaults
from pyhdtoolkit.utils._misc import call_lhc_sequence_and_optics, log_runtime_versions

# ----- Setup ----- #

defaults.config_logger(level="WARNING", enqueue=True)  # goes to stdout
install_rich_traceback(show_locals=False)

# ----- Utilities ----- #


@dataclass
class ScenarioResult:
    tilt: float
    coupling_rdts: tfs.TfsDataFrame
    error_table: tfs.TfsDataFrame


def get_bpms_coupling_rdts(madx: Madx) -> pd.DataFrame:
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


# ----- Simulation ----- #


def do_beam_1(
    tilt_std: float = 0.0,
    quadrupoles: List[int] = list(range(1, 11)),
    opticsfile: str = "opticsfile.22",
    temp_file: str = None,
) -> Tuple[tfs.TfsDataFrame, tfs.TfsDataFrame]:
    with Madx(stdout=False) as madxb1:
        # ----- Init ----- #
        logger.info(f"Running B1 with a mean tilt of {tilt_std:.1E}")
        madxb1.option(echo=False, warn=False)
        madxb1.option(rand="best", randid=np.random.randint(1, 11))  # random number generator
        madxb1.eoption(seed=np.random.randint(1, 999999999))  # not using default seed

        # ----- Machine Setup ----- #
        call_lhc_sequence_and_optics(madxb1, opticsfile=opticsfile)
        lhc.re_cycle_sequence(madxb1, sequence="lhcb1", start="MSIA.EXIT.B1")
        lhc.make_lhc_beams(madxb1, energy=7000, emittance=3.75e-6)
        madxb1.command.use(sequence="lhcb1")
        # Tunes are matched from opticsfile and no modification was made so no need for next line
        # matching.match_tunes_and_chromaticities(madx, "lhc", "lhcb1", 62.31, 60.32, 2.0, 2.0, calls=200)

        # ----- Introduce Errors, Twiss and RDTs ----- #
        errors.misalign_lhc_ir_quadrupoles(
            madxb1,
            ips=[1, 2, 5, 8],
            beam=1,
            quadrupoles=quadrupoles,
            sides="RL",
            dpsi=f"{tilt_std} * TGAUSS(2.5)",
            table="ir_quads_errors_b1",
        )
        coupling_rdts_b1 = get_bpms_coupling_rdts(madxb1)
        known_errors_b1 = utils.get_table_tfs(madxb1, table_name="ir_quads_errors_b1").set_index("NAME")
        madxb1.command.select(flag="ERROR", pattern="^MQX[AB]\..*")  # common triplets, regex understood by MAD-X
        madxb1.command.esave(file=temp_file)
    return coupling_rdts_b1, known_errors_b1


def do_beam_2(
    tilt_std: float = 0.0,
    quadrupoles: List[int] = list(range(1, 11)),
    opticsfile: str = "opticsfile.22",
    temp_file: str = None,
) -> Tuple[tfs.TfsDataFrame, tfs.TfsDataFrame]:
    b2_quadrupoles = [num for num in quadrupoles if num not in (1, 2, 3)]  # keep same but without triplets

    with Madx(stdout=False) as madxb2:
        # ----- Init ----- #
        logger.info(f"Running B2 with a mean tilt of {tilt_std:.1E}")
        madxb2.option(echo=False, warn=False)
        madxb2.option(rand="best", randid=np.random.randint(1, 11))  # random number generator
        madxb2.eoption(seed=np.random.randint(1, 999999999))  # not using default seed

        # ----- Machine Setup ----- #
        call_lhc_sequence_and_optics(madxb2, opticsfile=opticsfile)
        lhc.re_cycle_sequence(madxb2, sequence="lhcb2", start="MSIA.EXIT.B2")
        lhc.make_lhc_beams(madxb2, energy=7000, emittance=3.75e-6)
        madxb2.command.use(sequence="lhcb2")
        # Tunes are matched from opticsfile and no modification was made so no need for next line
        # matching.match_tunes_and_chromaticities(madx, "lhc", "lhcb2, 62.31, 60.32, 2.0, 2.0, calls=200)

        # ----- Introduce Errors to Common Magnets ----- #
        # Only done if one of Q1, Q2 or Q3 is in the quadrupoles list, otherwise the error
        # file will not havebeen exported from the beam 1 simulation and this would fail
        if any(x in quadrupoles for x in (1, 2, 3)):
            madxb2.command.readtable(file=temp_file, table="common_errors")
            madxb2.command.seterr(table="common_errors")

        # ----- Introduce Errors to Other IR Quadrupoles, Twiss and RDTs ----- #
        errors.misalign_lhc_ir_quadrupoles(
            madxb2,
            ips=[1, 2, 5, 8],
            beam=2,
            quadrupoles=b2_quadrupoles,
            sides="RL",
            dpsi=f"{tilt_std} * TGAUSS(2.5)",
            table="ir_quads_errors_b2",
        )
        coupling_rdts_b2 = get_bpms_coupling_rdts(madxb2)
        known_errors_b2 = utils.get_table_tfs(madxb2, table_name="ir_quads_errors_b2").set_index("NAME")  # no triplets
    return coupling_rdts_b2, known_errors_b2


def make_simulation(
    tilt_std: float = 0.0,
    quadrupoles: List[int] = list(range(1, 11)),
    opticsfile: str = "opticsfile.22",
) -> ScenarioResult:
    """
    Get a complete LHC setup, implement coupling sources as tilt errors in the desired IR quadrupoles. The
    coupling RDTs are calculated from a Twiss call at monitor elements throughout the machine, through a
    CMatrix approach.

    Args:
        tilt_std (float): standard dev of the dpsi tilt distribution when applying to quadrupoles. To be
            provided throught the command line arguments.
        quadrupoles (List[int]) the list of quadrupoles to apply errors to. Defaults to all IR quads (applied
            on both sides of IP), to be provided throught the command line arguments.
        opticsfile (str): name of the optics configuration file to use. Defaults to **opticsfile.22**.

    Returns:
        A custom dataclass holding both the twiss result including coupling RDTs and the assigned errors
        table. The coupling RDTs are for all errors of B1 then B2 in a single dataframe. The errors table
        is for all errors of B1, then all errors of B2 without the triplets (common to both beams and already
        present in the B1 information) in a single dataframe.
    """
    temp_file = str(uuid4()) + ".tfs"
    try:
        rdts_b1, errors_b1 = do_beam_1(
            tilt_std=tilt_std, quadrupoles=quadrupoles, opticsfile=opticsfile, temp_file=temp_file
        )
        time.sleep(0.5)
        rdts_b2, errors_b2 = do_beam_2(
            tilt_std=tilt_std, quadrupoles=quadrupoles, opticsfile=opticsfile, temp_file=temp_file
        )
        Path(temp_file).unlink(missing_ok=True)

        coupling_rdts = pd.concat([rdts_b1, rdts_b2])
        known_errors = pd.concat([errors_b1, errors_b2])
        assert not coupling_rdts.index.has_duplicates
        assert not known_errors.index.has_duplicates
        result = ScenarioResult(tilt=tilt_std, coupling_rdts=coupling_rdts, error_table=known_errors)
    except:
        result = 1  # will be used for filtering later
        Path(temp_file).unlink(missing_ok=False)
    return result


def gather_batches(tilt_std: float = 0.0, n_batches: int = 50) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    """
    Parallelize batches of different simulation runs.

    Args:
        tilt_std (float): standard dev of the dpsi tilt distribution when applying to quadrupoles.
            To be provided throught the command line.
        n_batches (int): the number of batches to run. Defaults to 50.

    Returns:
        A `list` of `pd.DataFrame` objects with the coupling RDTs columns for inputs and a `list`
        of `pd.DataFrame` objects with error table columns for outputs.
    """
    # Using Joblib's threading backend as computation happens in MAD-X who releases the GIL
    # Also because cpymad itself uses theads and a multiprocessing backend would refuse that
    n_threads = int(multiprocessing.cpu_count()) - 2

    # ----- Run simulations concurrently ----- #
    logger.info(f"Computing using Joblib's 'threading' backing, with {n_threads} threads")
    # Setting verbose >= 50 so that joblib streams progress to stdout and not stderr
    # Lots of output -> pipe and filter every 100 iterations with 'python ... 2>/dev/null/ | rg "00 tasks"'
    results: List[ScenarioResult] = Parallel(n_jobs=n_threads, backend="threading", verbose=50)(
        delayed(make_simulation)(tilt_std) for _ in range(n_batches)
    )
    results: List[ScenarioResult] = [res for res in results if isinstance(res, ScenarioResult)]
    inputs: List[tfs.TfsDataFrame] = [res.coupling_rdts for res in results]
    outputs: List[tfs.TfsDataFrame] = [res.error_table for res in results]
    return inputs, outputs


# ----- Concatenation Helpers ----- #


def _stack_result_rdts_df_to_numpy_array(result: ScenarioResult) -> np.ndarray:
    """
    Converts the coupling RDTs dataframe from the `ScenarioResult` with
    RDTs components columns to a numpy array. The output is a one-dimentional
    ndarray with all the BPM values for F1001REAL, then all for F1001IMAG,
    then all F1010REAL and finally all for F1010IMAG.
    """
    df = result.coupling_rdts
    return np.concatenate(
        (df.F1001REAL.to_numpy(), df.F1001IMAG.to_numpy(), df.F1010REAL.to_numpy(), df.F1010IMAG.to_numpy())
    )


def _stack_dpsi_errors_to_numpy_array(result: ScenarioResult) -> np.ndarray:
    """
    Converts the tilt errors dataframe from the `ScenarioResult` to a numpy
    array, only selecting the DPSI column. The array corresponds to the DPSI
    value at each affected quadrupole.
    """
    return result.error_table.DPSI.to_numpy()


# ----- Running ----- #


@click.command()
@click.option(
    "--tilt_std",
    type=click.FloatRange(min=0),
    required=True,
    default=0,
    show_default=True,
    help="Standard dev of the dpsi tilt distribution applied to IR quadrupoles",
)
@click.option(
    "--n_batches",
    type=click.IntRange(min=0),
    required=True,
    default=50,
    show_default=True,
    help="Number of simulations to run to generate the data",
)
@click.option(
    "--outputdir",
    type=click.Path(resolve_path=True, path_type=Path),
    required=True,
    help="Output directory in which to write the training data files.",
)
@click.option(
    "--returns",
    type=click.Choice(["numpy", "pandas", "both"], case_sensitive=False),
    default="both",
    show_default=True,
    help="The format in which to return the data. Can be 'numpy' (results in .npz files), 'pandas' "
    "(results in pickled dataframes) or 'both'.",
)
def main(tilt_std: float, n_batches: int, outputdir: Path, returns: str) -> None:
    """
    Run 'n_batches' simulations and gather all data to create a training set, output at the desired
    location.
    """
    assert returns in ("numpy", "pandas", "both"), "Invalid value for 'returns' option."
    log_runtime_versions()

    # Each here is a list of dataframes, one for each simulation run
    ml_inputs, ml_outputs = gather_batches(tilt_std=tilt_std, n_batches=n_batches)

    if returns in ("pandas", "both"):
        with (outputdir / f"{n_batches:d}_sims_ir_sources_inputs.pkl").open("wb") as file:
            pickle.dump(ml_inputs, file)
        with (outputdir / f"{n_batches:d}_sims_ir_sources_outputs.pkl").open("wb") as file:
            pickle.dump(ml_outputs, file)
    if returns in ("numpy", "both"):
        ml_inputs: List[np.ndarray] = [_stack_result_rdts_df_to_numpy_array(res) for res in ml_inputs]
        ml_outputs: List[np.ndarray] = [_stack_dpsi_errors_to_numpy_array(res) for res in ml_outputs]
        np.savez(outputdir / f"{n_batches:d}_sims_ir_sources_inputs.npz", ml_inputs)
        np.savez(outputdir / f"{n_batches:d}_sims_ir_sources_outputs.npz", ml_outputs)


if __name__ == "__main__":
    main()
