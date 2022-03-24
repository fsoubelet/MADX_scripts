"""
This script starts a batch of simulations of given errors that introduce coupling, and returns the true
errors introduced as well as the calculated coupling RDTs from 'optics_functions'.

For this script, the errors are distributed with a prodived standard deviation according to the MAD-X
'value * TGAUSS(2.5)' command, with the standard deviation value being provided at the commandline (or by the
htcondor_submitter). Errors are distributed to all quadrupoles.

Seeds run concurrently through joblib's threading backend. If using HTCondor, make sure to request enough
CPUs when increasing the number of seeds, or your jobs will run out of memory.

NOTE: this script requires pyhdtoolkit >= 0.15.1 and click >= 8.0

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
from typing import List, Tuple

import click
import numpy as np
import pandas as pd
import tfs

from cpymad.madx import Madx
from joblib import Parallel, delayed
from loguru import logger
from optics_functions.coupling import coupling_via_cmatrix

from pyhdtoolkit.cpymadtools import lhc, twiss, utils
from pyhdtoolkit.cpymadtools.constants import MONITOR_TWISS_COLUMNS
from pyhdtoolkit.utils import defaults
from pyhdtoolkit.utils._misc import call_lhc_sequence_and_optics, log_runtime_versions

# ----- Setup ----- #

defaults.config_logger(level="WARNING", enqueue=True)  # goes to stdout

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


def apply_dpsi_to_arc_quads(madx: Madx, tilt_std: float = 0.0) -> None:
    """Apply a the provided stdev as a TGAUSS(2.5) to all LHC quads in the machine."""
    madx.command.select(flag="error", clear=True)
    madx.command.select(flag="error", pattern=r"^MQ\.")
    madx.command.ealign(dpsi=f"{tilt_std} * TGAUSS(2.5)")
    madx.command.etable(table="quad_errors")  # save errors in table
    madx.select(flag="error", clear=True)  # cleanup the flag


# ----- Simulation ----- #


def do_beam_1(tilt_std: float = 0.0, opticsfile: str = "opticsfile.22") -> Tuple[tfs.TfsDataFrame, tfs.TfsDataFrame]:
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

        # # ----- Introduce Errors, Twiss and RDTs ----- #
        logger.debug(f"Introducing tilts in arc quadrupoles")
        apply_dpsi_to_arc_quads(madxb1, tilt_std=tilt_std)
        coupling_rdts_b1 = get_bpms_coupling_rdts(madxb1)
        known_errors_b1 = utils.get_table_tfs(madxb1, table_name="quad_errors").set_index("NAME")
    return coupling_rdts_b1, known_errors_b1


def do_beam_2(tilt_std: float = 0.0, opticsfile: str = "opticsfile.22") -> Tuple[tfs.TfsDataFrame, tfs.TfsDataFrame]:
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
        # matching.match_tunes_and_chromaticities(madx, "lhc", "lhcb2", 62.31, 60.32, 2.0, 2.0, calls=200)

        # # ----- Introduce Errors, Twiss and RDTs ----- #
        logger.debug(f"Introducing tilts in arc quadrupoles")
        apply_dpsi_to_arc_quads(madxb2, tilt_std=tilt_std)
        coupling_rdts_b2 = get_bpms_coupling_rdts(madxb2)
        known_errors_b2 = utils.get_table_tfs(madxb2, table_name="quad_errors").set_index("NAME")
    return coupling_rdts_b2, known_errors_b2


def make_simulation(tilt_std: float = 0.0, opticsfile: str = "opticsfile.22") -> ScenarioResult:
    """
    Get a complete LHC setup, implement coupling sources as tilt errors in the arc IR quadrupoles. The
    coupling RDTs are calculated from a Twiss call at monitor elements throughout the machine, through a
    CMatrix approach.

    Args:
        tilt_std (float): standard dev of the dpsi tilt distribution when applying to quadrupoles. To be
            provided throught the command line arguments.
        opticsfile (str): name of the optics configuration file to use. Defaults to **opticsfile.22**.

    Returns:
        A custom dataclass holding both the twiss result including coupling RDTs and the assigned errors
        table. The coupling RDTs are for all errors of B1 then B2 in a single dataframe. The errors table
        is for all errors of B1, then all errors of B2 without the triplets (common to both beams and already
        present in the B1 information) in a single dataframe.
    """
    try:
        rdts_b1, errors_b1 = do_beam_1(tilt_std=tilt_std, opticsfile=opticsfile)
        time.sleep(0.5)
        rdts_b2, errors_b2 = do_beam_2(tilt_std=tilt_std, opticsfile=opticsfile)

        coupling_rdts = pd.concat([rdts_b1, rdts_b2])
        known_errors = pd.concat([errors_b1, errors_b2])
        assert not coupling_rdts.index.has_duplicates
        assert not known_errors.index.has_duplicates
        result = ScenarioResult(tilt=tilt_std, coupling_rdts=coupling_rdts, error_table=known_errors)
    except:
        result = 1  # will be used for filtering later
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
    results = [res for res in results if isinstance(res, ScenarioResult)]
    inputs = [res.coupling_rdts for res in results]
    outputs = [res.error_table for res in results]
    return inputs, outputs


# ----- Running ----- #


@click.command()
@click.option(
    "--tilt_std",
    type=click.FloatRange(min=0),
    required=True,
    default=0,
    show_default=True,
    help="Standard dev of the dpsi tilt distribution applied to arc quadrupoles",
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
    type=click.Path(resolve_path=True, path_type=Path, file_okay=False, writable=True),
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

    ml_inputs, ml_outputs = gather_batches(tilt_std=tilt_std, n_batches=n_batches)

    if returns in ("pandas", "both"):
        with (outputdir / f"{n_batches:d}_sims_arc_sources_inputs.pkl").open("wb") as file:
            pickle.dump(ml_inputs, file)
        with (outputdir / f"{n_batches:d}_sims_arc_sources_outputs.pkl").open("wb") as file:
            pickle.dump(ml_outputs, file)
    if returns in ("numpy", "both"):
        ml_inputs = [np.hstack(res.to_numpy()) for res in ml_inputs]
        ml_outputs = [res.DPSI.to_numpy() for res in ml_outputs]
        np.savez(outputdir / f"{n_batches:d}_sims_arc_sources_inputs.npz", ml_inputs)
        np.savez(outputdir / f"{n_batches:d}_sims_arc_sources_outputs.npz", ml_outputs)


if __name__ == "__main__":
    main()
