"""
Loads pickled pandas dataframe output by the 'ir_sources_rdt_script' with '--returns' set 
to 'pandas', and adds the wanted gaussian noise to IR and ARC BPMs. The data is later on 
saved as a compressed numpy file.

Example use:
```python
python add_bpm_noise_to_coupling.py --input <your_file> --max_ir_number=6 --ir_stdev=1e-2 --arc_stdev=5e-4
"""
import pickle

from logging import log
from pathlib import Path
from typing import List, Match

import click
import numpy as np
import pandas as pd

from loguru import logger
from rich.progress import track
from rich.traceback import install as install_rich_traceback

RNG = np.random.default_rng()
IR_BPM_REGEX = r"BPM\S?\S?\.[0-{max_index}][LR][1258]\.*"
install_rich_traceback(show_locals=False, word_wrap=True)

# ----- Utilities ----- #


def add_noise_to_ir_bpms(df: pd.DataFrame, max_index: int, stdev: float) -> None:
    """
    Selects the appropriate BPMs according to the max index provided, and add gaussian noise
    to each column with the provided standard deviation. Modifies inplace!
    """
    logger.trace(f"Adding noise to IR BPMs")
    ir_bpms = IR_BPM_REGEX.format(max_index=max_index)
    array_length = len(df[df.index.str.match(ir_bpms, case=False)])
    logger.trace(f"Number of affected BPMs: {array_length}")

    for column in df.columns:
        logger.trace(f"Adding noise to column {column}")
        df[column][df.index.str.match(ir_bpms, case=False)] += RNG.normal(0, stdev, array_length)


def add_noise_to_arc_bpms(df: pd.DataFrame, max_index: int, stdev: float) -> None:
    """
    Selects the appropriate BPMs according to the max index provided, and add gaussian noise
    to each column with the provided standard deviation. Modifies inplace!
    """
    logger.trace(f"Adding noise to arc BPMs")
    ir_bpms = IR_BPM_REGEX.format(max_index=max_index)  # regex for that max index
    array_length = len(df[~df.index.str.match(ir_bpms, case=False)])
    logger.trace(f"Number of affected BPMs: {array_length}")

    for column in df.columns:
        logger.trace(f"Adding noise to column {column}")
        df[column][~df.index.str.match(ir_bpms, case=False)] += RNG.normal(0, stdev, array_length)


# ----- Concatenation Helpers ----- #


def _stack_rdts_df_to_numpy_array(df: pd.DataFrame) -> np.ndarray:
    """
    Converts the coupling RDTs dataframe from the `ScenarioResult` with
    RDTs components columns to a numpy array. The output is a one-dimentional
    ndarray with all the BPM values for F1001REAL, then all for F1001IMAG,
    then all F1010REAL and finally all for F1010IMAG.
    """
    return np.concatenate(
        (df.F1001REAL.to_numpy(), df.F1001IMAG.to_numpy(), df.F1010REAL.to_numpy(), df.F1010IMAG.to_numpy())
    )


# ----- Running ----- #


@click.command()
@click.option(
    "--input",
    type=click.Path(resolve_path=True, path_type=Path),
    required=True,
    help="Path to the input file to load.",
)
@click.option(
    "--max_ir_number",
    type=click.IntRange(min=0),
    default=5,
    required=True,
    show_default=True,
    help="Max element index for a BPM to be considered part of the IR setup",
)
@click.option(
    "--ir_stdev",
    type=click.FloatRange(min=0),
    required=True,
    default=0,
    show_default=True,
    help="Standard dev of the gaussian noise distribution to add to IR BPMs",
)
@click.option(
    "--arc_stdev",
    type=click.FloatRange(min=0),
    required=True,
    default=0,
    show_default=True,
    help="Standard dev of the gaussian noise distribution to add to arc BPMs",
)
def main(input: Path, max_ir_number: int, ir_stdev: float, arc_stdev: float) -> None:
    """
    Load the input file, apply the wanted noise to IR and arc BPMs in all dataframes, then stack them as a numpy array
    and save to disk with a new extension.
    """
    logger.info(f"Loading pickled dataframes from '{input}'")
    with input.open("rb") as file:
        data: List[pd.DataFrame] = pickle.load(file)

    logger.info(f"Adding noise to BPMs")
    for dataframe in track(data, description="Adding noise to BPMs"):
        add_noise_to_ir_bpms(dataframe, max_ir_number, ir_stdev)
        add_noise_to_arc_bpms(dataframe, max_ir_number, arc_stdev)

    output_file = input.with_stem(input.stem + "_noised").with_suffix(".npz")  # type: Path
    logger.info(f"Stacking dataframes and exporting to '{output_file}'")
    data_stacked = [_stack_rdts_df_to_numpy_array(df) for df in data]
    np.savez(output_file, data_stacked)


if __name__ == "__main__":
    main()
