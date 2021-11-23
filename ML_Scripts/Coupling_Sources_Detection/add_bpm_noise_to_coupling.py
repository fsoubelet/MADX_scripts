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

RNG = np.random.default_rng()
IR_BPM_REGEX = r"BPM\S?\S?\.[0-{max_index}][LR][1258]\.*"

# ----- Utilities ----- #


def add_noise_to_ir_bpms(df: pd.DataFrame, max_index: int, stdev: float) -> None:
    """
    Selects the appropriate BPMs according to the max index provided, and add gaussian noise
    to each column with the provided standard deviation. Modifies inplace!
    """
    logger.debug(f"Adding noise to IR BPMs")
    ir_bpms = IR_BPM_REGEX.format(max_index=max_index)
    array_length = len(df[df.index.str.match(ir_bpms, case=False)])
    logger.debug(f"Number of affected BPMs: {array_length}")

    for column in df.columns:
        logger.trace(f"Adding noise to column {column}")
        df[column][df.index.str.match(ir_bpms, case=False)] += RNG.normal(0, stdev, array_length)


def add_noise_to_arc_bpms(df: pd.DataFrame, max_index: int, stdev: float) -> None:
    """
    Selects the appropriate BPMs according to the max index provided, and add gaussian noise
    to each column with the provided standard deviation. Modifies inplace!
    """
    logger.debug(f"Adding noise to arc BPMs")
    ir_bpms = IR_BPM_REGEX.format(max_index=max_index)  # regex for that max index
    array_length = len(df[~df.index.str.match(ir_bpms, case=False)])
    logger.debug(f"Number of affected BPMs: {array_length}")

    for column in df.columns:
        logger.trace(f"Adding noise to column {column}")
        df[column][~df.index.str.match(ir_bpms, case=False)] += RNG.normal(0, stdev, array_length)


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
    for dataframe in data:
        add_noise_to_ir_bpms(dataframe, max_ir_number, ir_stdev)
        add_noise_to_arc_bpms(dataframe, max_ir_number, arc_stdev)

    logger.info(f"Stacking dataframes and exporting")
    data_stacked = [np.hstack(df.to_numpy()) for df in data]
    output_file = input.with_name(input.name + "_noised").with_suffix(".npy")
    np.savez(output_file, data_stacked)


if __name__ == "__main__":
    main()
