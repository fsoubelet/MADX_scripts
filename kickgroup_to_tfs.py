"""
Utility scripts that takes the master level JSON file created by the Multiturn app, fetches the various
kick files for the given kick group, and summarizes desired data into a TFS file at the provided location.
"""
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, NewType, Union

import click
import pandas as pd
import pendulum
import tfs

# ----- Setup ----- #


class CustomFormatter(logging.Formatter):
    """Logging colored formatter, adapted from https://stackoverflow.com/a/56944256/3638629"""

    grey = "\x1b[38;21m"
    blue = "\x1b[38;5;39m"
    yellow = "\x1b[38;5;226m"
    red = "\x1b[38;5;196m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    def __init__(self, fmt):
        super().__init__()
        self.fmt = fmt
        self.FORMATS = {
            logging.DEBUG: self.grey + self.fmt + self.reset,
            logging.INFO: self.blue + self.fmt + self.reset,
            logging.WARNING: self.yellow + self.fmt + self.reset,
            logging.ERROR: self.red + self.fmt + self.reset,
            logging.CRITICAL: self.bold_red + self.fmt + self.reset,
        }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG if sys.flags.debug else logging.INFO)
stdout_handler.setFormatter(CustomFormatter("%(asctime)s | [%(levelname)8s] | %(message)s"))
LOGGER.addHandler(stdout_handler)

FilePath = NewType("FilePath", Union[Path, str])

# ----- Fetching ----- #


def get_files_from_master_json(master_json_path: FilePath) -> List[Path]:
    """
    Fetch all kick files for a given kick group, and return

    Args:
        master_json_path (FilePath): path to to the kick group JSON file output by the Multiturn app.

    Returns:
        A list of Path objects to each kick file in the group.
    """
    LOGGER.debug("Getting kick files locations from master json file")
    with Path(master_json_path).open() as file:
        master_data = json.load(file)
    kick_info_files = master_data["jsonFiles"]
    kick_info_paths = [Path(filepath) for filepath in kick_info_files]
    return kick_info_paths


def load_kick_file_to_pandas(kick_file_path: FilePath) -> pd.DataFrame:
    """
    Given a kick file, return a constructed dataframe with relevant data.

    Args:
        kick_file_path (FilePath): path to to the kick file.

    Returns:
        A pandas DataFrame with relevant data.
    """
    LOGGER.debug(f"Extracting relevant information from kick file at {kick_file_path.absolute()}")
    with Path(kick_file_path).open() as file:
        kick_data = json.load(file)

    # path is .../OP_DATA/FILL_DATA/<FILL_NUMBER>/BPM/<sddsfile>
    sdds_file: str = Path(kick_data["sddsFile"]).name
    fill_number = int(Path(kick_data["sddsFile"]).parts[-3])

    qx = float(kick_data["measurementEnvironment"]["horizontalTune"])
    qy = float(kick_data["measurementEnvironment"]["verticalTune"])

    excitation: list = kick_data["excitationSettings"]
    if excitation[0]["plane"].upper() == "HORIZONTAL":
        delta_qx = float(excitation[0]["deltaTuneEnd"])
        delta_qy = float(excitation[1]["deltaTuneEnd"])
    else:
        delta_qx = float(excitation[1]["deltaTuneEnd"])
        delta_qy = float(excitation[0]["deltaTuneEnd"])

    amp_x = float(kick_data["excitationSettings"][0]["amplitude"])  # in % of max ADC kick strength
    amp_y = float(kick_data["excitationSettings"][1]["amplitude"])  # in % of max ADC kick strength

    peak_to_peak_x = 0  # not in the json file
    peak_to_peak_y = 0  # not in the json file

    dpp = 0  # not in the json file
    coupling_knob_real = 0  # not in the json file
    coupling_knob_imag = 0  # not in the json file

    # fmt: off
    return pd.DataFrame(
        index=["FILE", "FILL", "QX", "QY", "QDX", "QDY", "AMPX", "AMPY", "PK2PKX", "PK2PKY",
               "DPP", "REAL", "IMAGINARY"],
        data=[sdds_file, fill_number, qx, qy, delta_qx, delta_qy, amp_x, amp_y,
              peak_to_peak_x, peak_to_peak_y, dpp, coupling_knob_real, coupling_knob_imag],
    ).T
    # fmt: on


def make_headers_dict_from_master_json(kick_file_path: FilePath) -> Dict[str, str]:
    """
    Construct final TFS file's headers from the first file in the kick group.

    Args:
        kick_file_path (FilePath): path to to the first kick json file.

    Returns:
        A dictionary with date, beam (LHC only), model file and optics model location.
    """
    LOGGER.debug("Creating headers for output TFS file")
    headers = {}
    with Path(kick_file_path).open() as file:
        kick_data = json.load(file)

    headers["Comment"] = "Created by kickgroup_to_tfs converter"
    headers["Date"] = pendulum.from_format(  # from custom format in these files to format desired by GUI
        kick_data["acquisitionTime"], fmt="DD-MM-YY_HH-mm-ss", tz="Europe/Paris"
    ).format("YYYY-MM-DD HH:mm:ss")
    meas_env = kick_data["measurementEnvironment"]
    headers["Beam"] = "LHCB1" if meas_env["lhcBeam"]["beamName"] == "BEAM1" else "LHCB2"
    headers["Model"] = meas_env["opticsModel"]["opticModelURI"]  # fucked by symlinks...?
    headers["Optics"] = meas_env["opticsModel"]["opticName"]
    return headers


# ----- Main Call ----- #


@click.command()
@click.argument("master_json_path", type=click.Path(exists=True, resolve_path=True, path_type=Path), nargs=1)
@click.option(
    "--output_path",
    type=click.Path(resolve_path=True, path_type=Path),
    required=True,
    help="Output path for the kick group TFS file",
)
def generate_kick_tfs(master_json_path: FilePath, output_path: FilePath) -> None:
    """
    Fetch kick files from the kick group JSON file output by Multiturn, get relevant info and write to disk
    a summary TFS file.

    Args:
        master_json_path (FilePath): path to to the kick group JSON file output by the Multiturn app.
        output_path (FilePath): location for the output TFS file.
    """
    LOGGER.info("Summarizing kick group information into a single TFS file")
    all_kick_files: List[Path] = get_files_from_master_json(master_json_path)
    all_dataframes: List[pd.DataFrame] = [load_kick_file_to_pandas(kick_file) for kick_file in all_kick_files]
    single_dframe = pd.concat(all_dataframes).reset_index(drop=True)
    headers = make_headers_dict_from_master_json(all_kick_files[0])
    output_tfs_dframe = tfs.TfsDataFrame(single_dframe, headers=headers)
    tfs.write(Path(output_path).absolute(), output_tfs_dframe, headerswidth=0)


if __name__ == "__main__":
    generate_kick_tfs()
