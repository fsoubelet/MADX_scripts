import json
import sys
from pathlib import Path
from typing import List, Union

import cpymad
import numpy as np
import pyhdtoolkit
from cpymad.madx import Madx
from loguru import logger
from pydantic import BaseModel
from pyhdtoolkit.cpymadtools import latwiss, matching, orbit, special
from pyhdtoolkit.utils.defaults import LOGURU_FORMAT

# ----- Setup ----- #

PATHS = {
    "optics2018": Path("/afs/cern.ch/eng/lhc/optics/runII/2018"),
    "local": Path("/Users/felixsoubelet/cernbox/OMC/MADX_scripts/Local_Coupling"),
    "htc_outputdir": Path("Outputdata"),
}

logger.remove()
logger.add(sys.stderr, format=LOGURU_FORMAT, level="DEBUG")
logger.add(PATHS["htc_outputdir"] / "full_pylog.log", format=LOGURU_FORMAT, level="TRACE")

# ----- Utilities ----- #


class Results(BaseModel):
    coupling_knob: float
    dqmin_after_waist: float
    mean_betabeat_x: float
    stdev_betabeat_x: float
    mean_betabeat_y: float
    stdev_betabeat_y: float

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


def cycle_from_ip3(madx: Madx, sequence: str = "lhcb1") -> None:
    """Input a few commands to cycle the provided sequence from IP3."""
    logger.debug(f"Re-cycling sequence '{sequence}' from IP3")
    madx.command.seqedit(sequence=sequence)
    madx.command.flatten()
    madx.command.cycle(start="IP3")
    madx.command.endedit()


def fullpath(filepath: Path) -> str:
    return str(filepath.absolute())


def abs_betabeat(betas_no_waist: np.ndarray, betas_with_waist: np.ndarray) -> np.ndarray:
    """Get the betabeat from two arrays"""
    # take np.abs here because we do a mean afterwards and it will be skewed by negative values
    betabeat: np.ndarray = np.abs((betas_with_waist - betas_no_waist) / betas_no_waist)
    return betabeat


# ----- Simulation ----- #


def make_simulation(coupling_knob: float = 0.0) -> Results:
    """
    Get a complete LHC setup, implement colinearity knob and rigidity waist shift knob, get the Cminus.
    Both parameters and the results are stored in a structure and exported to JSON.

    Args:
        coupling_knob (float): value of the coupling knob to apply.
    """
    madx = Madx(command_log=fullpath(PATHS["htc_outputdir"] / "cpymad_commands.log"))
    madx.option(echo=False, warn=False)
    logger.info(f"Running with a coupling knob of {coupling_knob}")

    # Working point
    tune_x, tune_y = 62.31, 60.32
    chrom_x, chrom_y = 2.0, 2.0

    logger.info("Calling optics")
    madx.call(fullpath(PATHS["optics2018"] / "lhc_as-built.seq"))  # afs
    madx.call(fullpath(PATHS["optics2018"] / "PROTON" / "opticsfile.22"))  # afs
    # madx.call(fullpath(PATHS["local"] / "sequences" / "lhc_as-built.seq"))  # local testing
    # madx.call(fullpath(PATHS["local"] / "optics" / "opticsfile.22"))  # local testing
    madx.command.beam()
    special.make_lhc_thin(madx, sequence="lhcb1", slicefactor=4)
    cycle_from_ip3(madx, sequence="lhcb1")

    logger.info("Setting up orbit and creating beams")
    orbit_scheme = orbit.setup_lhc_orbit(madx, scheme="flat")
    special.make_lhc_beams(madx, energy=6500, emittance=3.75e-6)
    madx.use(sequence="lhcb1")

    logger.info("Applying coupling knob and matching working point")
    special.apply_lhc_coupling_knob(madx, coupling_knob=coupling_knob, telescopic_squeeze=True)
    matching.match_tunes_and_chromaticities(
        madx, "lhc", "lhcb1", tune_x, tune_y, chrom_x, chrom_y, calls=200, telescopic_squeeze=True
    )
    correct_machine = madx.table.twiss.dframe().copy()

    logger.info("Applying rigidity waist shift knob, matching working point & getting closest tune approach")
    special.apply_lhc_rigidity_waist_shift_knob(madx, rigidty_waist_shift_value=1, ir=1)
    matching.match_tunes_and_chromaticities(
        madx, "lhc", "lhcb1", tune_x, tune_y, chrom_x, chrom_y, calls=200, telescopic_squeeze=True
    )
    after_waist_shift = madx.table.twiss.dframe().copy()
    dqmin_with_waist = matching.get_closest_tune_approach(madx, "lhc", "lhcb1")
    madx.exit()

    logger.info("Computing beta-beating quantites")
    mean_betabeat_x = abs_betabeat(correct_machine.betx.to_numpy(), after_waist_shift.betx.to_numpy()).mean()
    stdev_betabeat_x = abs_betabeat(correct_machine.betx.to_numpy(), after_waist_shift.betx.to_numpy()).std()
    mean_betabeat_y = abs_betabeat(correct_machine.bety.to_numpy(), after_waist_shift.bety.to_numpy()).mean()
    stdev_betabeat_y = abs_betabeat(correct_machine.bety.to_numpy(), after_waist_shift.bety.to_numpy()).std()

    return Results(
        coupling_knob=coupling_knob,
        dqmin_after_waist=dqmin_with_waist,
        mean_betabeat_x=mean_betabeat_x,
        stdev_betabeat_x=stdev_betabeat_x,
        mean_betabeat_y=mean_betabeat_y,
        stdev_betabeat_y=stdev_betabeat_y,
    )


if __name__ == "__main__":
    with Madx(stdout=False) as mad:
        logger.critical(
            f"Using: pyhdtoolkit {pyhdtoolkit.__version__} | cpymad {cpymad.__version__}  | {mad.version}"
        )
    # simulation_results = make_simulation(  # afs run
    #     coupling_knob=%(COUPLING_KNOB)s,
    # )
    # simulation_results = make_simulation(  # local testing
    #     coupling_knob=0.003,
    # )
    # simulation_results.to_json(PATHS["htc_outputdir"] / "result_params.json")
