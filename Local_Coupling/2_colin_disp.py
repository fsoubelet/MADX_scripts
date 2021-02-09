import json
import sys
from pathlib import Path
from typing import List, Union

from cpymad.madx import Madx
from loguru import logger
from pydantic import BaseModel
from pyhdtoolkit.cpymadtools import latwiss, matching, special
from pyhdtoolkit.utils.defaults import LOGURU_FORMAT

# ----- Setup ----- #

PATHS = {
    "optics2018": Path("/afs/cern.ch/eng/lhc/optics/runII/2018"),
    "htc_outputdir": Path("Outputdata"),
}

logger.remove()
logger.add(sys.stderr, format=LOGURU_FORMAT, level="DEBUG")
logger.add(PATHS["htc_outputdir"] / "full_pylog.log", format=LOGURU_FORMAT, level="TRACE")

# ----- Utilities ----- #


class Results(BaseModel):
    rigidity_knob: float
    colin_knob: float
    dqmin: float

    def to_json(self, filepath: Union[str, Path]) -> None:
        logger.debug(f"Exporting results structure to '{Path(filepath).absolute()}'")
        with Path(filepath).open("w") as results_file:
            json.dump(results.dict(), results_file)


def cycle_from_ip3(madx: Madx, sequence: str = "lhcb1") -> None:
    """Input a few commands to cycle the provided sequence from IP3."""
    logger.debug(f"Re-cycling sequence '{sequence}' from IP3")
    madx.command.seqedit(sequence=sequence)
    madx.command.flatten()
    madx.command.cycle(start="IP3")
    madx.command.endedit()


def fullpath(filepath: Path) -> str:
    return str(filepath.absolute())


# ----- Simulation ----- #


def make_simulation(colin_knob: float = 0.0, rigidity_knob: float = 0.0) -> None:
    """
    Get a complete LHC setup, implement colinearity knob and rigidity waist shift knob, get the Cminus.
    Both parameters and the results are stored in a structure and exported to JSON.

    Args:
        colin_knob (float): unit setting of the colinearity knob.
        rigidity_knob (float): unit setting of the rigidity waist shift knob.
    """
    madx = Madx(stdout=False)
    madx.option(echo=False, warn=False)
    logger.info(
        f"Running with a colinearity knob of {colin_knob} and a rigidity waist shift knob of {rigidity_knob}"
    )

    # Working point
    tune_x, tune_y = 62.31, 60.32
    chrom_x, chrom_y = 2.0, 2.0

    logger.info("Calling optics")
    madx.call(fullpath(PATHS["optics2018"] / "lhc_as-built.seq"))
    madx.call(fullpath(PATHS["optics2018"] / "PROTON" / "opticsfile.22"))
    cycle_from_ip3(madx, sequence="lhcb1")

    logger.info("Setting up orbit, creating beams and matching working point")
    orbit_scheme = orbit.setup_lhc_orbit(madx, scheme="flat")
    special.make_lhc_beams(madx, energy=6500, emittance=3.75e-6, npart=1.0e10)
    madx.use(sequence="lhcb1")
    matching.match_tunes_and_chromaticities(
        madx, "lhc", "lhcb1", tune_x, tune_y, chrom_x, chrom_y, telescopic_squeeze=True
    )

    logger.info("Setting up colinearity and rigidity waist shift knobs for IR1")
    special.apply_lhc_colinearity_knob(madx, colinearity_knob_value=colin_knob, ir=1)
    special.apply_lhc_rigidity_waist_shift_knob(madx, rigidty_waist_shift_value=rigidity_knob, ir=1)
    matching.match_tunes_and_chromaticities(
        madx, "lhc", "lhcb1", tune_x, tune_y, chrom_x, chrom_y, telescopic_squeeze=True
    )

    logger.info("Executing Closest Tune Approach\n")
    madx.twiss(chrom=True, ripken=True)
    dqmin = matching.get_closest_tune_approach(madx, "lhc", "lhcb1", telescopic_squeeze=True)
    madx.twiss(chrom=True, ripken=True)

    results = Results(rigidity_knob=rigidity_knob, colin_knob=colin_knob, dqmin=dqmin)
    results.to_json(PATHS["htc_outputdir"] / "result_params.json")


if __name__ == "__main__":
    make_simulation(
        colin_knob=%(COLIN_KNOB)s,
        rigidity_knob=%(RIGIDITY_WAIST_SHIFT_KNOB)s,
    )
