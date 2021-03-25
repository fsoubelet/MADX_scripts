import json
import sys
from pathlib import Path
from typing import List, Union

import cpymad
import numpy as np
import pyhdtoolkit
from cpymad.madx import Madx
from loguru import logger
from optics_functions import coupling
from pydantic import BaseModel
from pyhdtoolkit.cpymadtools import latwiss, matching, orbit, special, twiss
from pyhdtoolkit.utils import defaults

# ----- Setup ----- #

PATHS = {
    "optics2018": Path("/afs/cern.ch/eng/lhc/optics/runII/2018"),
    "local": Path("/Users/felixsoubelet/cernbox/OMC/MADX_scripts/Local_Coupling"),
    "htc_outputdir": Path("Outputdata"),
}

defaults.config_logger(level="INFO")
logger.add(PATHS["htc_outputdir"] / "full_pylog.log", format=defaults.LOGURU_FORMAT, level="TRACE")


# ----- Utilities ----- #


class Results(BaseModel):
    colinearity_knob: float
    tilt_angle: float
    rigidity_knob: float
    f1001_abs_ip1: float
    dqmin_cta: float
    dqmin_calaga: float
    dqmin_persson: float

    def to_json(self, filepath: Union[str, Path]) -> None:
        logger.debug(f"Exporting results structure to '{Path(filepath).absolute()}'")
        with Path(filepath).open("w") as results_file:
            json.dump(self.dict(), results_file)

    @classmethod
    def from_json(cls, json_file: Union[str, Path]):
        """
        Load a saved Results structure data from disk, from the JSON format.
        Args:
                json_file (Union[Path, str]): PosixPath object or string with the save file location.
        """
        logger.info(f"Loading JSON data from file at '{Path(json_file).absolute()}'")
        return cls.parse_file(json_file, content_type="application/json")


def fullpath(filepath: Path) -> str:
    return str(filepath.absolute())


# ----- Simulation ----- #


def make_simulation(
    colinearity_knob: float = 0.0, tilt_angle: float = 0, rigidity_knob: float = 0.0
) -> Results:
    """
    Get a complete LHC setup, implement colinearity knob and rigidity waist shift knob, get the Cminus.
    Both parameters and the results are stored in a structure and exported to JSON.

    Args:
        colinearity_knob (float): value of the coupling knob to apply.
        tilt_angle (float): tilt error to apply to Q1 triplets.
        rigidity_knob (float): unit setting of the rigidity waist shift knob.
    """
    with Madx(command_log=fullpath(PATHS["htc_outputdir"] / "cpymad_commands.log")) as madx:
        madx.option(echo=False, warn=False)

        logger.info("Calling optics")
        # madx.call(fullpath(PATHS["optics2018"] / "lhc_as-built.seq"))  # afs
        # madx.call(fullpath(PATHS["optics2018"] / "PROTON" / "opticsfile.22"))  # afs
        madx.call(fullpath(PATHS["local"] / "sequences" / "lhc_as-built.seq"))  # local testing
        madx.call(fullpath(PATHS["local"] / "optics" / "opticsfile.22"))  # local testing

        logger.info("Setting up orbit and creating beams")
        special.re_cycle_sequence(madx, sequence="lhcb1", start="IP3")
        orbit_scheme = orbit.setup_lhc_orbit(madx, scheme="flat")
        special.make_lhc_beams(madx, energy=6500, emittance=3.75e-6)
        special.make_lhc_thin(madx, sequence="lhcb1", slicefactor=4)
        madx.use(sequence="lhcb1")
        # Widen tune split first to avoid a tune flip when applying knobs
        matching.match_tunes_and_chromaticities(
            # madx, "lhc", "lhcb1", 62.27, 60.36, 2.0, 2.0, telescopic_squeeze=True
            madx, "lhc", "lhcb1", 62.31, 60.32, 2.0, 2.0, calls=500, telescopic_squeeze=True
        )

        logger.info("Applying tilt error to Q1s")
        madx.command.select(flag="error", pattern="MQXA.1L1")
        madx.command.ealign(dpsi=tilt_angle)
        madx.command.select(flag="error", clear=True)
        madx.command.select(flag="error", pattern="MQXA.1R1")
        madx.command.ealign(dpsi=-tilt_angle)

        logger.info("Applying colinearity knob, rigidity waist shift knob and matching working point")
        special.apply_lhc_colinearity_knob(madx, colinearity_knob_value=colinearity_knob, ir=1)
        special.apply_lhc_rigidity_waist_shift_knob(madx, rigidty_waist_shift_value=rigidity_knob, ir=1)
        matching.match_tunes_and_chromaticities(
            madx, "lhc", "lhcb1", 62.31, 60.32, 2.0, 2.0, calls=500, telescopic_squeeze=True,
        )
        dqmin_cta = matching.get_closest_tune_approach(madx, "lhc", "lhcb1", telescopic_squeeze=True)
        twiss_tfs = twiss.get_twiss_tfs(madx)

    logger.info("Computing coupling RDTs and Cminus (Persson method)")
    twiss_tfs[["F1001", "F1010"]] = coupling.coupling_via_cmatrix(twiss_tfs, output=["rdts"])
    f1001_abs_ip1 = twiss_tfs.F1001.abs()["IP1"]
    dqmin_df_calaga = coupling.closest_tune_approach(twiss_tfs, twiss_tfs.Q1, twiss_tfs.Q2, "calaga")
    dqmin_df_persson = coupling.closest_tune_approach(twiss_tfs, twiss_tfs.Q1, twiss_tfs.Q2, "persson")

    return Results(
        colinearity_knob=colinearity_knob,
        tilt_angle=tilt_angle,
        rigidity_knob=rigidity_knob,
        f1001_abs_ip1=f1001_abs_ip1,
        dqmin_cta=dqmin_cta,
        dqmin_calaga=dqmin_df_calaga.mean()[0],
        dqmin_persson=dqmin_df_persson.abs().mean()[0],  # persson gives complex values, use .abs()
    )


if __name__ == "__main__":
    with Madx(stdout=False) as mad:
        logger.critical(
            f"Using: pyhdtoolkit {pyhdtoolkit.__version__} | cpymad {cpymad.__version__}  | {mad.version}"
        )
    # simulation_results = make_simulation(  # afs run
    #     colinearity_knob=%(COLIN_KNOB)s,
    #     tilt_angle=%(TILT_ANGLE)s,
    #     rigidity_knob=%(RIGIDITY_WAIST_SHIFT_KNOB)s,
    # )
    simulation_results = make_simulation(  # local testing
        colinearity_knob=-2,
        tilt_angle=6e-4,
        rigidity_knob=1,
    )
    simulation_results.to_json(PATHS["htc_outputdir"] / "result_params.json")
