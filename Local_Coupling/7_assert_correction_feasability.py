import json
import sys
from pathlib import Path
from typing import List, Sequence, Union

import cpymad
import numpy as np
import pandas as pd
import pyhdtoolkit
from cpymad.madx import Madx
from loguru import logger
from pydantic import BaseModel
from pyhdtoolkit.cpymadtools import matching, orbit, special
from pyhdtoolkit.utils import defaults

# ----- Setup ----- #

PATHS = {
    "optics2018": Path("/afs/cern.ch/eng/lhc/optics/runII/2018"),
    "local": Path("/Users/felixsoubelet/cernbox/OMC/MADX_scripts/Local_Coupling"),
    "htc_outputdir": Path("Outputdata"),
}

defaults.config_logger(level="DEBUG")
logger.add(PATHS["htc_outputdir"] / "full_pylog.log", format=defaults.LOGURU_FORMAT, level="TRACE")

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


def fullpath(filepath: Path) -> str:
    return str(filepath.absolute())


def misalign_lhc_q1_to_q6(
    madx: Madx, ip: int, beam: int, sides: Sequence[str] = ("r", "l"), table: str = "errors", **kwargs
) -> None:
    """
    Apply misalignment errors to triplet quadrupoles on a given side of a given IP. In case of a sliced
    lattice, this will misalign all slices of the magnet together.

    Args:
        madx (cpymad.madx.Madx): an instanciated cpymad Madx object.
        ip (int): the interaction point around which to apply errors.
        beam (int): beam number to apply the errors to. Unlike triplet quadrupoles which are single
            aperture, Q4 to Q6 are not.
        sides (Sequence[str]): sides of the IP to apply error on the triplets, either L or R or both.
            Defaults to both.
        table (str): the name of the internal table that will save the assigned errors. Defaults to
            'triplet_errors'.

    Keyword Args:
        Any keyword argument to give to the EALIGN command, including the error to apply (DX, DY,
        DPSI etc) as a string, like it would be given directly into MAD-X.

    Examples:
        dist_center = 1e-3  # for rad tilt
        misalign_lhc_q4_q6(madx, ip=1, sides="RL", dx="1E-5 * TGAUSS(2.5)")
        misalign_lhc_q4_q6(madx, ip=5, sides="RL", dpsi=f"{dist_center} * TGAUSS(2.5)")
    """
    if ip not in (1, 2, 5, 8) or any(side.upper() not in ("R", "L") for side in sides):
        logger.error("Either the IP number of the side provided are invalid, not applying any error.")
        raise ValueError("Invalid 'ip' or 'sides' argument")

    logger.info(f"Applying alignment errors to individual quadrupoles 4 to 6, with arguments {kwargs}")
    madx.command.select(flag="error", clear=True)
    for side in sides:
        logger.debug(f"Applying errors on {'right' if side.upper() == 'R' else 'left'} side of IP{ip:d}")
        madx.command.select(flag="error", pattern=f"^MQXA.1{side.upper()}{ip:d}")  # Q1 LHC
        madx.command.select(flag="error", pattern=f"^MQXFA.[AB]1{side.upper()}{ip:d}")  # Q1A & Q1B HL-LHC
        madx.command.select(flag="error", pattern=f"^MQXB.[AB]2{side.upper()}{ip:d}")  # Q2A & Q2B LHC
        madx.command.select(flag="error", pattern=f"^MQFXB.[AB]2{side.upper()}{ip:d}")  # Q2A & Q2B HL-LHC
        madx.command.select(flag="error", pattern=f"^MQXA.3{side.upper()}{ip:d}")  # Q3 LHC
        madx.command.select(flag="error", pattern=f"^MQXFA.[AB]3{side.upper()}{ip:d}")  # Q3A & Q3B HL-LHC
        madx.command.select(flag="error", pattern=f"^MQY.4{side.upper()}{ip:d}.B{beam:d}")  # Q4 (HL)LHC
        madx.command.select(flag="error", pattern=f"^MQML.5{side.upper()}{ip:d}.B{beam:d}")  # Q5 (HL)LHC
        madx.command.select(flag="error", pattern=f"^MQML.6{side.upper()}{ip:d}.B{beam:d}")  # Q6 (HL)LHC
    madx.command.ealign(**kwargs)

    logger.debug(f"Saving assigned errors in internal table '{table if table else 'etable'}'")
    madx.command.etable(table=table)

    logger.trace("Clearing up error flag")
    madx.command.select(flag="error", clear=True)


def match_no_coupling_at_ip_through_rterms(madx: Madx, sequence: str, ip: int) -> None:
    """
    Matching commands to get R-matrix terms to be 0 at given IP, using skew quad correctors independently.
    """
    logger.info(f"Matching R-terms for no coupling at IP {ip:d}")
    madx.command.match(sequence=sequence, chrom=True)
    madx.command.vary(name=f"KQSX3.R{ip:d}")  # using skew quad correctors independently here!
    madx.command.vary(name=f"KQSX3.L{ip:d}")
    madx.command.constraint(range_=f"IP{ip:d}", R11=0)
    madx.command.constraint(range_=f"IP{ip:d}", R12=0)
    madx.command.constraint(range_=f"IP{ip:d}", R21=0)
    madx.command.constraint(range_=f"IP{ip:d}", R22=0)
    madx.command.lmdif(calls=500, tolerance=1e-21)
    madx.command.endmatch()


# ----- Simulation ----- #


def make_simulation(tilt_mean: float = 0.0, seeds: int = 50) -> Results:
    """
    Get a complete LHC setup, implement colinearity knob and rigidity waist shift knob, get the Cminus.
    Both parameters and the results are stored in a structure and exported to JSON.

    Args:
        tilt_mean (float): mean value of the dpsi tilt distribution when applying to Q1-6 quadrupoles.
        seeds (int): the number of runs with different seeds to do for each tilt value.
    """
    tilt_errors: List[pd.DataFrame] = []
    corrected_twisses: List[pd.DataFrame] = []

    for _ in range(seeds):
        with Madx(command_log=fullpath(PATHS["htc_outputdir"] / "cpymad_commands.log")) as madx:
            logger.info(f"Running with a mean tilt of {tilt_mean:.1E}")
            madx.option(echo=False, warn=False)
            madx.option(rand="best", randid=np.random.randint(1, 11))  # random number generator
            madx.eoption(seed=np.random.randint(1, 999999999))  # not using default seed

            logger.debug("Calling optics")
            # madx.call(fullpath(PATHS["optics2018"] / "lhc_as-built.seq"))  # afs
            # madx.call(fullpath(PATHS["optics2018"] / "PROTON" / "opticsfile.22"))  # afs
            madx.call(fullpath(PATHS["local"] / "sequences" / "lhc_as-built.seq"))  # local testing
            madx.call(fullpath(PATHS["local"] / "optics" / "opticsfile.22"))  # local testing

            special.re_cycle_sequence(madx, sequence="lhcb1", start="IP3")
            orbit_scheme = orbit.setup_lhc_orbit(madx, scheme="flat")
            special.make_lhc_beams(madx, energy=6500, emittance=3.75e-6)
            madx.use(sequence="lhcb1")
            matching.match_tunes_and_chromaticities(
                madx, "lhc", "lhcb1", 62.31, 60.32, 2.0, 2.0, calls=200, telescopic_squeeze=True
            )

            logger.info("Applying misalignments to IR quads 1 to 6")
            misalign_lhc_q1_to_q6(
                madx, ip=1, beam=1, sides="RL", dpsi=f"{tilt_mean} + {abs(tilt_mean) * 0.15} * TGAUSS(2.5)"
            )
            tilt_errors.append(
                madx.table.errors.dframe().copy().set_index("name", drop=True).loc[:, ["dpsi"]]
            )  # just get the dpsi column, save memory

            match_no_coupling_at_ip_through_rterms(madx, sequence="lhcb1", ip=1)
            madx.twiss(ripken=True)
            corrected_twisses.append(madx.table.twiss.dframe().copy().set_index("name", drop=True))

    all_errors = pd.concat(tilt_errors)  # concatenating all errors for this tilt's runs
    all_results = pd.concat(corrected_twisses)  # concatenating all resulting twisses for this tilt's runs

    return Results(
        tilt_mean=tilt_mean,
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
    )


# ----- Running ----- #


if __name__ == "__main__":
    with Madx(stdout=False) as mad:
        logger.critical(
            f"Using: pyhdtoolkit {pyhdtoolkit.__version__} | cpymad {cpymad.__version__}  | {mad.version}"
        )
    simulation_results = make_simulation(  # afs run
        tilt_mean=%(DPSI_MEAN)s,
        seeds=%(SEEDS)s,
    )
    # simulation_results = make_simulation(  # local testing
    #     tilt_mean=5e-4,
    #     seeds=2,
    # )
    simulation_results.to_json(PATHS["htc_outputdir"] / "results.json")
