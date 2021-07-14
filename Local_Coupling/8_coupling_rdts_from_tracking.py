"""
This script is used to start a batch simulations looking at the coupling RDTs and CRDTs calculation in
the presence of the colinearity knob in an IR (here IR1) in both theoretical calculations from TWISS
output with optics_functions and "measurement" output from omc3's analysis codes.

The measurement is simulated by tracking a particle and defining BPMs as observation points. The BPMs
are taken from the model directory, which should be made ahead of time with omc3's model_creator as shown
below.

To create the appropriate model through Python:
```python
from pathlib import Path
from omc3.model_creator import create_instance_and_model
create_instance_and_model(
    accel="lhc",
    type="nominal",
    year="2018",
    beam=1,
    energy=6.5,
    nat_tunes=[62.31, 60.32],
    modifiers=[Path("/afs/cern.ch/eng/lhc/optics/runII/2018/PROTON/opticsfile.22")],
    outputdir=Path("/afs/cern.ch/work/f/fesoubel/lhc_model"),  # or wherever
)
```

Need to provide at HTCondor submission time with `job_submitter`:
- LHC_MODEL_DIR -> location of a single model dir for all simulations to tap into
- COLIN_KNOB_SETTING -> value of the colinearty knob
"""
import shutil
from pathlib import Path

import cpymad
import numpy as np
import pyhdtoolkit
import tfs
from cpymad.madx import Madx
from loguru import logger
from omc3.hole_in_one import hole_in_one_entrypoint as hole_in_one
from omc3.tbt_converter import converter_entrypoint as tbt_converter
from optics_functions.coupling import coupling_via_cmatrix
from pyhdtoolkit.cpymadtools import matching, orbit, special, track, twiss
from pyhdtoolkit.utils import defaults

# ----- Setup ----- #

PATHS = {
    "optics2018": Path("/afs/cern.ch/eng/lhc/optics/runII/2018"),
    "local": Path("/Users/felixsoubelet/cernbox/OMC/MADX_scripts/Local_Coupling"),
    "htc_outputdir": Path("Outputdata"),
}

defaults.config_logger(level="DEBUG")
logger.add(
    PATHS["htc_outputdir"] / "full_pylog.log",
    format=defaults.LOGURU_FORMAT,
    enqueue=True,
    level="DEBUG",
)

# ----- Utilities ----- #


def fullpath(filepath: Path) -> str:
    return str(filepath.absolute())


def split_rdt_complex_columns(coupling_data_frame: tfs.TfsDataFrame) -> tfs.TfsDataFrame:
    """
    Split the coupling RDT columns in the provided TfsDataFrame into real, imaginary and amplitude columns.
    This is done to guarantee the data can be written to disk as complex columns are not supported by the
    TFS data format.

    Args:
        coupling_data_frame (tfs.TfsDataFrame): your coupling data frame with

    Returns:
        The data frame with RDT complex columns split and then removed.
    """
    logger.debug("Splitting and removing complex columns from provided TfsDataFrame")
    data_frame = coupling_data_frame.copy()
    logger.trace("Splitting F1001 column")
    data_frame["F1001_REAL"] = np.real(data_frame.F1001.to_numpy())
    data_frame["F1001_IMAG"] = np.imag(data_frame.F1001.to_numpy())
    data_frame["F1001_AMP"] = np.abs(data_frame.F1001.to_numpy())
    logger.trace("Splitting F1010 column")
    data_frame["F1010_REAL"] = np.real(data_frame.F1010.to_numpy())
    data_frame["F1010_IMAG"] = np.imag(data_frame.F1010.to_numpy())
    data_frame["F1010_AMP"] = np.abs(data_frame.F1010.to_numpy())
    return data_frame


# ----- Simulation ----- #


def make_simulation(colin_knob: float, lhc_model_dir: str) -> None:
    """
    Get a complete LHC setup, implement colnearity knob in IR, perform tracking and do data analysis on
    the outputs. Everything of interest is written to disk in the `Outputdata` folder, which HTCondor
    will bring back to local space after the jobs are done.

    Args:
        colin_knob (float): unit setting of the colinearity knob.
        lhc_model_dir (str): location of a single model dir for all simulations to tap into.

    Returns:
        None, the outputs are analysis results from `omc3` and will be written to disk.
    """
    logger.debug("Loading observation BPMs from model")
    observation_bpms = tfs.read(f"{lhc_model_dir}/twiss.dat").NAME.tolist()

    with Madx() as madx:
        # ----- Init & Machine ----- #
        logger.info(f"Running with a colinearity knob setting of {colin_knob:d}")
        madx.option(echo=False, warn=False)
        logger.debug("Calling optics")
        # madx.call(fullpath(PATHS["optics2018"] / "lhc_as-built.seq"))  # afs
        # madx.call(fullpath(PATHS["optics2018"] / "PROTON" / "opticsfile.22"))  # afs
        # madx.call(fullpath(PATHS["local"] / "sequences" / "lhc_as-built.seq"))  # local testing
        # madx.call(fullpath(PATHS["local"] / "optics" / "opticsfile.22"))  # local testing

        # ----- Setup ----- #
        special.re_cycle_sequence(madx, sequence="lhcb1", start="MSIA.EXIT.B1")  # same re-cycling as model
        orbit_scheme = orbit.setup_lhc_orbit(madx, scheme="flat")
        special.make_lhc_beams(madx, energy=6500, emittance=3.75e-6)
        madx.use(sequence="lhcb1")

        # ----- Colin Setting ----- #
        logger.info(f"Applying colinearity knob setting")
        matching.match_tunes_and_chromaticities(madx, "lhc", "lhcb1", 62.27, 60.36, 2.0, 2.0)  # avoid flip
        special.apply_lhc_colinearity_knob(madx, colinearity_knob_value=colin_knob, ir=1)
        matching.match_tunes_and_chromaticities(madx, "lhc", "lhcb1", 62.31, 60.32, 2.0, 2.0)  # guarantee
        twiss_df: tfs.TfsDataFrame = twiss.get_twiss_tfs(madx).drop(index="IP1.L1")

        # ----- Tracking ----- #
        special.make_lhc_thin(madx, sequence="lhcb1", slicefactor=4)
        _ = track.track_single_particle(
            madx,
            sequence="lhcb1",
            nturns=1023,
            initial_coordinates=(2e-4, 0, 2e-4, 0, 0, 0),
            observation_points=observation_bpms,
            FILE="Outputdata/track",  # MAD-X will append "one" to this name since we give ONETABLE
            RECLOSS=True,  # Create an internal table recording lost particles
            ONEPASS=True,  # Do not search closed orbit and give coordinates relatively to the reference orbit
            DUMP=True,  # Write to file
            ONETABLE=True,  # Gather all observation points data into a single table (and file if DUMP set)
        )

    logger.info("Computing coupling RDTs through TWISS theoretical approach and writing to disk")
    coupling_df = coupling_via_cmatrix(twiss_df)
    twiss_df[["F1001", "F1010"]] = coupling_df[["F1001", "F1010"]]
    twiss_df_bpm: tfs.TfsDataFrame = twiss_df[twiss_df.KEYWORD == "monitor"]
    twiss_df_bpm = split_rdt_complex_columns(twiss_df_bpm)
    tfs.write("Outputdata/coupling_twiss_bpm.tfs", twiss_df_bpm)

    logger.info("Doing trackone file conversion with omc3's tbt_converter")
    tbt_converter(
        files=[Path("Outputdata/trackone")],
        tbt_datatype="trackone",
        outputdir="Outputdata",
        drop_elements=["LHCB1MSIA.EXIT.B1_P_"],  # start of machine point added by MAD-X
    )

    logger.info("Performing harmonic and optics analysis of tracking data wiith omc3's hole_in_one")
    hole_in_one(
        harpy=True,
        files=[Path("Outputdata/trackone.sdds")],
        tbt_datatype="lhc",
        turns=[0, 1023],
        autotunes="transverse",
        to_write=["lin", "spectra", "full_spectra", "bpm_summary"],  # in `lin_files` subfolder of outputdir
        turn_bits=17,
        optics=True,
        accel="lhc",
        year="2018",
        beam=1,
        energy=6.5,
        model_dir=Path(lhc_model_dir),
        compensation="none",
        three_bpm_method=True,
        nonlinear=["rdt", "crdt"],
        outputdir=Path("Outputdata/measured_optics"),
    )

    logger.info("Cleaning up: removing 'trackone' and 'lin_files'")
    Path("Outputdata/trackone").unlink()
    shutil.rmtree("Outputdata/measured_optics/lin_files")


# ----- Running ----- #


if __name__ == "__main__":
    with Madx(stdout=False) as mad:
        logger.critical(
            f"Using: pyhdtoolkit {pyhdtoolkit.__version__} | cpymad {cpymad.__version__}  | {mad.version}"
        )
    # make_simulation(  # afs run
    #     colin_knob=%(COLIN_KNOB_SETTING)s,
    #     lhc_model_dir="%(LHC_MODEL_DIR)s",
    # )
    # make_simulation(  # local testing
    #     colin_knob=3,
    #     lhc_model_dir="/Users/felixsoubelet/cernbox/OMC/Local_Coupling_Correction/8_check_crdts_1bpm/data/lhc_model",
    # )
