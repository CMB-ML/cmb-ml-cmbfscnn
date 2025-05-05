"""
This script runs a pipeline for prediction and analysis of the cleaned CMB signal using CMBFSCNN.

The pipeline consists of the following steps:
1. Preprocessing the data
2. Training the model
3. Predicting the cleaned CMB signal
4. Postprocessing the predictions
5. Converting predictions to common form for comparison across models
6. Generating per-pixel analysis results for each simulation
7. Generating per-pixel summary statistics for each simulation
8. Converting the theory power spectrum to a format that can be used for analysis
9. Generating per-ell power spectrum analysis results for each simulation
10. Generating per-ell power spectrum summary statistics for each simulation

And also generating various analysis figures, throughout.

Final comparison is performed in the main_analysis_compare.py script.

Usage:
    python main_cmbfscnn.py
"""
import logging

import hydra

from cmbml.core import (
                      PipelineContext,
                      LogMaker
                      )
from cmbml.core.A_check_hydra_configs import HydraConfigCheckerExecutor
from cmbml.sims import MaskCreatorExecutor
from cmbfscnn_local import (
                           HydraConfigCMBFSCNNCheckerExecutor,
                           PreprocessMakeScaleExecutor,
                           PreprocessExecutor,
                           TrainingExecutor,
                           PredictionExecutor,
                           PostprocessExecutor,
                           )

from cmbml.analysis import (
                            CommonRealPostExecutor,
                            CommonPredPostExecutor,
                            CommonShowSimsPostExecutor,
                            PixelAnalysisExecutor,
                            PixelSummaryExecutor,
                            )


logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="cfg", config_name="config_cmbfscnn")
def run_cmbfscnn(cfg):
    logger.debug(f"Running {__name__} in {__file__}")

    log_maker = LogMaker(cfg)
    log_maker.log_procedure_to_hydra(source_script=__file__)

    pipeline_context = PipelineContext(cfg, log_maker)

    pipeline_context.add_pipe(HydraConfigCheckerExecutor)
    pipeline_context.add_pipe(HydraConfigCMBFSCNNCheckerExecutor)

    pipeline_context.add_pipe(PreprocessMakeScaleExecutor)
    
    pipeline_context.add_pipe(PreprocessExecutor)
    
    pipeline_context.add_pipe(TrainingExecutor)

    pipeline_context.add_pipe(PredictionExecutor)
    
    pipeline_context.add_pipe(PostprocessExecutor)

    # In the following, "Common" means "Apply the same postprocessing to all models"; requires a mask
    # Apply to the target (CMB realization)
    pipeline_context.add_pipe(CommonRealPostExecutor)

    # Apply to CMBFSCNN's predictions
    pipeline_context.add_pipe(CommonPredPostExecutor)

    # Show results of cleaning
    pipeline_context.add_pipe(CommonShowSimsPostExecutor)

    pipeline_context.add_pipe(PixelAnalysisExecutor)
    pipeline_context.add_pipe(PixelSummaryExecutor)

    pipeline_context.prerun_pipeline()

    try:
        pipeline_context.run_pipeline()
    except Exception as e:
        logger.exception("An exception occured during the pipeline.", exc_info=e)
        raise e
    finally:
        logger.info("Pipeline completed.")
        log_maker.copy_hydra_run_to_dataset_log()


if __name__ == "__main__":
    run_cmbfscnn()
