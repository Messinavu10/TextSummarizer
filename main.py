from src.textSummarizer.logging import logger

from src.textSummarizer.pipeline.data_ingestion_stage import DataIngestionTrainingPipeline
from src.textSummarizer.pipeline.data_transformation_stage import DataTransformationTrainingPipeline
from src.textSummarizer.pipeline.model_trainer_stage import ModelTrainerTrainingPipeline
from src.textSummarizer.pipeline.model_evaluation_stage import ModelEvaluationTrainingPipeline

STAGE_NAME="Data Ingestion stage"
try:
    logger.info(f"stage {STAGE_NAME} initiated")
    data_ingestion_pipeline=DataIngestionTrainingPipeline()
    data_ingestion_pipeline.initiate_data_ingestion()
    logger.info(f"Stage {STAGE_NAME} Completed")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME="Data Transformation stage"
try:
    logger.info(f"stage {STAGE_NAME} initiated")
    data_ingestion_pipeline=DataTransformationTrainingPipeline()
    data_ingestion_pipeline.initiate_data_transformation()
    logger.info(f"Stage {STAGE_NAME} Completed")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME="Model Trainer stage"

try:
    logger.info(f"stage {STAGE_NAME} initiated")
    model_trainer_pipeline=ModelTrainerTrainingPipeline()
    model_trainer_pipeline.initiate_model_trainer()
    logger.info(f"Stage {STAGE_NAME} Completed")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Model Evaluation stage"
try: 
   logger.info(f"stage {STAGE_NAME} initiated")
   model_evaluation = ModelEvaluationTrainingPipeline()
   model_evaluation.initiate_model_evaluation()
   logger.info(f"Stage {STAGE_NAME} Completed")
except Exception as e:
        logger.exception(e)
        raise e
