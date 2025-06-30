from src.textSummarizer.logging import logger

from src.textSummarizer.pipeline.data_ingestion_stage import DataIngestionTrainingPipeline
#from src.textSummarizer.pipeline.data_transformation import DataTransformationTrainingPipeline
#from src.textSummarizer.pipeline.model_trainer import ModelTrainerTrainingPipeline
#from src.textSummarizer.pipeline.model_evaluation import ModelEvaluationTrainingPipeline

STAGE_NAME="Data Ingestion stage"
try:
    logger.info(f"stage {STAGE_NAME} initiated")
    data_ingestion_pipeline=DataIngestionTrainingPipeline()
    data_ingestion_pipeline.initiate_data_ingestion()
    logger.info(f"Stage {STAGE_NAME} Completed")
except Exception as e:
    logger.exception(e)
    raise e
