from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
import torch
from datasets import load_from_disk
import os

from src.textSummarizer.entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        device = torch.device("cpu")
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model_falcons = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(device)
        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_falcons)

        #loading the data
        dataset_samsum_pt = load_from_disk(self.config.data_path)

        trainer_args = TrainingArguments(
            output_dir=self.config.root_dir, num_train_epochs=3, warmup_steps=500,
            per_device_train_batch_size=1, per_device_eval_batch_size=1,
            weight_decay=0.01, logging_steps=10,
            eval_strategy='steps', eval_steps=500, save_steps=1e6,
            gradient_accumulation_steps=16
        ) 
        trainer = Trainer(model=model_falcons, args=trainer_args,
                  tokenizer=tokenizer, data_collator=seq2seq_data_collator,
                  train_dataset=dataset_samsum_pt["test"],
                  eval_dataset=dataset_samsum_pt["validation"])
        
        trainer.train()

        ## Save model
        model_falcons.save_pretrained(os.path.join(self.config.root_dir,"falconsai-samsum-model"))
        ## Save tokenizer
        tokenizer.save_pretrained(os.path.join(self.config.root_dir,"tokenizer"))
