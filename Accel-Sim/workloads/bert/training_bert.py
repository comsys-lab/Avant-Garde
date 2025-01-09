import numpy as np
from datasets import Dataset
from pynvml import *
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, logging
import torch

no_deprecation_warning=True

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()
    
torch.ones((1, 1)).to("cuda")
print_gpu_utilization()

seq_len, dataset_size = 512, 160
dummy_data = {
    "input_ids": np.random.randint(100, 30000, (dataset_size, seq_len)),
    "labels": np.random.randint(0, 1, (dataset_size)),
}
ds = Dataset.from_dict(dummy_data)
ds.set_format("pt")

# Load Model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased").to("cuda")
# model = AutoModelForSequenceClassification.from_pretrained("gpt2").to("cuda")
print_gpu_utilization()

default_args = {
    "output_dir": "tmp",
    "evaluation_strategy": "steps",
    "num_train_epochs": 10,
    "log_level": "error",
 
    "report_to": "none",
}

# Vanilla Training
logging.set_verbosity_error()

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# model.config.pad_token_id = model.config.eos_token_id

training_args = TrainingArguments(per_device_train_batch_size=16, fp16=False, **default_args)
trainer = Trainer(model=model, args=training_args, train_dataset=ds)
result = trainer.train()
print_summary(result)


