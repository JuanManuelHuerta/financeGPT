import torch
from torch.utils.data import IterableDataset
from torch.utils.data.dataloader import DataLoader
from datasets import load_dataset, DownloadConfig
from transformers import AutoTokenizer
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
import psutil, os
from tqdm.auto import tqdm
import keyword
from argparse import Namespace
from torch.utils.tensorboard import SummaryWriter
import logging
import wandb
import datasets
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline, set_seed
from accelerate import Accelerator
from huggingface_hub import Repository, get_full_repo_name
from torch.optim import AdamW
from transformers import get_scheduler



org="jmhuerta"
tokenizer_ckpt="jmhuerta/financeTokenizer_english"
model_ckpt = "jmhuerta/financeGPT2"
project_name="jmhuerta/financeGPT2"
repo_name="JuanManuelHuerta/financeGPT2"
dataset_name="JanosAudran/financial-reports-sec"
subset="large_full"
field_name='sentence'



model = AutoModelForCausalLM.from_pretrained("./"+model_ckpt)
model.save_pretrained(model_ckpt, push_to_hub=True, organization=org)


