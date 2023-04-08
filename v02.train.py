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


def model_size(model):
    return sum(t.numel() for t in model.parameters())

class ConstantLengthDataset(IterableDataset):
    
    def __init__(self, tokenizer, dataset, seq_length=1024,
                 num_of_sequences=1024, chars_per_token=3.6,field_name="text"):
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.eos_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.field_name = field_name
        self.input_characters = seq_length * chars_per_token * num_of_sequences
    
    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.input_characters:
                    m=f"Buffer full: {buffer_len}>={self.input_characters:.0f}"
                    print(m)
                    break
                try:
                    m=f"Fill buffer: {buffer_len}<{self.input_characters:.0f}"
                    print(m)
                    buffer.append(next(iterator)[field_name])
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    iterator = iter(self.dataset)
            all_token_ids = []
            tokenized_inputs = self.tokenizer(buffer, truncation=False)
            for tokenized_input in tokenized_inputs['input_ids']:
                all_token_ids.extend(tokenized_input + [self.concat_token_id])
            
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    yield torch.tensor(input_ids)

def setup_logging(project_name):
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO, handlers=[
        logging.FileHandler(f"log/debug_{accelerator.process_index}.log"),
        logging.StreamHandler()])
    if accelerator.is_main_process: # We only want to set up logging once
        wandb.init(project=project_name, config=args)
        run_name = wandb.run.name
        tb_writer = SummaryWriter()
        tb_writer.add_hparams(vars(args), {'0': 0})
        logger.setLevel(logging.INFO)
        datasets.utils.logging.set_verbosity_debug()
        transformers.utils.logging.set_verbosity_info()
    else:
        tb_writer = None
        run_name = ''
        logger.setLevel(logging.ERROR)
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    return logger, tb_writer, run_name

def log_metrics(step, metrics):
    logger.info(f"Step {step}: {metrics}")
    if accelerator.is_main_process:
        wandb.log(metrics)
        [tb_writer.add_scalar(k, v, step) for k, v in metrics.items()]
        
def create_dataloaders(dataset_name,subset,field_name):
    train_data = load_dataset(dataset_name, subset,split="train")
    train_data = train_data.shuffle(seed=args.seed)
    valid_data = dataset = load_dataset(dataset_name,subset,split="validation")
    train_dataset = ConstantLengthDataset(tokenizer, train_data,
                                          seq_length=args.seq_length,field_name=field_name)
    valid_dataset = ConstantLengthDataset(tokenizer, valid_data,
                                          seq_length=args.seq_length,field_name=field_name)    
    train_dataloader=DataLoader(train_dataset, batch_size=args.train_batch_size)
    eval_dataloader=DataLoader(valid_dataset, batch_size=args.valid_batch_size)
    return train_dataloader, eval_dataloader

def get_grouped_params(model, no_decay=["bias", "LayerNorm.weight"]):
    params_with_wd, params_without_wd = [], []
    for n, p in model.named_parameters():
        if any(nd in n for nd in no_decay):
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)
    return [{'params': params_with_wd, 'weight_decay': args.weight_decay},
            {'params': params_without_wd, 'weight_decay': 0.0}]

def evaluate():
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(batch, labels=batch)
        loss = outputs.loss.repeat(args.valid_batch_size)
        losses.append(accelerator.gather(loss))
        if args.max_eval_steps > 0 and step >= args.max_eval_steps: break
    loss = torch.mean(torch.cat(losses))
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = torch.tensor(float("inf"))
    return loss.item(), perplexity.item()

org="jmhuerta"
tokenizer_ckpt="jmhuerta/financeTokenizer_english"
model_ckpt = "jmhuerta/financeGPT2"
project_name="jmhuerta/financeGPT2"
repo_name="JuanManuelHuerta/financeGPT2"
dataset_name="JanosAudran/financial-reports-sec"
subset="large_full"
field_name='sentence'

continue_from_hub = False
if continue_from_hub is True:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_ckpt)
    config = AutoConfig.from_pretrained("gpt2", vocab_size=len(tokenizer),gradient_checkpointing= True,)
    model = AutoModelForCausalLM.from_pretrained(model_ckpt)
else:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_ckpt)
    config = AutoConfig.from_pretrained("gpt2", vocab_size=len(tokenizer),gradient_checkpointing= True,)
    model = AutoModelForCausalLM.from_config(config)
print(f'GPT-2 (xl) size: {model_size(model)/1000**2:.1f}M parameters')
model.save_pretrained(model_ckpt, push_to_hub=True, organization=org)

examples, total_characters, total_tokens = 500, 0, 0
dataset=load_dataset(dataset_name, subset,split="train")
for _, example in tqdm(zip(range(examples), iter(dataset)), total=examples):
    total_characters += len(example[field_name])
    total_tokens += len(tokenizer(example[field_name]).tokens())
characters_per_token = total_characters / total_tokens
print(f"Characters per token {characters_per_token}")

#shuffled_dataset = dataset.shuffle(buffer_size=100)
shuffled_dataset = dataset.shuffle()
constant_length_dataset = ConstantLengthDataset(tokenizer, shuffled_dataset,num_of_sequences=10)
dataset_iterator = iter(constant_length_dataset)
lengths = [len(b) for _, b in zip(range(10), dataset_iterator)]
print(f"Lengths of the sequences: {lengths}")

# Commented parameters correspond to the small model
config = {"train_batch_size": 4, # 12, was 2
          "valid_batch_size": 2, # 12, this is the validation batchsize
          "weight_decay": 0.1,
          "shuffle_buffer": 1000,
          "learning_rate": 2e-4, # 5e-4
          "lr_scheduler_type": "cosine",
          "num_warmup_steps": 750, # 2000
          "gradient_accumulation_steps": 16, # 16
          "max_train_steps": 10000, # 50000, 150000
          "max_eval_steps": 2000,  # -1,
          "seq_length": 512,
          "seed": 1,
          "save_checkpoint_steps": 10000} # 15000

args = Namespace(**config)
set_seed(args.seed)

# Accelerator
accelerator = Accelerator()
samples_per_step = accelerator.state.num_processes * args.train_batch_size

# Logging
logger, tb_writer, run_name = setup_logging(project_name.split("/")[1])
logger.info(accelerator.state)

# Load model and tokenizer
# "gradient_checkpointing": true, in config.json

if accelerator.is_main_process:
    hf_repo = Repository(repo_name, clone_from=project_name, revision=run_name)
    #model = AutoModelForCausalLM.from_pretrained("models/codeparrot", gradient_checkpointing=True)
model = AutoModelForCausalLM.from_pretrained(project_name)
model.gradient_checkpointing_enable()
#tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_ckpt)

# Load dataset and dataloader
train_dataloader, eval_dataloader = create_dataloaders(dataset_name,subset,field_name)

# Prepare the optimizer and learning rate scheduler
optimizer = AdamW(get_grouped_params(model), lr=args.learning_rate)
lr_scheduler = get_scheduler(name=args.lr_scheduler_type, optimizer=optimizer,
                             num_warmup_steps=args.num_warmup_steps,
                             num_training_steps=args.max_train_steps,)
def get_lr():
    return optimizer.param_groups[0]['lr']

# Prepare everything with our `accelerator` (order of args is not important)
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader)

# Train model
model.train()
completed_steps = 0
for step, batch in enumerate(train_dataloader, start=1):
    loss = model(batch, labels=batch).loss
    log_metrics(step, {'lr': get_lr(), 'samples': step*samples_per_step,
                       'steps': completed_steps, 'loss/train': loss.item()})
    loss = loss / args.gradient_accumulation_steps
    accelerator.backward(loss)
    if step % args.gradient_accumulation_steps == 0:
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        completed_steps += 1
    if step % args.save_checkpoint_steps == 0:
        logger.info('Evaluating and saving model checkpoint')
        eval_loss, perplexity = evaluate()
        log_metrics(step, {'loss/eval': eval_loss, 'perplexity': perplexity})
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        if accelerator.is_main_process:
            unwrapped_model.save_pretrained(model_ckpt)
            hf_repo.push_to_hub(commit_message=f'step {step}')
        model.train()
    if completed_steps >= args.max_train_steps:
        break

# Evaluate and save the last checkpoint
logger.info('Evaluating and saving model after training')
eval_loss, perplexity = evaluate()
log_metrics(step, {'loss/eval': eval_loss, 'perplexity': perplexity})
accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(model)
if accelerator.is_main_process:
    unwrapped_model.save_pretrained(model_ckpt)
    hf_repo.push_to_hub(commit_message=f'final model')


