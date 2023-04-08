from datasets import load_dataset, DownloadConfig
from transformers import AutoTokenizer
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
import psutil, os
from tqdm.auto import tqdm
import keyword

model_ckpt = "financeTokenizer_english"
org = ""
vocab_size=128000
dataset_name="JanosAudran/financial-reports-sec"
subset="large_full"
field_name='sentence'


tokenizer = AutoTokenizer.from_pretrained("gpt2")

byte_to_unicode_map = bytes_to_unicode()
unicode_to_byte_map = dict((v, k) for k, v in byte_to_unicode_map.items())
base_vocab = list(unicode_to_byte_map.keys())
print(f'Size of our base vocabulary: {len(base_vocab)}')
print(f'First element: `{base_vocab[0]}`, last element: `{base_vocab[-1]}`')

## This is done in the CPU
length = 1000000

#dataset=load_dataset("wikipedia", "20220301.en",split="train", streaming=True)
dataset=load_dataset(dataset_name, subset,split="train")
#dataset_name = 'transformersbook/codeparrot-train'
#dataset = load_dataset(dataset_name, split="train", streaming=True)
iter_dataset = iter(dataset)

def batch_iterator(batch_size=100):
    for _ in tqdm(range(0, length, batch_size)):
        yield [next(iter_dataset)[field_name] for _ in range(batch_size)]

new_tokenizer = tokenizer.train_new_from_iterator(batch_iterator(), 
                                                  vocab_size=vocab_size,
                                                  initial_alphabet=base_vocab)

i=0
for w in new_tokenizer.vocab:
    print(w)
    if i>=10:
        break
    i+=1
    

new_tokenizer.push_to_hub(model_ckpt, organization=org)

