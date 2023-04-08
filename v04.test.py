import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline, set_seed
import re
from transformers import set_seed 
from transformers import pipeline, set_seed



#model = AutoModelForCausalLM.from_pretrained("./")
#model.save_pretrained(model_ckpt, push_to_hub=True, organization=org)
#model_ckpt = 'transformersbook/codeparrot-small'



org="jmhuerta"
tokenizer_ckpt="jmhuerta/financeTokenizer_english"
model_ckpt = "jmhuerta/financeGPT2"
project_name="jmhuerta/financeGPT2"
repo_name="JuanManuelHuerta/financeGPT2"
dataset_name="JanosAudran/financial-reports-sec"
subset="large_full"
field_name='sentence'



tokenizer = AutoTokenizer.from_pretrained(tokenizer_ckpt)
generation = pipeline('text-generation', model=model_ckpt, \
                      tokenizer=tokenizer,device=0)


def complete_code(pipe, prompt, max_length=64, num_completions=4, seed=1):
    set_seed(seed)

    print("Prompt is:",prompt)
    gen_kwargs = {"temperature":0.4, "top_p":0.95, "top_k":0, "num_beams":1,
                  "do_sample":True,}
    code_gens = generation(prompt, num_return_sequences=num_completions, max_length=max_length, **gen_kwargs)
    for item in code_gens:
        print(item['generated_text'])
        print("\n\n")



#prompt = 'The French Revolution started by a mob storming the Bastille.'
#complete_code(generation, prompt,max_length=24)


#prompt = 'Clara Newman was a philosopher, politician and actress.'
#complete_code(generation, prompt,max_length=64)


#prompt = 'Boston is a city in the northeast United State with many colleges and universities.'
#complete_code(generation, prompt, max_length=96)


while True:
    prompt = input("Enter text>>")
    complete_code(generation, prompt, max_length=24)
