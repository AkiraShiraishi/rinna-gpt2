!pip install transformers SentencePiece
import torch
from transformers import T5Tokenizer, AutoModelForCausalLM
tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt-1b")
model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt-1b")
if torch.cuda.is_available():
    model = model.to("cuda")

def generate_text(text):
    token_ids = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")

    with torch.no_grad():
        output_ids = model.generate(
            token_ids.to(model.device),
            max_length=600,
            min_length=550,
            do_sample=True,
            top_k=500,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            bad_word_ids=[[tokenizer.unk_token_id]]
        )

    output = tokenizer.decode(output_ids.tolist()[0])
    return output 
text='こんにちは技術部です。この先の文はりんなちゃんが考えてくれます'
generate_text(text)