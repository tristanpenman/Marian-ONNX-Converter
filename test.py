import argparse

from transformers import MarianTokenizer
from core.marian import MarianOnnx

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Input model directory or name.")
    parser.add_argument("--device", default="cpu", help="Which device to run the model on (e.g. cpu or cuda)")
    return parser.parse_args()

def main(params):
    tokenizer = MarianTokenizer.from_pretrained(params.input)
    model = MarianOnnx(params.input, device=params.device)
    print("Enter text to translate (empty line to quit):")
    while True:
        line = input("> ").strip()
        if not line:
            break
        input_ids = tokenizer([line], return_tensors='pt', padding=True).to(params.device)
        tokens = model.generate(**input_ids)
        output = tokenizer.batch_decode(tokens, skip_special_tokens=True)
        print(output[0])

if __name__ == "__main__":
    params = parse_args()
    main(params)
