# Marian-ONNX

An implementation of Marian for ONNX.

Includes a conversion script for converting from Hugging Face to ONNX. Originally implemented by [kcosta42](https://github.com/kcosta42).

## Conversion

1. Make sure you have the python dependencies installed.
    ```sh
    python3 -m pip install -r requirements.txt
    ```

2. Download a Marian model from huggingface hub (you may need to install `git-lfs`)

    ```sh
    git lfs clone https://huggingface.co/Helsinki-NLP/opus-mt-fr-en ./models/fr-en
    ```

3. Convert the Marian model to ONNX.

    ```sh
    python3 convert.py ./models/fr-en
    ```

    Alternatively, convert without quantization:

    ```sh
    python3 convert.py --no-quantize ./models
    ```

    The output will be written to `./outs/fr-en`

## Testing

To test the converted model, use `test.py`:

```
python3 test.py --device cpu ./outs/fr-en
```

For example:
```
$ python3 test.py --device cpu ./outs/fr-en

Enter text to translate (empty line to quit):
> Je m'appelle Bob
My name is Bob.
>
```

### Other Devices

The `--device` argument can be used to run the model on another device supported by ONNX:

```
python3 test.py --device cuda ./outs/fr-en
```

### Details

All that `test.py` does is wrap the following code in a user-friendly prompt:

```py
from transformers import MarianTokenizer
from core.marian import MarianOnnx

DEVICE = 'cpu'
SENTENCES = ["Bonjour", "Je m'appelle Bob"]
MODEL_PATH = "./outs/fr-en"

tokenizer = MarianTokenizer.from_pretrained(MODEL_PATH)

input_ids = tokenizer(SENTENCES, return_tensors='pt', padding=True).to(DEVICE)
model = MarianOnnx(MODEL_PATH, device=DEVICE)
tokens = model.generate(**input_ids)
print(tokenizer.batch_decode(tokens, skip_special_tokens=True))

# ['Hello.', 'My name is Bob.']
```

## Benchmark

Use `core.benchmark` to measure performance.

Benchmark for the `opus-mt-fr-en` model on a NVIDIA GeForce RTX 2070:

```sh
$> python3 -m core.benchmark

CPU Benchmark:

Warming up ORT...
ORT CPU: 48 ms / sentence
PyTorch CPU: 152 ms / sentence

-----
GPU Benchmark:

Warming up ORT...
ORT GPU: 31 ms / sentence
PyTorch GPU: 66 ms / sentence
```
