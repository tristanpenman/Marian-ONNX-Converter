import argparse
import os
import shutil
import warnings

warnings.filterwarnings('ignore')

from core.utils import generate_onnx_graph
from core.benchmark import verify_export


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Input model directory or name.")
    parser.add_argument("-o", "--output", type=str, default="./outs", help="Output directory.")
    parser.add_argument("--no-quantize", action="store_false", default=True,
                        help="Disable model quantization.")
    return parser.parse_args()


def main(params):
    outdir = os.path.join(params.output, os.path.basename(params.input))
    os.makedirs(outdir, exist_ok=True)

    encoder_path = os.path.join(outdir, "encoder.onnx")
    decoder_path = os.path.join(outdir, "decoder.onnx")

    generate_onnx_graph(params.input, encoder_path, decoder_path, outdir, quant=params.no_quantize)

    try:
        verify_export(params.input, outdir)
    except Exception as e:
        print(e)

    print("Creating archive file...")
    shutil.make_archive(outdir, format="zip", root_dir=outdir)
    print("Done.")


if __name__ == "__main__":
    params = parse_args()
    main(params)
