import argparse
import os

def int_or_float(value):
    if '.' in value:
        return float(value)    
    return int(value)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-location",
        type=str,
        default=os.path.expanduser("~/Documents/data"),
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="openai",
        help="Zero shot model name",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Name of the experiment, for organization purposes only.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ViT-B-32",
        help="The type of model (e.g. RN50, ViT-B-32).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=16,
        help="LoRA rank."
    )
    parser.add_argument(
        "--num-grad-accumulation",
        type=int,
        default=1,
        help="Number of gradient accumulation steps.",
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--wd", type=float, default=0.1, help="Weight decay")
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--openclip-cachedir",
        type=str,
        default=os.path.expanduser("~/openclip-cachedir/open_clip"),
        help="Directory for caching models from OpenCLIP",
    )
    parser.add_argument(
        "--datasets",
        default=None,
        nargs='+',
        help="Subset of datasets to train on."
    )
    parser.add_argument(
        "--no-log",
        default=False,
        action='store_true',
        help="No logging in the results/ folder"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed."
    )
    parser.add_argument(
        "--data-ratio",
        type=int_or_float,
        default=1.,
        help="N of few-shot samples per class"
    )
    parser.add_argument(
        "--fname",
        type=str,
        default=None,
        help="Save file name",
    )
    parser.add_argument(
        "--no-tqdm",
        default=False,
        action="store_true",
        help="Decativate tqdm logging on stderr."
    )
    parser.add_argument(
        "--merge",
        default=False,
        action='store_true',
        help="Will run on the missing datasets from the results.txt specified through fname and complete the specified results.txt file. Needs fname to be specified."
    )
    parser.add_argument(
        "--full-clip",
        default=False,
        action='store_true',
        help="Train both language and vision encoders of CLIP"
    )
    parser.add_argument(
        "--lp-clip",
        default=False,
        action='store_true',
        help="Train PEFT method and the classifier"
    )
    parser.add_argument(
        "--mlp-only",
        default=False,
        action='store_true',
        help="PEFT on mlp layers only."
    )
    parser.add_argument(
        "--attn-only",
        default=False,
        action='store_true',
        help="PEFT on attn layers only."
    )
    parser.add_argument(
        "--base-augs",
        default=False,
        action='store_true',
        help="Use CLIP's training augs"
    )
    parser.add_argument(
        "--strong-augs",
        default=False,
        action='store_true',
        help="Use SimCLR training augs"
    )
    #RandLora args
    parser.add_argument(
        "--rand-lora",
        default=False,
        action='store_true',
        help="Use random lora combinations"
    )
    parser.add_argument(
        "--sparse",
        default=False,
        action='store_true',
        help="Sparse random matrices"
    )
    parser.add_argument(
        "--param-type",
        default="randlora",
        choices=["randlora", "lora", "vera", "nola"],
        help="Which algorithm to train"
    )
    parser.add_argument(
        "--dist-type",
        default="uniform",
        type=str,
        choices=["uniform", "normal"],
        help="Type of distribution used for the random matrices"
    )
    parser.add_argument(
        "--sparcity",
        default=6,
        type=int,
        help="Degree of sparcity for the random matrices. The --sparse argument must be specified as well. The final degree of sparcity will be 2/sparcity. It is recommended not to exceed sqrt(dim) where dim is the smallest dimension of the weight matrices."
    )
    parser.add_argument(
        "--save-weights",
        default=False,
        action='store_true',
        help="Save the whole model"
    )


    ##Args for the loss landscape visualizations
    parser.add_argument(
        "--vis-models",
        default=None,
        nargs='+',
        help="3 saved models to interpolate between"
    )
    
    parsed_args = parser.parse_args()

    return parsed_args
