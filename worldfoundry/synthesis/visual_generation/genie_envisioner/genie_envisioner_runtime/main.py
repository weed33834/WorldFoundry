import os
import sys
import argparse
from utils import import_custom_class


def main():

    parser = argparse.ArgumentParser(
        description="Run Genie-Envisioner inference from the in-tree runtime."
    )
    parser.add_argument('--config_file', type=str, required=True, help='Path for the config file')
    parser.add_argument('--runner_class_path', type=str, default="runner/ge_inferencer.py")
    parser.add_argument('--runner_class', type=str, default="Inferencer")
    parser.add_argument('--mode', type=str, default="infer", choices=("infer",))
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to trained checkpoint, used in inference stage only')
    parser.add_argument('--n_validation', type=int, default=1, help='num of samples to predict, used in inference stage only')
    parser.add_argument('--n_chunk_action', type=int, default=1, help='num of action chunks to predict, used in action inference stage only')
    parser.add_argument('--output_path', type=str, default=None, help='Path to save outputs, used in inference stage only')
    parser.add_argument('--domain_name', type=str, default="agibotworld", help='Domain name of the validation dataset, used in inference stage only')

    args = parser.parse_args()
    Runner = import_custom_class(
        args.runner_class, args.runner_class_path, 
    )
    

    runner = Runner(args.config_file, output_dir=args.output_path)
    if args.checkpoint_path is not None:
        runner.args.load_weights = True
        runner.args.load_diffusion_model_weights = True
        runner.args.diffusion_model['model_path'] = args.checkpoint_path
    runner.prepare_val_dataset()
    runner.prepare_models()
    runner.infer(
        n_chunk_action=args.n_chunk_action,
        n_validation=args.n_validation,
        domain_name=args.domain_name
    )



if __name__ == "__main__":
    main()
