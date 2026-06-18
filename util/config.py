import argparse
import os


def _path_from_env(env_name, default):
    return os.environ.get(env_name, default)


def _abs_path(path):
    return os.path.abspath(os.path.expanduser(path))


def _named_base_dir(base_dir, name=""):
    if not name:
        return base_dir
    parent = os.path.dirname(base_dir)
    leaf = os.path.basename(base_dir.rstrip(os.sep))
    return os.path.join(parent, leaf + name)


def get_result_dir(args, dataset, run_name):
    return os.path.join(_named_base_dir(args.output_dir, args.name), dataset, run_name)


def get_log_dir(args, phase, dataset):
    return os.path.join(_named_base_dir(args.log_dir, args.name), phase, dataset)


def get_loader_path_kwargs(args):
    return {
        "data_root": args.data_root,
        "cache_dir": args.cache_dir,
        "vipl_root": args.vipl_root,
        "vipl_gt_root": args.vipl_gt_root,
        "vipl_bg_root": args.vipl_bg_root,
    }


def _finalize_paths(args):
    args.data_root = _abs_path(args.data_root)
    args.output_dir = _abs_path(args.output_dir)
    args.log_dir = _abs_path(args.log_dir)
    args.cache_dir = _abs_path(args.cache_dir)

    if args.vipl_root is None:
        args.vipl_root = os.path.join(args.data_root, "VIPL", "RGB_crop")
    if args.vipl_gt_root is None:
        args.vipl_gt_root = os.path.join(args.data_root, "VIPL", "GT")
    if args.vipl_bg_root is None:
        args.vipl_bg_root = os.path.join(args.data_root, "VIPL", "VIPL-HR_MiDaS")

    args.vipl_root = _abs_path(args.vipl_root)
    args.vipl_gt_root = _abs_path(args.vipl_gt_root)
    args.vipl_bg_root = _abs_path(args.vipl_bg_root)

    return args


def get_args():
    
    parser = argparse.ArgumentParser()
    
    # ----------------- General ------------------
    parser.add_argument('--train_dataset', default="", type=str,
                        help="""
                        Options => C: COHFACE, P: PURE, U: UBFC, M: MR-NIRP, V: VIPL-HR,
                        e.g. --dataset="C"  for intra-training/testing on COHFACE
                             --dataset="UP" for cross-training/testing on PURE and UBFC        
                        """)
    parser.add_argument('--test_dataset', default="", type=str,
                        help="Same as above")
    parser.add_argument('--finetune_dataset', default="", type=str,
                        help="Same as above")

    parser.add_argument('--data_root', default=_path_from_env("RPPG_DATA_ROOT", "./data"),
                        type=str, help="Root directory containing public rPPG datasets.")
    parser.add_argument('--output_dir', default=_path_from_env("RPPG_OUTPUT_DIR", "./results"),
                        type=str, help="Directory for checkpoints and generated outputs.")
    parser.add_argument('--log_dir', default=_path_from_env("RPPG_LOG_DIR", "./logs"),
                        type=str, help="Directory for training and evaluation logs.")
    parser.add_argument('--cache_dir', default=_path_from_env("RPPG_CACHE_DIR", "./cache/preprocessed"),
                        type=str, help="Directory for preprocessed frame tensor caches.")
    parser.add_argument('--vipl_root', default=None, type=str,
                        help="VIPL-HR RGB frame root. Defaults to <data_root>/VIPL/RGB_crop.")
    parser.add_argument('--vipl_gt_root', default=None, type=str,
                        help="VIPL-HR ground-truth root. Defaults to <data_root>/VIPL/GT.")
    parser.add_argument('--vipl_bg_root', default=None, type=str,
                        help="VIPL-HR background frame root. Defaults to <data_root>/VIPL/VIPL-HR_MiDaS.")
    
    parser.add_argument('--in_ch', default=3, type=int,
                        help="input channel, you may change to 1 if dataset type is NIR")

    parser.add_argument('--model_S', default=2, type=int,
                        help="spatial dimension of model")
    parser.add_argument('--conv', default="LDC_M", type=str,
                        help="Convolution type for 3DCNN")
        
    parser.add_argument('--bs', default=6, type=int,
                        help="batch size")
    parser.add_argument('--epoch', default=100, type=int,
                        help="training/testing epoch")
    parser.add_argument('--fps', default=30, type=int,
                        help="fps for dataset")
    parser.add_argument('--lr', default=1e-5, type=float,
                        help="learning rate")

    # parser.add_argument('--lr', default=0.005, type=float,
    #                 help="learning rate")
    
    parser.add_argument('--high_pass', default=40, type=int)
    parser.add_argument('--low_pass', default=250, type=int)
    
    # ----------------- Training -----------------
    parser.add_argument('--train_T', default=10, type=int,
                        help="training clip length(seconds))")
    parser.add_argument('--delta_T', default=5, type=int,)
    parser.add_argument('--numSample', default=4, type=int,)

    # ----------------- Testing -----------------
    parser.add_argument('--test_T', default=10, type=int,
                        help="testing clip length(seconds))")
    parser.add_argument('--test_seq', default=60, type=int,
                        help="Number of frames used as the ultra-short test clip.")
    
    parser.add_argument('--inject_noise', action='store_true')
    
    # ----------------- Finetune -----------------
    parser.add_argument('--fix_weight', action='store_true')
    
    
    parser.add_argument('--do_not_preload', action='store_false')
    parser.add_argument('--bg', action='store_true')
    
    parser.add_argument('--do_not_adapt', action='store_true')
    
    # ----------------- VIPL -----------------
    parser.add_argument('--testFold', default=0, type=int)
    
    # ----------------- VIPL -----------------
    parser.add_argument('--MB_size', default=3, type=int,)
    parser.add_argument('--weight_std', default=5, type=int,)

    parser.add_argument('--name', default="", type=str,
                        help="Optional suffix appended to output/log base directories.")
    
    
    return _finalize_paths(parser.parse_args())


def get_name(args, finetune=False, model_name=""):
    

     
    trainName = f"{args.train_dataset}_{args.conv}_train_T{args.train_T}_S{args.model_S}_K{args.numSample}_{model_name}"
    testName  = f"{args.train_dataset}_to_{args.test_dataset}_{args.conv}_test_T{args.test_T}_S{args.model_S}_K{args.numSample}_{model_name}"
    finetuneName = None
    
    if args.inject_noise:
        testName += "_injectNoise"
        
    if finetune:
        
        finetune_dataset = args.finetune_dataset
        test_dataset = args.test_dataset

                    
        finetuneName = f"{args.train_dataset}_finetune_{finetune_dataset}_{args.conv}_train_T{args.train_T}_delta_T_{args.delta_T}_S{args.model_S}_K{args.numSample}_{model_name}"
        testName = f"{args.train_dataset}_finetune_{finetune_dataset}_to_{test_dataset}_{args.conv}_test_T{args.test_T}_delta_T_{args.delta_T}_S{args.model_S}_K{args.numSample}_{model_name}"
        
        if args.fix_weight:
            finetuneName += "_fixWeight"
            
    
    if args.bg:
        trainName += "_bg"
        testName += "_bg"
        if finetune:
            finetuneName += "_bg"
            
            
    if args.do_not_adapt:
        
        testName += "_noAdapt"
    
    
    testName += f"_MB{args.MB_size}_std{args.weight_std}"
    
    return trainName, testName, finetuneName



if __name__ == "__main__":
    args = get_args()
    print(args)
    
