import configargparse

def get_sampling_args():
    parser = configargparse.ArgumentParser()
    parser.add_argument("--max_depth", default=7,type=int)
    # parser.add_argument("--n_samples", default=100**3,type=int)
    parser.add_argument("--n_samples", default=500000,type=int)
    parser.add_argument("--ratio", default=0.4,type=float)
    parser.add_argument("--skip_feature", default=0,type=int)
    parser.add_argument("--angle_threshold_degrees", default=140.0,type=float)

    # Dataset
    parser.add_argument("--output_dir", default="./tmp_output")
    parser.add_argument("--datasets_dir", default="./datasets")
    parser.add_argument("-d", "--dataset", required=True)


    args = parser.parse_args()

    return args

def get_prjection_args():
    parser = configargparse.ArgumentParser()

    parser.add_argument("--singular_value_threshold", default=0.001,type=float)
    parser.add_argument("--margin_threshold", default=0.1,type=float)
    parser.add_argument("--max_depth", default=6,type=int)
    parser.add_argument("--angle_threshold_degrees", default=135.0,type=float)
    
    # Network
    parser.add_argument("-n", "--experiment_name", required=True)
    parser.add_argument("--experiments_dir", default="./experiments")

    # Dataset
    parser.add_argument("--output_dir", default="./tmp_output")
    parser.add_argument("--datasets_dir", default="./datasets")
    parser.add_argument("-d", "--dataset", required=True)

    # Model
    parser.add_argument("--common_layers", default=6, type=int)
    parser.add_argument("--per_head_layers", default=2, type=int)
    parser.add_argument("--common_layer_units", default=256, type=int)
    parser.add_argument("--per_head_units", default=64, type=int)
    parser.add_argument("--n_fourier_basis", default=64, type=int)
    parser.add_argument("--fourier_freq_sigma", default=16.0, type=float)
    parser.add_argument("--model_seed", default=42, type=int)
    # Training
    parser.add_argument("-b", "--batch_size", default=2**13, type=int)
    parser.add_argument("-i", "--total_iterations", default=1000001, type=int)
    parser.add_argument("--learning_rate", default=1e-4, type=float)

    args = parser.parse_args()

    return args

def get_sqef_extraction_args():
    parser = configargparse.ArgumentParser()

    parser.add_argument("--singular_value_threshold", default=0.01,type=float)
    parser.add_argument("--margin_threshold", default=0.19,type=float)
    parser.add_argument("--max_depth", default=7,type=int)
    parser.add_argument("--sampling_depth", default=2,type=int)#need more samples in min depth
    parser.add_argument("--angle_threshold0", default=10.0,type=float)
    parser.add_argument("--angle_threshold", default=15.0,type=float)
    # parser.add_argument("--type_first", default=False,type=bool)

    # Dataset
    parser.add_argument("--input_dir", default="./tmp_output")
    parser.add_argument("--output_dir", default="./final_output")
    parser.add_argument("--datasets_dir", default="./datasets")
    parser.add_argument("-d", "--dataset", required=True)
    parser.add_argument("--debug", default=0,type=int)

    args = parser.parse_args()


    return args

def get_extraction_args():
    parser = configargparse.ArgumentParser()

    parser.add_argument("--singular_value_threshold", default=0.001,type=float)
    parser.add_argument("--margin_threshold", default=0.1,type=float)
    parser.add_argument("--max_depth", default=5,type=int)
    
    # Network
    parser.add_argument("-n", "--experiment_name", required=True)
    parser.add_argument("--experiments_dir", default="./experiments")

    # Dataset
    parser.add_argument("--input_dir", default="./tmp_output")
    parser.add_argument("--output_dir", default="./final_output")
    parser.add_argument("-d", "--dataset", required=True)

    args = parser.parse_args()


    return args

def get_feature_args():
    parser = configargparse.ArgumentParser()

    parser.add_argument("--singular_value_threshold", default=0.001,type=float)
    parser.add_argument("--ratio", default=0.5,type=float)
    
    parser.add_argument("--max_depth", default=5,type=int)
    parser.add_argument("--K", default=64,type=int)
    # Network
    parser.add_argument("-n", "--experiment_name", required=True)
    parser.add_argument("--experiments_dir", default="./experiments")

    # Dataset
    parser.add_argument("--input_dir", default="./tmp_output")
    parser.add_argument("--output_dir", default="./final_output")
    parser.add_argument("--datasets_dir", default="./datasets")
    parser.add_argument("-d", "--dataset", required=True)

    # Model
    parser.add_argument("--common_layers", default=6, type=int)
    parser.add_argument("--per_head_layers", default=2, type=int)
    parser.add_argument("--common_layer_units", default=256, type=int)
    parser.add_argument("--per_head_units", default=64, type=int)
    parser.add_argument("--n_fourier_basis", default=64, type=int)
    parser.add_argument("--fourier_freq_sigma", default=16.0, type=float)
    parser.add_argument("--model_seed", default=42, type=int)
    # Training
    parser.add_argument("-b", "--batch_size", default=2**13, type=int)
    parser.add_argument("-i", "--total_iterations", default=1000001, type=int)
    parser.add_argument("--learning_rate", default=1e-4, type=float)

    args = parser.parse_args()
    return args