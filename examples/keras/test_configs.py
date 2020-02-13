"""
MorphNet Model Zoo

Lei Mao
NVIDIA
https://github.com/leimao

Test script for MorphNetModel training configurations and model zoo.
"""

import subprocess
import os
import itertools


def test_combinations(log_dir="./morphnet_log",
                      csv_filename="morphnet_test_results.csv",
                      num_cuda_device=4):
    """
    Test the MorphNet model zoo configuration combinations. Most of the networks and MorphNet regularization configurations have been supported.
    Args: 
        log_dir: Directory to log the test results. String.
        csv_filename: CSV filename to save the test results. String.
        num_cuda_device: Number of the CUDA device used for testing. Integer.
    Returns:
        successes: A list of the tuples of the configurations which MorphNet model zoo ran successfully.
        failures: A list of the tuples of the configurations which MorphNet model zoo failed to run.
    """

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    fhand = open(os.path.join(log_dir, csv_filename), "w")
    fhand.write(
        "base_model,morphnet_regularizer_algorithm,morphnet_target_cost,morphnet_hardware,success\n"
    )
    fhand.close()

    successes = []
    failures = []

    num_epochs_default = 1
    num_classes_default = 10
    batch_size_default = 1024
    learning_rate_default = 0.0001
    morphnet_regularizer_threshold_default = 1e-2
    morphnet_regularization_multiplier_default = 1000.0
    log_dir_default = "./morphnet_log"
    num_cuda_device_default = 4
    base_model_choices = [
        "ResNet50", "ResNet101", "ResNet152", "ResNet50V2", "ResNet101V2",
        "ResNet101V2", "ResNet152V2", "VGG16", "VGG19", "Xception",
        "InceptionV3", "InceptionResNetV2", "MobileNet", "MobileNetV2",
        "DenseNet121", "DenseNet169", "DenseNet201", "NASNetLarge",
        "NASNetMobile"
    ]
    morphnet_regularizer_algorithm_choices = ["GroupLasso", "Gamma"]
    morphnet_target_cost_choices = ["FLOPs", "Latency", "ModelSize"]
    morphnet_hardware_choices = ["V100", "P100"]

    with open(os.path.join(log_dir, csv_filename), "a", buffering=1) as fhand:

        for base_model, morphnet_regularizer_algorithm, morphnet_target_cost, morphnet_hardware in itertools.product(
                base_model_choices, morphnet_regularizer_algorithm_choices,
                morphnet_target_cost_choices, morphnet_hardware_choices):

            print(
                "Testing MorphNet Algorithm Combinations [{}, {}, {}, {}] ...".
                format(base_model, morphnet_regularizer_algorithm,
                       morphnet_target_cost, morphnet_hardware))
            shell_command = "python morphnet_model_zoo.py --num-epochs {num_epoch} --num-classes {num_classes} --batch-size {batch_size} --learning-rate {learning_rate} --base-model-name {base_model_name} --morphnet-regularizer-algorithm {morphnet_regularizer_algorithm} --morphnet-target-cost {morphnet_target_cost} --morphnet-hardware {morphnet_hardware} --morphnet-regularizer-threshold {morphnet_regularizer_threshold} --morphnet-regularization-multiplier {morphnet_regularization_multiplier} --log-dir {log_dir} --num-cuda-device {num_cuda_device}".format(
                num_epoch=num_epochs_default,
                num_classes=num_classes_default,
                batch_size=batch_size_default,
                base_model_name=base_model,
                learning_rate=learning_rate_default,
                morphnet_regularizer_algorithm=morphnet_regularizer_algorithm,
                morphnet_target_cost=morphnet_target_cost,
                morphnet_hardware=morphnet_hardware,
                morphnet_regularizer_threshold=
                morphnet_regularizer_threshold_default,
                morphnet_regularization_multiplier=
                morphnet_regularization_multiplier_default,
                log_dir=log_dir_default,
                num_cuda_device=num_cuda_device_default)

            try:
                process = subprocess.run(shell_command,
                                         shell=True,
                                         check=True,
                                         stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE)
                successes.append((base_model, morphnet_regularizer_algorithm,
                                  morphnet_target_cost, morphnet_hardware))
                fhand.write("{},{},{},{},{}\n".format(
                    base_model, morphnet_regularizer_algorithm,
                    morphnet_target_cost, morphnet_hardware, "True"))
                print("Trial Successful!")
            except subprocess.CalledProcessError as exception:
                failures.append((base_model, morphnet_regularizer_algorithm,
                                 morphnet_target_cost, morphnet_hardware))
                fhand.write("{},{},{},{},{}\n".format(
                    base_model, morphnet_regularizer_algorithm,
                    morphnet_target_cost, morphnet_hardware, "False"))
                print("Trial Failed!")
                print(exception.stderr)

    return successes, failures


if __name__ == "__main__":

    test_combinations(log_dir="./morphnet_log",
                      csv_filename="morphnet_test_results.csv",
                      num_cuda_device=4)
