"""
MorphNet Model Zoo

Lei Mao
NVIDIA
https://github.com/leimao

Main script to start MorphNet training for selected models.
"""

import argparse
import tensorflow as tf

from model import MorphNetModel
from utils import set_reproducible_environment, select_keras_base_model, train_epoch, validate_epoch


def main():

    parser = argparse.ArgumentParser(
        description="Run MorphNet Algorithm on Image Classification Model Zoo.")

    num_epochs_default = 1000
    num_classes_default = 10
    batch_size_default = 1024
    base_model_name_default = "ResNet50"
    learning_rate_default = 0.0001
    morphnet_regularizer_algorithm_default = "GroupLasso"
    morphnet_target_cost_default = "FLOPs"
    morphnet_hardware_default = "V100"
    morphnet_regularizer_threshold_default = 1e-2
    morphnet_regularization_multiplier_default = 1000.0
    log_dir_default = "./morphnet_log"
    main_train_device_default = "/cpu:0"
    main_eval_device_default = "/gpu:0"
    num_cuda_device_default = 4
    random_seed_default = 0
    base_model_choices = [
        "ResNet50", "ResNet101", "ResNet152", "ResNet50V2", "ResNet101V2",
        "ResNet101V2", "ResNet152V2", "VGG16", "VGG19", "Xception",
        "InceptionV3", "InceptionResNetV2", "MobileNet", "MobileNetV2",
        "DenseNet121", "DenseNet169", "DenseNet201", "NASNetLarge",
        "NASNetMobile"
    ]
    morphnet_regularizer_algorithm_choices = ["GroupLasso", "Gamma"]
    morphnet_target_cost_choices = ["FLOPs", "Latency", "ModelSize"]
    morphnet_hardware_choices = ["V100", "P100", "Others"]

    parser.add_argument("--num-epochs",
                        type=int,
                        help="The number of epochs for training.",
                        default=num_epochs_default)
    parser.add_argument("--num-classes",
                        type=int,
                        help="The number of classes for image classification.",
                        default=num_classes_default)
    parser.add_argument("--batch-size",
                        type=int,
                        help="Batch size.",
                        default=batch_size_default)
    parser.add_argument("--learning-rate",
                        type=float,
                        help="Learning rate.",
                        default=learning_rate_default)
    parser.add_argument("--base-model-name",
                        type=str,
                        choices=base_model_choices,
                        help="Select base model for image classification.",
                        default=base_model_name_default)
    parser.add_argument("--morphnet-regularizer-algorithm",
                        type=str,
                        choices=morphnet_regularizer_algorithm_choices,
                        help="Select MorphNet regularization algorithm.",
                        default=morphnet_regularizer_algorithm_default)
    parser.add_argument("--morphnet-target-cost",
                        type=str,
                        choices=morphnet_target_cost_choices,
                        help="Select MorphNet target cost.",
                        default=morphnet_target_cost_default)
    parser.add_argument("--morphnet-hardware",
                        type=str,
                        choices=morphnet_hardware_choices,
                        help="Select MorphNet hardware.",
                        default=morphnet_hardware_default)
    parser.add_argument(
        "--morphnet-regularizer-threshold",
        type=float,
        help="Set the threshold [0, 1] for killing neuron layers.",
        default=morphnet_regularizer_threshold_default)
    parser.add_argument(
        "--morphnet-regularization-multiplier",
        type=float,
        help=
        "Set MorphNet regularization multiplier for regularization strength. The regularization strength for training equals the regularization multiplier divided by the initial cost of the model. Set this value to zero turns of MorphNet regularization.",
        default=morphnet_regularization_multiplier_default)
    parser.add_argument(
        "--log-dir",
        type=str,
        help="Log directory for TensorBoard and optimized model architectures.",
        default=log_dir_default)
    parser.add_argument("--num-cuda-device",
                        type=int,
                        help="Number of CUDA device to use.",
                        default=num_cuda_device_default)
    parser.add_argument("--random-seed",
                        type=int,
                        help="Random seed.",
                        default=random_seed_default)
    parser.add_argument(
        "--main-train-device",
        type=str,
        help="The device where the model parameters were located.",
        default=main_train_device_default)
    parser.add_argument("--main-eval-device",
                        type=str,
                        help="The device used for model evaluation",
                        default=main_eval_device_default)

    argv = parser.parse_args()

    num_epochs = argv.num_epochs
    num_classes = argv.num_classes
    batch_size = argv.batch_size
    base_model_name = argv.base_model_name
    learning_rate = argv.learning_rate
    morphnet_regularizer_algorithm = argv.morphnet_regularizer_algorithm
    morphnet_target_cost = argv.morphnet_target_cost
    morphnet_hardware = argv.morphnet_hardware
    morphnet_regularizer_threshold = argv.morphnet_regularizer_threshold
    morphnet_regularization_multiplier = argv.morphnet_regularization_multiplier
    log_dir = argv.log_dir
    num_cuda_device = argv.num_cuda_device
    random_seed = argv.random_seed
    main_train_device = argv.main_train_device
    main_eval_device = argv.main_eval_device

    set_reproducible_environment(random_seed=random_seed)

    (x_train, y_train), (x_valid,
                         y_valid) = tf.keras.datasets.cifar10.load_data()
    # Convert class vectors to binary class matrices.
    y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes)
    y_valid_onehot = tf.keras.utils.to_categorical(y_valid, num_classes)
    image_shape = x_train[1:]
    # Normalize image inputs
    x_train = x_train.astype("float32") / 255.0
    x_valid = x_valid.astype("float32") / 255.0

    base_model = select_keras_base_model(base_model_name=base_model_name)
    morphnet_regularization_strength_dummy = 1e-9
    model = MorphNetModel(
        base_model=base_model,
        num_classes=num_classes,
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_gpus=num_cuda_device,
        main_train_device=main_train_device,
        main_eval_device=main_eval_device,
        morphnet_regularizer_algorithm=morphnet_regularizer_algorithm,
        morphnet_target_cost=morphnet_target_cost,
        morphnet_hardware=morphnet_hardware,
        morphnet_regularizer_threshold=morphnet_regularizer_threshold,
        morphnet_regularization_strength=morphnet_regularization_strength_dummy,
        log_dir=log_dir)

    # Export the unmodified model configures.
    initial_cost = model.get_model_cost(inputs=x_train[:batch_size])
    print("*" * 100)
    print("Initial Model Cost: {:.1f}".format(initial_cost))
    morphnet_regularization_strength = 1.0 / initial_cost * morphnet_regularization_multiplier
    print("Use Regularization Strength: {}".format(
        morphnet_regularization_strength))
    model.set_morphnet_regularization_strength(
        morphnet_regularization_strength=morphnet_regularization_strength)
    print("*" * 100)
    # Export the unmodified model configures.
    model.export_model_config_with_inputs(inputs=x_train[:batch_size])

    for epoch in range(num_epochs):
        validate_epoch(epoch=epoch,
                       model=model,
                       x_valid=x_valid,
                       y_valid_onehot=y_valid_onehot,
                       batch_size=batch_size)
        train_epoch(epoch=epoch,
                    model=model,
                    x_train=x_train,
                    y_train_onehot=y_train_onehot,
                    batch_size=batch_size,
                    shuffle=True,
                    print_batch_info=False)
        # Export the model configure routinely.
        model.export_model_config_with_inputs(inputs=x_train[:batch_size])

    validate_epoch(epoch=num_epochs,
                   model=model,
                   x_valid=x_valid,
                   y_valid_onehot=y_valid_onehot,
                   batch_size=batch_size)

    model.close()

    return 0


if __name__ == "__main__":

    main()
