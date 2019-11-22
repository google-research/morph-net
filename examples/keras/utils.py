"""
MorphNet Model Zoo

Lei Mao
NVIDIA
https://github.com/leimao

Utility functions and classes for MorphNet model zoo.
"""

import tensorflow as tf
import numpy as np
from tqdm import tqdm, trange


def select_keras_base_model(base_model_name):
    """
    Select base model architecture from Keras model zoo.
    https://keras.io/applications/
    Args:
        base_model_name: The base model name. String.
    Returns:
        Keras model class.
    """
    # This has to be set before the Keras model was created.
    # Otherwise the Batch normalization layer in the model might not be compatible with the MorphNet library.
    tf.keras.backend.set_learning_phase(1)

    if base_model_name == "ResNet50":
        base_model = tf.keras.applications.resnet.ResNet50
    elif base_model_name == "ResNet101":
        base_model = tf.keras.applications.resnet.ResNet101
    elif base_model_name == "ResNet152":
        base_model = tf.keras.applications.resnet.ResNet152
    elif base_model_name == "ResNet50V2":
        base_model = tf.keras.applications.resnet_v2.ResNet50V2
    elif base_model_name == "ResNet101V2":
        base_model = tf.keras.applications.resnet_v2.ResNet101V2
    elif base_model_name == "ResNet152V2":
        base_model = tf.keras.applications.resnet_v2.ResNet152V2
    elif base_model_name == "VGG16":
        base_model = tf.keras.applications.vgg16.VGG16
    elif base_model_name == "VGG19":
        base_model = tf.keras.applications.vgg19.VGG19
    elif base_model_name == "Xception":
        base_model = tf.keras.applications.xception.Xception
    elif base_model_name == "InceptionV3":
        base_model = tf.keras.applications.inception_v3.InceptionV3
    elif base_model_name == "InceptionResNetV2":
        base_model = tf.keras.applications.inception_resnet_v2.InceptionResNetV2
    elif base_model_name == "MobileNet":
        base_model = tf.keras.applications.mobilenet.MobileNet
    elif base_model_name == "MobileNetV2":
        base_model = tf.keras.applications.mobilenet_v2.MobileNetV2
    elif base_model_name == "DenseNet121":
        base_model = tf.keras.applications.densenet.DenseNet121
    elif base_model_name == "DenseNet169":
        base_model = tf.keras.applications.densenet.DenseNet169
    elif base_model_name == "DenseNet201":
        base_model = tf.keras.applications.densenet.DenseNet201
    elif base_model_name == "NASNetLarge":
        base_model = tf.keras.applications.nasnet.NASNetLarge
    elif base_model_name == "NASNetMobile":
        base_model = tf.keras.applications.nasnet.NASNetMobile
    else:
        raise Exception("Unsupported Base Model!")

    return base_model


def set_reproducible_environment(random_seed=None):
    """
    Set a reproducible environment using random see.
    Args:
        random_seed: Random seed. Integer.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        tf.set_random_seed(random_seed)


def validate_epoch(epoch, model, x_valid, y_valid_onehot, batch_size):
    """
    Validating the model using an epoch of the validation dataset.
    Args:
        epoch: Epoch ID. Interger.
        model: MorphNetModel instance.
        x_valid: Input RGB images from validation dataset. Numpy array.
        y_valid_onehot: One-hot encoded labels from validation dataset. Numpy array.
        batch_size: Batch size. Integer.
    """
    validation_idx = np.arange(len(x_valid))
    mini_batch_idx = [
        validation_idx[k:k + batch_size]
        for k in range(0, len(x_valid), batch_size)
    ]
    tqdm_iterator = trange(len(mini_batch_idx),
                           desc="Validation Epoch: {}".format(epoch))
    num_validation_samples = 0
    num_correct_predictions = 0
    total_cost = 0

    for i in tqdm_iterator:
        idx = mini_batch_idx[i]
        accuracy, cost = model.validate(inputs=x_valid[idx],
                                        labels=y_valid_onehot[idx])
        num_correct_predictions += accuracy * len(idx)
        num_validation_samples += len(idx)
        total_cost += cost

    validation_acc = num_correct_predictions / num_validation_samples
    cost_avg = total_cost / len(mini_batch_idx)
    print("Epoch: {}, Validation Acc: {:.4f}, Cost Avg: {:.4f}".format(
        epoch, validation_acc, cost_avg))


def train_epoch(epoch,
                model,
                x_train,
                y_train_onehot,
                batch_size,
                shuffle=True,
                print_batch_info=False):
    """
    Training the model using an epoch of the training dataset.
    Args:
        epoch: Epoch ID. Interger.
        model: MorphNetModel instance.
        x_train: Input RGB images from training dataset. Numpy array.
        y_train_onehot: One-hot encoded labels from training dataset. Numpy array.
        batch_size: Batch size. Integer.
        shuffle: Whether to shuffle the training dataset before each epoch. Boolean.
        print_batch_info: Whether to print the training statistics for each batch. Boolean.
    """
    train_idx = np.arange(len(x_train))
    if shuffle == True:
        np.random.shuffle(train_idx)
    mini_batch_idx = [
        train_idx[k:k + batch_size] for k in range(0, len(x_train), batch_size)
    ]
    tqdm_iterator = trange(len(mini_batch_idx),
                           desc="Train Epoch: {}".format(epoch))
    num_train_samples = 0
    num_correct_predictions = 0
    total_cost = 0
    total_loss = 0

    for i in tqdm_iterator:
        idx = mini_batch_idx[i]
        loss, accuracy, cost = model.train(inputs=x_train[idx],
                                           labels=y_train_onehot[idx])
        num_correct_predictions += accuracy * len(idx)
        num_train_samples += len(idx)
        total_cost += cost
        total_loss += loss * len(idx)
        if print_batch_info == True:
            train_batch_acc = accuracy
            tqdm.write(
                "Epoch: {}, Batch: {}, Train Acc: {:.4f}, Loss: {:.4f}, Cost: {:.4f}"
                .format(epoch, i, train_batch_acc, loss, cost))

    train_acc = num_correct_predictions / num_train_samples
    cost_avg = total_cost / len(mini_batch_idx)
    loss_avg = total_loss / len(x_train)
    print("Epoch: {}, Train Acc: {:.4f}, Loss Avg: {:.4f}, Cost Avg: {:.4f}".
          format(epoch, train_acc, loss_avg, cost_avg))
