# MorphNet Model Zoo

Lei Mao

NVIDIA

## Introduction

MorphNet algorithm uses Group Lasso regularization for all different kinds of models, or Lasso regularization on Batch Normalization parameters for models containing Batch Normalizations, to reduce the number neurons required for certain training tasks.

This example shows the user how to use various image classification models and get them trained using the MorphNet algorithm.

## Dependencies

```
* Python 3.6
* TensorFlow 1.15
* tqdm 4.36.1+
* Git
* Docker (Optional)
* NVIDIA-Docker (Optional)
```

## Usages

### Docker Container

#### Build Image

Build a Docker image using the Dockerfile provided.

```bash
$ docker build -f Dockerfile -t tensorflow:1.15 .
```

#### Run Container

Run the Docker container using the Docker image just built.

```bash
$ nvidia-docker run -it --rm -v $(pwd):/mnt -p 5001:6006 tensorflow:1.15
```

### Run MorphNet Model Zoo Examples

#### Install MorphNet Package

```bash
$ cd /tmp/
$ git clone https://github.com/google-research/morph-net.git
$ pip install --editable morph-net/
```

#### Run Training

`main.py` is the script to start MorphNet training for selected models. Please use `--base-model-name` to specify the base model for training. For other model parameters, including MorphNet regularization configurations, please check via `python main.py --help`.


```bash
$ python main.py --help
usage: main.py [-h] [--num-epochs NUM_EPOCHS] [--num-classes NUM_CLASSES]
               [--batch-size BATCH_SIZE] [--learning-rate LEARNING_RATE]
               [--base-model-name {ResNet50,ResNet101,ResNet152,ResNet50V2,ResNet101V2,ResNet101V2,ResNet152V2,VGG16,VGG19,Xception,InceptionV3,InceptionResNetV2,MobileNet,MobileNetV2,DenseNet121,DenseNet169,DenseNet201,NASNetLarge,NASNetMobile}]
               [--morphnet-regularizer-algorithm {GroupLasso,Gamma}]
               [--morphnet-target-cost {FLOPs,Latency,ModelSize}]
               [--morphnet-hardware {V100,P100,Others}]
               [--morphnet-regularizer-threshold MORPHNET_REGULARIZER_THRESHOLD]
               [--morphnet-regularization-multiplier MORPHNET_REGULARIZATION_MULTIPLIER]
               [--log-dir LOG_DIR] [--num-cuda-device NUM_CUDA_DEVICE]
               [--random-seed RANDOM_SEED]
               [--main-train-device MAIN_TRAIN_DEVICE]
               [--main-eval-device MAIN_EVAL_DEVICE]

Run MorphNet Algorithm on Image Classification Model Zoo.

optional arguments:
  -h, --help            show this help message and exit
  --num-epochs NUM_EPOCHS
                        The number of epochs for training.
  --num-classes NUM_CLASSES
                        The number of classes for image classification.
  --batch-size BATCH_SIZE
                        Batch size.
  --learning-rate LEARNING_RATE
                        Learning rate.
  --base-model-name {ResNet50,ResNet101,ResNet152,ResNet50V2,ResNet101V2,ResNet101V2,ResNet152V2,VGG16,VGG19,Xception,InceptionV3,InceptionResNetV2,MobileNet,MobileNetV2,DenseNet121,DenseNet169,DenseNet201,NASNetLarge,NASNetMobile}
                        Select base model for image classification.
  --morphnet-regularizer-algorithm {GroupLasso,Gamma}
                        Select MorphNet regularization algorithm.
  --morphnet-target-cost {FLOPs,Latency,ModelSize}
                        Select MorphNet target cost.
  --morphnet-hardware {V100,P100,Others}
                        Select MorphNet hardware.
  --morphnet-regularizer-threshold MORPHNET_REGULARIZER_THRESHOLD
                        Set the threshold [0, 1] for killing neuron layers.
  --morphnet-regularization-multiplier MORPHNET_REGULARIZATION_MULTIPLIER
                        Set MorphNet regularization multiplier for
                        regularization strength. The regularization strength
                        for training equals the regularization multiplier
                        divided by the initial cost of the model. Set this
                        value to zero turns of MorphNet regularization.
  --log-dir LOG_DIR     Log directory for TensorBoard and optimized model
                        architectures.
  --num-cuda-device NUM_CUDA_DEVICE
                        Number of CUDA device to use.
  --random-seed RANDOM_SEED
                        Random seed.
  --main-train-device MAIN_TRAIN_DEVICE
                        The device where the model parameters were located.
  --main-eval-device MAIN_EVAL_DEVICE
                        The device used for model evaluation
```

To use the default settings for training, simply run the following commands in the terminal.

```bash
# Single GPU training
$ python main.py --num-cuda-device 1 --batch-size 256
# Four GPU training
$ python main.py --num-cuda-device 4 --batch-size 1024
```

Currently, the model zoo is using CIFAR10 dataset for training. To use more public datasets in the future, [`tensorflow-datasets`](https://www.tensorflow.org/datasets) library could be employed.

#### Optimized Model Architecture

The model architectures would be saved to `${log-dir}/${datetime}/learned_structure` routinely. A typical ResNet50 example for would be

Before optimization:

```
{
  "conv1_conv/Conv2D": 64,
  "conv2_block1_0_conv/Conv2D": 256,
  "conv2_block1_1_conv/Conv2D": 64,
  "conv2_block1_2_conv/Conv2D": 64,
  "conv2_block1_3_conv/Conv2D": 256,
  "conv2_block2_1_conv/Conv2D": 64,
  "conv2_block2_2_conv/Conv2D": 64,
  "conv2_block2_3_conv/Conv2D": 256,
  "conv2_block3_1_conv/Conv2D": 64,
  "conv2_block3_2_conv/Conv2D": 64,
  "conv2_block3_3_conv/Conv2D": 256,
  "conv3_block1_0_conv/Conv2D": 512,
  "conv3_block1_1_conv/Conv2D": 128,
  "conv3_block1_2_conv/Conv2D": 128,
  "conv3_block1_3_conv/Conv2D": 512,
  "conv3_block2_1_conv/Conv2D": 128,
  "conv3_block2_2_conv/Conv2D": 128,
  "conv3_block2_3_conv/Conv2D": 512,
  "conv3_block3_1_conv/Conv2D": 128,
  "conv3_block3_2_conv/Conv2D": 128,
  "conv3_block3_3_conv/Conv2D": 512,
  "conv3_block4_1_conv/Conv2D": 128,
  "conv3_block4_2_conv/Conv2D": 128,
  "conv3_block4_3_conv/Conv2D": 512,
  "conv4_block1_0_conv/Conv2D": 1024,
  "conv4_block1_1_conv/Conv2D": 256,
  "conv4_block1_2_conv/Conv2D": 256,
  "conv4_block1_3_conv/Conv2D": 1024,
  "conv4_block2_1_conv/Conv2D": 256,
  "conv4_block2_2_conv/Conv2D": 256,
  "conv4_block2_3_conv/Conv2D": 1024,
  "conv4_block3_1_conv/Conv2D": 256,
  "conv4_block3_2_conv/Conv2D": 256,
  "conv4_block3_3_conv/Conv2D": 1024,
  "conv4_block4_1_conv/Conv2D": 256,
  "conv4_block4_2_conv/Conv2D": 256,
  "conv4_block4_3_conv/Conv2D": 1024,
  "conv4_block5_1_conv/Conv2D": 256,
  "conv4_block5_2_conv/Conv2D": 256,
  "conv4_block5_3_conv/Conv2D": 1024,
  "conv4_block6_1_conv/Conv2D": 256,
  "conv4_block6_2_conv/Conv2D": 256,
  "conv4_block6_3_conv/Conv2D": 1024,
  "conv5_block1_0_conv/Conv2D": 2048,
  "conv5_block1_1_conv/Conv2D": 512,
  "conv5_block1_2_conv/Conv2D": 512,
  "conv5_block1_3_conv/Conv2D": 2048,
  "conv5_block2_1_conv/Conv2D": 512,
  "conv5_block2_2_conv/Conv2D": 512,
  "conv5_block2_3_conv/Conv2D": 2048,
  "conv5_block3_1_conv/Conv2D": 512,
  "conv5_block3_2_conv/Conv2D": 512,
  "conv5_block3_3_conv/Conv2D": 2048
}
```

After optimization:

```
{
  "conv1_conv/Conv2D": 13,
  "conv2_block1_0_conv/Conv2D": 64,
  "conv2_block1_1_conv/Conv2D": 25,
  "conv2_block1_2_conv/Conv2D": 4,
  "conv2_block1_3_conv/Conv2D": 64,
  "conv2_block2_1_conv/Conv2D": 0,
  "conv2_block2_2_conv/Conv2D": 0,
  "conv2_block2_3_conv/Conv2D": 64,
  "conv2_block3_1_conv/Conv2D": 0,
  "conv2_block3_2_conv/Conv2D": 0,
  "conv2_block3_3_conv/Conv2D": 64,
  "conv3_block1_0_conv/Conv2D": 54,
  "conv3_block1_1_conv/Conv2D": 13,
  "conv3_block1_2_conv/Conv2D": 7,
  "conv3_block1_3_conv/Conv2D": 54,
  "conv3_block2_1_conv/Conv2D": 0,
  "conv3_block2_2_conv/Conv2D": 0,
  "conv3_block2_3_conv/Conv2D": 54,
  "conv3_block3_1_conv/Conv2D": 0,
  "conv3_block3_2_conv/Conv2D": 0,
  "conv3_block3_3_conv/Conv2D": 54,
  "conv3_block4_1_conv/Conv2D": 0,
  "conv3_block4_2_conv/Conv2D": 0,
  "conv3_block4_3_conv/Conv2D": 54,
  "conv4_block1_0_conv/Conv2D": 69,
  "conv4_block1_1_conv/Conv2D": 51,
  "conv4_block1_2_conv/Conv2D": 42,
  "conv4_block1_3_conv/Conv2D": 69,
  "conv4_block2_1_conv/Conv2D": 0,
  "conv4_block2_2_conv/Conv2D": 0,
  "conv4_block2_3_conv/Conv2D": 69,
  "conv4_block3_1_conv/Conv2D": 0,
  "conv4_block3_2_conv/Conv2D": 0,
  "conv4_block3_3_conv/Conv2D": 69,
  "conv4_block4_1_conv/Conv2D": 4,
  "conv4_block4_2_conv/Conv2D": 16,
  "conv4_block4_3_conv/Conv2D": 69,
  "conv4_block5_1_conv/Conv2D": 6,
  "conv4_block5_2_conv/Conv2D": 8,
  "conv4_block5_3_conv/Conv2D": 69,
  "conv4_block6_1_conv/Conv2D": 0,
  "conv4_block6_2_conv/Conv2D": 0,
  "conv4_block6_3_conv/Conv2D": 69,
  "conv5_block1_0_conv/Conv2D": 102,
  "conv5_block1_1_conv/Conv2D": 0,
  "conv5_block1_2_conv/Conv2D": 0,
  "conv5_block1_3_conv/Conv2D": 102,
  "conv5_block2_1_conv/Conv2D": 0,
  "conv5_block2_2_conv/Conv2D": 0,
  "conv5_block2_3_conv/Conv2D": 102,
  "conv5_block3_1_conv/Conv2D": 24,
  "conv5_block3_2_conv/Conv2D": 3,
  "conv5_block3_3_conv/Conv2D": 102
}
```

#### TensorBoard

To monitor training using TensorBoard, start TensorBoard in the background first by running the follow command in the terminal.

```bash
$ tensorboard --logdir ${log-dir} &
```

Then open browser and go to [`127.0.0.1:5001`](http://127.0.0.1:5001).


#### Customization

It is also possible to use customized model for MorphNet. Simply implement your model similar to Keras application models and plug it into the `MorphNetModel` class. An example of the Keras application model implementation is [ResNet50](https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnet50.py#L142).


### MorphNetModel Implementation

MorphNet model instance is created using `MorphNetModel`.

```python
model = MorphNetModel(base_model=base_model, num_classes=num_classes, learning_rate=learning_rate, batch_size=batch_size, num_gpus=num_gpus, main_train_device=main_train_device, main_eval_device=main_eval_device, morphnet_regularizer_algorithm=morphnet_regularizer_algorithm, morphnet_target_cost=morphnet_target_cost, morphnet_hardware=morphnet_hardware, morphnet_regularizer_threshold=morphnet_regularizer_threshold, morphnet_regularization_strength=morphnet_regularization_strength_dummy, log_dir=log_dir)
```

#### Key Parameters

`base_model`: `tf.keras.applications` model class.

`num_classes`: The number of classes for classification.

`learning_rate`: The learning rate for training.

`batch_size`: The batch size for training. It is the total number of samples for training on all GPUs.

`num_gpus`: The number of GPUs to use for training.

`main_train_device`: The device where the model parameters were located. 

`main_eval_device`: The device used for model evaluation.

`morphnet_regularizer_algorithm`: The MorphNet regularization algorithm. Current supported options: `"GroupLasso"` and `"Gamma"`.

`morphnet_target_cost`: The MorphNet regularization target cost. Current supported costs: `"FLOPs"`, `"Latency"`, and `"ModelSize"`.

`morphnet_hardware`: The MorphNet hardware for `Latency` target cost. Current supported hardware: `"V100"` and `"P100"`

`morphnet_regularizer_threshold`: The channel alive threshold for MorphNet regularization algorithm. 

`morphnet_regularization_strength`: The MorphNet regularization strength. It is dynamic, and you could change the `morphnet_regularization_strength` during training.

`log_dir`: Log directory for TensorBoard and exported model structure files.

#### Key Methods

`model.train(inputs, labels)`: Train model with minibatch. The model will be subject to MorphNet regularization.

`model.test(inputs)`: Get predictions from the test samples.

`model.export_model_config_with_inputs(inputs)`: Export model configurations using the new inputs.

`model.get_model_cost(inputs)`: Get the model cost for inputs.

`model.set_morphnet_regularization_strength(morphnet_regularization_strength)`: Set MorphNet regularization strength during training. It could be changed anytime during training.



### Supported Models

The following models have been tested with all the MorphNet regularization configurations using `test_configs.py`.

|   Model   | Supported? |                                 Notes                                 |
|:---------:|:----------:|:---------------------------------------------------------------------:|
|   ResNet  |      ✓     | ResNet50, ResNet101, ResNet152, ResNet50V2, ResNet101V2,  ResNet152V2 |
|  DenseNet |      ✓     |                 DenseNet121, DenseNet169, DenseNet201                 |
| MobileNet |      ✓     |                         MobileNet, MobileNetV2                        |
|   NASNet  |      ✗     |                       NASNetLarge, NASNetMobile                       |
|    VGG    |      ✓     |                              VGG16, VGG19                             |
| Inception |      ✗     |                     InceptionResNetV2, InceptionV3                    |
|  Xception |      ✓     |                                                                       |


### Reference

* [MorphNet: Fast & Simple Resource-Constrained Structure Learning of Deep Networks](https://arxiv.org/abs/1711.06798)

### To-Do Lists

- [ ] Implement MorphNet model reconstructor for converting the JSON to TensorFlow models
- [ ] Use `kwargs` for `regularizer_fn`
- [x] Multi-GPU training
- [ ] Mixed precision training
- [ ] Support TensorFlow dataset API
- [ ] Support more datasets using TensorFlow dataset API
- [x] Add random seed
- [ ] Add options for optimizers
- [ ] Add assertions to prevent the models without batch normalization layer to use gamma regularizer

