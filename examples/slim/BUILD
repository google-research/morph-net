# Description:
#   Contains files for loading, training and evaluating TF-Slim-based models.

load("//third_party/bazel_rules/rules_python/python:py_binary.bzl", "py_binary")
load("//third_party/bazel_rules/rules_python/python:py_library.bzl", "py_library")
load("//third_party/bazel_rules/rules_python/python:py_test.bzl", "py_test")
load("//third_party/bazel_rules/rules_shell/shell:sh_binary.bzl", "sh_binary")

package(
    default_applicable_licenses = ["//third_party/py/morph_net:license"],
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

py_library(
    name = "dataset_utils",
    srcs = ["datasets/dataset_utils.py"],
    strict_deps = False,
    deps = [
        "//third_party/py/six",
        # "//tensorflow",
    ],
)

sh_binary(
    name = "download_and_convert_imagenet",
    srcs = ["datasets/download_and_convert_imagenet.sh"],
    data = [
        "datasets/download_imagenet.sh",
        "datasets/imagenet_2012_validation_synset_labels.txt",
        "datasets/imagenet_lsvrc_2015_synsets.txt",
        "datasets/imagenet_metadata.txt",
        "datasets/preprocess_imagenet_validation_data.py",
        "datasets/process_bounding_boxes.py",
        ":build_imagenet_data",
    ],
)

py_binary(
    name = "build_imagenet_data",
    srcs = ["datasets/build_imagenet_data.py"],
    strict_deps = False,
    deps = [
        # "//numpy",
        # "//tensorflow",
    ],
)

py_library(
    name = "download_and_convert_cifar10",
    srcs = ["datasets/download_and_convert_cifar10.py"],
    strict_deps = False,
    deps = [
        ":dataset_utils",
        # "//numpy",
        # "//tensorflow",
    ],
)

py_library(
    name = "download_and_convert_flowers",
    srcs = ["datasets/download_and_convert_flowers.py"],
    strict_deps = False,
    deps = [
        ":dataset_utils",
        # "//tensorflow",
    ],
)

py_library(
    name = "download_and_convert_mnist",
    srcs = ["datasets/download_and_convert_mnist.py"],
    strict_deps = False,
    deps = [
        ":dataset_utils",
        # "//numpy",
        # "//tensorflow",
    ],
)

py_binary(
    name = "download_and_convert_data",
    srcs = ["download_and_convert_data.py"],
    strict_deps = False,
    deps = [
        ":download_and_convert_cifar10",
        ":download_and_convert_flowers",
        ":download_and_convert_mnist",
        # "//tensorflow",
    ],
)

sh_binary(
    name = "download_mscoco",
    srcs = ["datasets/download_mscoco.sh"],
)

py_binary(
    name = "build_visualwakewords_data",
    srcs = ["datasets/build_visualwakewords_data.py"],
    strict_deps = False,
    deps = [
        ":build_visualwakewords_data_lib",
        # "//tensorflow",
    ],
)

py_library(
    name = "build_visualwakewords_data_lib",
    srcs = ["datasets/build_visualwakewords_data_lib.py"],
    strict_deps = False,
    deps = [
        ":dataset_utils",
        "//third_party/py/PIL:pil",
        # "//tensorflow",
    ],
)

py_library(
    name = "cifar10",
    srcs = ["datasets/cifar10.py"],
    strict_deps = False,
    deps = [
        ":dataset_utils",
        # "//tensorflow",
    ],
)

py_library(
    name = "flowers",
    srcs = ["datasets/flowers.py"],
    strict_deps = False,
    deps = [
        ":dataset_utils",
        # "//tensorflow",
    ],
)

py_library(
    name = "imagenet",
    srcs = ["datasets/imagenet.py"],
    strict_deps = False,
    deps = [
        ":dataset_utils",
        # "//tensorflow",
    ],
)

py_library(
    name = "mnist",
    srcs = ["datasets/mnist.py"],
    strict_deps = False,
    deps = [
        ":dataset_utils",
        # "//tensorflow",
    ],
)

py_library(
    name = "visualwakewords",
    srcs = ["datasets/visualwakewords.py"],
    strict_deps = False,
    deps = [
        ":dataset_utils",
        # "//tensorflow",
    ],
)

py_library(
    name = "dataset_factory",
    srcs = ["datasets/dataset_factory.py"],
    strict_deps = False,
    deps = [
        ":cifar10",
        ":flowers",
        ":imagenet",
        ":mnist",
        ":visualwakewords",
    ],
)

py_library(
    name = "model_deploy",
    srcs = ["deployment/model_deploy.py"],
    strict_deps = False,
    deps = [
        # "//tensorflow",
    ],
)

py_test(
    name = "model_deploy_test",
    srcs = ["deployment/model_deploy_test.py"],
    strict_deps = False,
    deps = [
        ":model_deploy",
        # "//numpy",
        # "//tensorflow",
    ],
)

py_library(
    name = "cifarnet_preprocessing",
    srcs = ["preprocessing/cifarnet_preprocessing.py"],
    strict_deps = False,
    deps = [
        # "//tensorflow",
    ],
)

py_library(
    name = "inception_preprocessing",
    srcs = ["preprocessing/inception_preprocessing.py"],
    strict_deps = False,
    deps = [
        # "//tensorflow",
        # "//tensorflow/python:control_flow_ops",
    ],
)

py_library(
    name = "lenet_preprocessing",
    srcs = ["preprocessing/lenet_preprocessing.py"],
    strict_deps = False,
    deps = [
        # "//tensorflow",
    ],
)

py_library(
    name = "vgg_preprocessing",
    srcs = ["preprocessing/vgg_preprocessing.py"],
    strict_deps = False,
    deps = [
        # "//tensorflow",
    ],
)

py_library(
    name = "preprocessing_factory",
    srcs = ["preprocessing/preprocessing_factory.py"],
    strict_deps = False,
    deps = [
        ":cifarnet_preprocessing",
        ":inception_preprocessing",
        ":lenet_preprocessing",
        ":vgg_preprocessing",
        # "//tensorflow",
    ],
)

# Typical networks definitions.

py_library(
    name = "nets",
    strict_deps = False,
    deps = [
        ":alexnet",
        ":cifarnet",
        ":cyclegan",
        ":i3d",
        ":inception",
        ":lenet",
        ":mobilenet",
        ":nasnet",
        ":overfeat",
        ":pix2pix",
        ":pnasnet",
        ":resnet_v1",
        ":resnet_v2",
        ":s3dg",
        ":vgg",
    ],
)

py_library(
    name = "alexnet",
    srcs = ["nets/alexnet.py"],
    strict_deps = False,
    deps = [
        # "//tensorflow",
    ],
)

py_test(
    name = "alexnet_test",
    size = "medium",
    srcs = ["nets/alexnet_test.py"],
    strict_deps = False,
    deps = [
        ":alexnet",
        # "//tensorflow",
    ],
)

py_library(
    name = "cifarnet",
    srcs = ["nets/cifarnet.py"],
    strict_deps = False,
    deps = [
        # "//tensorflow",
    ],
)

py_library(
    name = "cyclegan",
    srcs = ["nets/cyclegan.py"],
    strict_deps = False,
    deps = [
        # "//numpy",
        # "//tensorflow",
    ],
)

py_test(
    name = "cyclegan_test",
    srcs = ["nets/cyclegan_test.py"],
    shard_count = 3,
    strict_deps = False,
    deps = [
        ":cyclegan",
        # "//tensorflow",
    ],
)

py_library(
    name = "dcgan",
    srcs = ["nets/dcgan.py"],
    strict_deps = False,
    deps = [
        # "//tensorflow",
    ],
)

py_test(
    name = "dcgan_test",
    srcs = ["nets/dcgan_test.py"],
    shard_count = 3,
    strict_deps = False,
    deps = [
        ":dcgan",
        # "//tensorflow",
    ],
)

py_library(
    name = "i3d",
    srcs = ["nets/i3d.py"],
    strict_deps = False,
    deps = [
        ":i3d_utils",
        ":s3dg",
        # "//tensorflow",
    ],
)

py_test(
    name = "i3d_test",
    size = "large",
    srcs = ["nets/i3d_test.py"],
    shard_count = 3,
    strict_deps = False,
    deps = [
        ":i3d",
        # "//tensorflow",
    ],
)

py_library(
    name = "i3d_utils",
    srcs = ["nets/i3d_utils.py"],
    strict_deps = False,
    deps = [
        # "//tensorflow",
    ],
)

py_library(
    name = "inception",
    srcs = ["nets/inception.py"],
    strict_deps = False,
    deps = [
        ":inception_resnet_v2",
        ":inception_v1",
        ":inception_v2",
        ":inception_v3",
        ":inception_v4",
    ],
)

py_library(
    name = "inception_utils",
    srcs = ["nets/inception_utils.py"],
    strict_deps = False,
    deps = [
        # "//tensorflow",
    ],
)

py_library(
    name = "inception_v1",
    srcs = ["nets/inception_v1.py"],
    strict_deps = False,
    deps = [
        ":inception_utils",
        # "//tensorflow",
    ],
)

py_library(
    name = "inception_v2",
    srcs = ["nets/inception_v2.py"],
    strict_deps = False,
    deps = [
        ":inception_utils",
        # "//tensorflow",
    ],
)

py_library(
    name = "inception_v3",
    srcs = ["nets/inception_v3.py"],
    strict_deps = False,
    deps = [
        ":inception_utils",
        # "//tensorflow",
    ],
)

py_library(
    name = "inception_v4",
    srcs = ["nets/inception_v4.py"],
    strict_deps = False,
    deps = [
        ":inception_utils",
        # "//tensorflow",
    ],
)

py_library(
    name = "inception_resnet_v2",
    srcs = ["nets/inception_resnet_v2.py"],
    strict_deps = False,
    deps = [
        # "//tensorflow",
    ],
)

py_test(
    name = "inception_v1_test",
    size = "large",
    srcs = ["nets/inception_v1_test.py"],
    shard_count = 3,
    strict_deps = False,
    deps = [
        ":inception",
        # "//numpy",
        # "//tensorflow",
    ],
)

py_test(
    name = "inception_v2_test",
    size = "large",
    srcs = ["nets/inception_v2_test.py"],
    shard_count = 3,
    strict_deps = False,
    deps = [
        ":inception",
        # "//numpy",
        # "//tensorflow",
    ],
)

py_test(
    name = "inception_v3_test",
    size = "large",
    srcs = ["nets/inception_v3_test.py"],
    shard_count = 3,
    strict_deps = False,
    deps = [
        ":inception",
        # "//numpy",
        # "//tensorflow",
    ],
)

py_test(
    name = "inception_v4_test",
    size = "large",
    srcs = ["nets/inception_v4_test.py"],
    shard_count = 3,
    strict_deps = False,
    deps = [
        ":inception",
        # "//tensorflow",
    ],
)

py_test(
    name = "inception_resnet_v2_test",
    size = "large",
    srcs = ["nets/inception_resnet_v2_test.py"],
    shard_count = 3,
    strict_deps = False,
    deps = [
        ":inception",
        # "//tensorflow",
    ],
)

py_library(
    name = "lenet",
    srcs = ["nets/lenet.py"],
    strict_deps = False,
    deps = [
        # "//tensorflow",
    ],
)

py_library(
    name = "mobilenet_v1",
    srcs = ["nets/mobilenet_v1.py"],
    strict_deps = False,
    deps = [
        # "//tensorflow",
    ],
)

py_library(
    name = "mobilenet_v2",
    srcs = glob(["nets/mobilenet/*.py"]),
    strict_deps = False,
    deps = [
        # "//tensorflow",
    ],
)

py_test(
    name = "mobilenet_v2_test",
    srcs = ["nets/mobilenet/mobilenet_v2_test.py"],
    strict_deps = False,
    deps = [
        ":mobilenet",
        # "//tensorflow",
    ],
)

py_library(
    name = "mobilenet",
    strict_deps = False,
    deps = [
        ":mobilenet_v1",
        ":mobilenet_v2",
    ],
)

py_test(
    name = "mobilenet_v1_test",
    size = "large",
    srcs = ["nets/mobilenet_v1_test.py"],
    shard_count = 3,
    strict_deps = False,
    deps = [
        ":mobilenet_v1",
        # "//numpy",
        # "//tensorflow",
    ],
)

py_binary(
    name = "mobilenet_v1_train",
    srcs = ["nets/mobilenet_v1_train.py"],
    strict_deps = False,
    deps = [
        ":dataset_factory",
        ":mobilenet_v1",
        ":preprocessing_factory",
        # "//tensorflow",
    ],
)

py_binary(
    name = "mobilenet_v1_eval",
    srcs = ["nets/mobilenet_v1_eval.py"],
    strict_deps = False,
    deps = [
        ":dataset_factory",
        ":mobilenet_v1",
        ":preprocessing_factory",
        # "//tensorflow",
    ],
)

py_library(
    name = "nasnet_utils",
    srcs = ["nets/nasnet/nasnet_utils.py"],
    strict_deps = False,
    deps = [
        # "//tensorflow",
    ],
)

py_library(
    name = "nasnet",
    srcs = ["nets/nasnet/nasnet.py"],
    strict_deps = False,
    deps = [
        ":nasnet_utils",
        # "//tensorflow",
    ],
)

py_test(
    name = "nasnet_utils_test",
    size = "medium",
    srcs = ["nets/nasnet/nasnet_utils_test.py"],
    strict_deps = False,
    deps = [
        ":nasnet_utils",
        # "//tensorflow",
    ],
)

py_test(
    name = "nasnet_test",
    size = "large",
    srcs = ["nets/nasnet/nasnet_test.py"],
    shard_count = 10,
    strict_deps = False,
    deps = [
        ":nasnet",
        # "//tensorflow",
    ],
)

py_library(
    name = "pnasnet",
    srcs = ["nets/nasnet/pnasnet.py"],
    strict_deps = False,
    deps = [
        ":nasnet",
        ":nasnet_utils",
        # "//tensorflow",
    ],
)

py_test(
    name = "pnasnet_test",
    size = "large",
    srcs = ["nets/nasnet/pnasnet_test.py"],
    shard_count = 4,
    strict_deps = False,
    deps = [
        ":pnasnet",
        # "//tensorflow",
    ],
)

py_library(
    name = "overfeat",
    srcs = ["nets/overfeat.py"],
    strict_deps = False,
    deps = [
        # "//tensorflow",
    ],
)

py_test(
    name = "overfeat_test",
    size = "medium",
    srcs = ["nets/overfeat_test.py"],
    strict_deps = False,
    deps = [
        ":overfeat",
        # "//tensorflow",
    ],
)

py_library(
    name = "pix2pix",
    srcs = ["nets/pix2pix.py"],
    strict_deps = False,
    deps = [
        # "//tensorflow",
    ],
)

py_test(
    name = "pix2pix_test",
    srcs = ["nets/pix2pix_test.py"],
    strict_deps = False,
    deps = [
        ":pix2pix",
        # "//tensorflow",
    ],
)

py_library(
    name = "resnet_utils",
    srcs = ["nets/resnet_utils.py"],
    strict_deps = False,
    deps = [
        # "//tensorflow",
    ],
)

py_library(
    name = "resnet_v1",
    srcs = ["nets/resnet_v1.py"],
    strict_deps = False,
    deps = [
        ":resnet_utils",
        # "//tensorflow",
    ],
)

py_test(
    name = "resnet_v1_test",
    size = "medium",
    srcs = ["nets/resnet_v1_test.py"],
    shard_count = 2,
    strict_deps = False,
    deps = [
        ":resnet_utils",
        ":resnet_v1",
        "//third_party/py/six",
        # "//numpy",
        # "//tensorflow",
    ],
)

py_library(
    name = "resnet_v2",
    srcs = ["nets/resnet_v2.py"],
    strict_deps = False,
    deps = [
        ":resnet_utils",
        # "//tensorflow",
    ],
)

py_test(
    name = "resnet_v2_test",
    size = "medium",
    srcs = ["nets/resnet_v2_test.py"],
    shard_count = 2,
    strict_deps = False,
    deps = [
        ":resnet_utils",
        ":resnet_v2",
        # "//numpy",
        # "//tensorflow",
    ],
)

py_library(
    name = "s3dg",
    srcs = ["nets/s3dg.py"],
    strict_deps = False,
    deps = [
        ":i3d_utils",
        # "//tensorflow",
    ],
)

py_test(
    name = "s3dg_test",
    size = "large",
    srcs = ["nets/s3dg_test.py"],
    shard_count = 3,
    strict_deps = False,
    deps = [
        ":s3dg",
        # "//tensorflow",
    ],
)

py_library(
    name = "vgg",
    srcs = ["nets/vgg.py"],
    strict_deps = False,
    deps = [
        # "//tensorflow",
    ],
)

py_test(
    name = "vgg_test",
    size = "medium",
    srcs = ["nets/vgg_test.py"],
    strict_deps = False,
    deps = [
        ":vgg",
        # "//tensorflow",
    ],
)

py_library(
    name = "nets_factory",
    srcs = ["nets/nets_factory.py"],
    strict_deps = False,
    deps = [
        ":nets",
        # "//tensorflow",
    ],
)

py_test(
    name = "nets_factory_test",
    size = "large",
    srcs = ["nets/nets_factory_test.py"],
    shard_count = 3,
    strict_deps = False,
    deps = [
        ":nets_factory",
        # "//tensorflow",
    ],
)

py_library(
    name = "train_image_classifier_lib",
    srcs = ["train_image_classifier.py"],
    strict_deps = False,
    deps = [
        ":dataset_factory",
        ":model_deploy",
        ":nets_factory",
        ":preprocessing_factory",
        # "//tensorflow",
    ],
)

py_binary(
    name = "train_image_classifier",
    srcs = ["train_image_classifier.py"],
    strict_deps = False,
    # WARNING: not supported in bazel; will be commented out by copybara.
    # paropts = ["--compress"],
    deps = [
        ":train_image_classifier_lib",
    ],
)

py_library(
    name = "eval_image_classifier_lib",
    srcs = ["eval_image_classifier.py"],
    strict_deps = False,
    deps = [
        ":dataset_factory",
        ":nets_factory",
        ":preprocessing_factory",
        # "//tensorflow",
    ],
)

py_binary(
    name = "eval_image_classifier",
    srcs = ["eval_image_classifier.py"],
    strict_deps = False,
    deps = [
        ":eval_image_classifier_lib",
    ],
)

py_binary(
    name = "export_inference_graph",
    srcs = ["export_inference_graph.py"],
    strict_deps = False,
    # WARNING: not supported in bazel; will be commented out by copybara.
    # paropts = ["--compress"],
    deps = [":export_inference_graph_lib"],
)

py_library(
    name = "export_inference_graph_lib",
    srcs = ["export_inference_graph.py"],
    strict_deps = False,
    deps = [
        ":dataset_factory",
        ":nets_factory",
        # "//tensorflow",
        # "//tensorflow/python:platform",
    ],
)

py_test(
    name = "export_inference_graph_test",
    size = "medium",
    srcs = ["export_inference_graph_test.py"],
    strict_deps = False,
    tags = [
        "manual",
    ],
    deps = [
        ":export_inference_graph_lib",
        # "//tensorflow",
        # "//tensorflow/python:platform",
    ],
)
