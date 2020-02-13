load("//devtools/python/blaze:pytype.bzl", "pytype_library")
load("//devtools/python/blaze:python3.bzl", "py2and3_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

pytype_library(
    name = "activation_regularizer",
    srcs = ["activation_regularizer.py"],
    deps = [
        ":cost_calculator",
        ":resource_function",
        "//third_party/py/morph_net/framework:batch_norm_source_op_handler",
        "//third_party/py/morph_net/framework:conv2d_transpose_source_op_handler",
        "//third_party/py/morph_net/framework:conv_source_op_handler",
        "//third_party/py/morph_net/framework:generic_regularizers",
        "//third_party/py/morph_net/framework:matmul_source_op_handler",
        "//third_party/py/morph_net/framework:op_handler_decorator",
        "//third_party/py/morph_net/framework:op_handlers",
        "//third_party/py/morph_net/framework:op_regularizer_manager",
        "//third_party/py/tensorflow",
    ],
)

py2and3_test(
    name = "activation_regularizer_test",
    srcs = ["activation_regularizer_test.py"],
    deps = [
        ":activation_regularizer",
        ":resource_function",
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/morph_net/testing:dummy_decorator",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
        "//third_party/tensorflow/contrib/slim",
    ],
)

pytype_library(
    name = "cost_calculator",
    srcs = ["cost_calculator.py"],
    deps = ["//third_party/py/tensorflow"],
)

py2and3_test(
    name = "cost_calculator_test",
    srcs = ["cost_calculator_test.py"],
    deps = [
        ":cost_calculator",
        ":resource_function",
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/morph_net/framework:batch_norm_source_op_handler",
        "//third_party/py/morph_net/framework:concat_op_handler",
        "//third_party/py/morph_net/framework:grouping_op_handler",
        "//third_party/py/morph_net/framework:op_regularizer_manager",
        "//third_party/py/morph_net/framework:output_non_passthrough_op_handler",
        "//third_party/py/morph_net/testing:add_concat_model_stub",
        "//third_party/py/tensorflow",
        "//third_party/tensorflow/contrib/framework:framework_py",
        "//third_party/tensorflow/contrib/layers:layers_py",
    ],
)

pytype_library(
    name = "flop_regularizer",
    srcs = ["flop_regularizer.py"],
    deps = [
        ":cost_calculator",
        ":resource_function",
        "//third_party/py/morph_net/framework:batch_norm_source_op_handler",
        "//third_party/py/morph_net/framework:conv2d_transpose_source_op_handler",
        "//third_party/py/morph_net/framework:conv_source_op_handler",
        "//third_party/py/morph_net/framework:generic_regularizers",
        "//third_party/py/morph_net/framework:matmul_source_op_handler",
        "//third_party/py/morph_net/framework:op_handler_decorator",
        "//third_party/py/morph_net/framework:op_handlers",
        "//third_party/py/morph_net/framework:op_regularizer_manager",
        "//third_party/py/tensorflow",
    ],
)

py2and3_test(
    name = "flop_regularizer_test",
    srcs = ["flop_regularizer_test.py"],
    shard_count = 2,
    deps = [
        ":flop_regularizer",
        ":resource_function",
        "//learning/brain/contrib/slim/nets:resnet_v1",
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/morph_net/testing:dummy_decorator",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
        "//third_party/tensorflow/contrib/layers:layers_py",
        "//third_party/tensorflow/contrib/slim",
        "//third_party/tensorflow/contrib/slim:nets",
    ],
)

pytype_library(
    name = "latency_regularizer",
    srcs = ["latency_regularizer.py"],
    deps = [
        ":cost_calculator",
        ":resource_function",
        "//third_party/py/morph_net/framework:batch_norm_source_op_handler",
        "//third_party/py/morph_net/framework:conv2d_transpose_source_op_handler",
        "//third_party/py/morph_net/framework:conv_source_op_handler",
        "//third_party/py/morph_net/framework:generic_regularizers",
        "//third_party/py/morph_net/framework:matmul_source_op_handler",
        "//third_party/py/morph_net/framework:op_handler_decorator",
        "//third_party/py/morph_net/framework:op_handlers",
        "//third_party/py/morph_net/framework:op_regularizer_manager",
        "//third_party/py/tensorflow",
    ],
)

py2and3_test(
    name = "latency_regularizer_test",
    srcs = ["latency_regularizer_test.py"],
    shard_count = 4,
    deps = [
        ":flop_regularizer",
        ":latency_regularizer",
        ":model_size_regularizer",
        ":resource_function",
        "//learning/brain/contrib/slim/nets:inception",
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/tensorflow",
        "//third_party/tensorflow/contrib/layers:layers_py",
        "//third_party/tensorflow/contrib/slim",
        "//third_party/tensorflow/contrib/slim:nets",
    ],
)

pytype_library(
    name = "model_size_regularizer",
    srcs = ["model_size_regularizer.py"],
    deps = [
        ":cost_calculator",
        ":resource_function",
        "//third_party/py/morph_net/framework:batch_norm_source_op_handler",
        "//third_party/py/morph_net/framework:conv2d_transpose_source_op_handler",
        "//third_party/py/morph_net/framework:conv_source_op_handler",
        "//third_party/py/morph_net/framework:generic_regularizers",
        "//third_party/py/morph_net/framework:matmul_source_op_handler",
        "//third_party/py/morph_net/framework:op_handler_decorator",
        "//third_party/py/morph_net/framework:op_handlers",
        "//third_party/py/morph_net/framework:op_regularizer_manager",
        "//third_party/py/tensorflow",
    ],
)

py2and3_test(
    name = "model_size_regularizer_test",
    srcs = ["model_size_regularizer_test.py"],
    deps = [
        ":model_size_regularizer",
        ":resource_function",
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/morph_net/testing:dummy_decorator",
        "//third_party/py/tensorflow",
        "//third_party/tensorflow/contrib/slim",
    ],
)

pytype_library(
    name = "resource_function",
    srcs = ["resource_function.py"],
    deps = [
        ":cost_calculator",
        "//third_party/py/morph_net/framework:op_handler_util",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
    ],
)

py2and3_test(
    name = "resource_function_test",
    srcs = ["resource_function_test.py"],
    deps = [
        ":resource_function",
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
        "//third_party/tensorflow/contrib/layers:layers_py",
        "//third_party/tensorflow/python:framework_for_generated_wrappers",
    ],
)
