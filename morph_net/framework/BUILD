load("//devtools/python/blaze:strict.bzl", "py_strict_test")
load("//devtools/python/blaze:pytype.bzl", "pytype_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

pytype_library(
    name = "generic_regularizers",
    srcs = ["generic_regularizers.py"],
    deps = [
    ],
)

py_library(
    name = "batch_norm_source_op_handler",
    srcs = ["batch_norm_source_op_handler.py"],
    tags = ["notap"],
    deps = [
        ":grouping_op_handler",
        ":tpu_util",
        "//third_party/py/morph_net/op_regularizers:gamma_l1_regularizer",
    ],
)

py_test(
    name = "batch_norm_source_op_handler_test",
    srcs = ["batch_norm_source_op_handler_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":batch_norm_source_op_handler",
        ":op_regularizer_manager",
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "//third_party/py/mock",
        "//third_party/py/tensorflow",
        "//third_party/tensorflow/contrib/framework:framework_py",
        "//third_party/tensorflow/contrib/layers:layers_py",
    ],
)

py_library(
    name = "concat_op_handler",
    srcs = ["concat_op_handler.py"],
    deps = [
        ":grouping_op_handler",
        ":op_handler",
        ":op_handler_util",
    ],
)

py_test(
    name = "concat_op_handler_test",
    srcs = ["concat_op_handler_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":concat_op_handler",
        ":op_regularizer_manager",
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "//third_party/py/mock",
        "//third_party/py/tensorflow",
        "//third_party/tensorflow/contrib/framework:framework_py",
        "//third_party/tensorflow/contrib/layers:layers_py",
    ],
)

py_library(
    name = "constant_op_regularizer",
    srcs = ["constant_op_regularizer.py"],
    deps = [
        ":generic_regularizers",
        "//third_party/py/tensorflow",
    ],
)

py_library(
    name = "conv_source_op_handler",
    srcs = ["conv_source_op_handler.py"],
    deps = [":group_lasso_base_op_handler"],
)

py_test(
    name = "conv_source_op_handler_test",
    srcs = ["conv_source_op_handler_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":conv_source_op_handler",
        ":op_regularizer_manager",
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/mock",
        "//third_party/py/tensorflow",
        "//third_party/tensorflow/contrib/layers:layers_py",
    ],
)

py_library(
    name = "conv2d_transpose_source_op_handler",
    srcs = ["conv2d_transpose_source_op_handler.py"],
    deps = [":group_lasso_base_op_handler"],
)

py_library(
    name = "depth_to_space_op_handler",
    srcs = ["depth_to_space_op_handler.py"],
    deps = [
        ":op_handler",
        ":op_handler_util",
    ],
)

py_test(
    name = "depth_to_space_op_handler_test",
    srcs = ["depth_to_space_op_handler_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":depth_to_space_op_handler",
        ":op_regularizer_manager",
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "//third_party/py/mock",
        "//third_party/py/tensorflow",
    ],
)

py_library(
    name = "depthwise_convolution_op_handler",
    srcs = ["depthwise_convolution_op_handler.py"],
    deps = [
        ":grouping_op_handler",
        ":op_handler_util",
    ],
)

py_test(
    name = "depthwise_convolution_op_handler_test",
    srcs = ["depthwise_convolution_op_handler_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":depthwise_convolution_op_handler",
        ":op_regularizer_manager",
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "//third_party/py/mock",
        "//third_party/py/tensorflow",
        "//third_party/tensorflow/contrib/framework:framework_py",
        "//third_party/tensorflow/contrib/layers:layers_py",
    ],
)

py_library(
    name = "group_lasso_base_op_handler",
    srcs = ["group_lasso_base_op_handler.py"],
    deps = [
        ":op_handler",
        ":op_handler_util",
        ":tpu_util",
        "//third_party/py/morph_net/op_regularizers:group_lasso_regularizer",
    ],
)

py_library(
    name = "grouping_op_handler",
    srcs = ["grouping_op_handler.py"],
    deps = [
        ":op_handler",
        ":op_handler_util",
    ],
)

py_test(
    name = "grouping_op_handler_test",
    srcs = ["grouping_op_handler_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":grouping_op_handler",
        ":op_regularizer_manager",
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "//third_party/py/mock",
        "//third_party/py/tensorflow",
        "//third_party/tensorflow/contrib/framework:framework_py",
        "//third_party/tensorflow/contrib/layers:layers_py",
    ],
)

py_library(
    name = "leaf_op_handler",
    srcs = ["leaf_op_handler.py"],
    deps = [
        ":op_handler",
        ":op_handler_util",
    ],
)

py_test(
    name = "leaf_op_handler_test",
    srcs = ["leaf_op_handler_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":leaf_op_handler",
        ":op_regularizer_manager",
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "//third_party/py/mock",
        "//third_party/py/tensorflow",
        "//third_party/tensorflow/contrib/framework:framework_py",
        "//third_party/tensorflow/contrib/layers:layers_py",
    ],
)

py_library(
    name = "matmul_source_op_handler",
    srcs = ["matmul_source_op_handler.py"],
    deps = [
        ":group_lasso_base_op_handler",
        "//third_party/py/tensorflow",
    ],
)

py_library(
    name = "op_handler",
    srcs = ["op_handler.py"],
    deps = [
    ],
)

py_library(
    name = "op_handler_util",
    srcs = ["op_handler_util.py"],
    deps = [
        "//third_party/py/tensorflow",
    ],
)

py_test(
    name = "op_handler_util_test",
    srcs = ["op_handler_util_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":op_handler_util",
        ":op_regularizer_manager",
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "//third_party/py/mock",
        "//third_party/py/tensorflow",
        "//third_party/tensorflow/contrib/framework:framework_py",
        "//third_party/tensorflow/contrib/layers:layers_py",
    ],
)

py_library(
    name = "op_regularizer_manager",
    srcs = ["op_regularizer_manager.py"],
    deps = [
        ":concat_and_slice_regularizers",
        ":constant_op_regularizer",
        ":grouping_regularizers",
        ":op_handler_util",
        "//third_party/py/tensorflow",
        "//third_party/py/typing",
    ],
)

py_test(
    name = "op_regularizer_manager_test",
    srcs = ["op_regularizer_manager_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":batch_norm_source_op_handler",
        ":concat_op_handler",
        ":conv_source_op_handler",
        ":depthwise_convolution_op_handler",
        ":generic_regularizers",
        ":grouping_op_handler",
        ":op_regularizer_manager",
        ":output_non_passthrough_op_handler",
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/morph_net/testing:add_concat_model_stub",
        "//third_party/py/morph_net/testing:grouping_concat_model_stub",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
        "//third_party/tensorflow/contrib/framework:framework_py",
        "//third_party/tensorflow/contrib/layers:layers_py",
    ],
)

py_library(
    name = "output_non_passthrough_op_handler",
    srcs = ["output_non_passthrough_op_handler.py"],
    deps = [
        ":op_handler",
        ":op_handler_util",
    ],
)

py_test(
    name = "output_non_passthrough_op_handler_test",
    srcs = ["output_non_passthrough_op_handler_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":op_regularizer_manager",
        ":output_non_passthrough_op_handler",
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "//third_party/py/mock",
        "//third_party/py/tensorflow",
        "//third_party/tensorflow/contrib/framework:framework_py",
        "//third_party/tensorflow/contrib/layers:layers_py",
    ],
)

py_test(
    name = "matmul_source_op_handler_test",
    srcs = ["matmul_source_op_handler_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":matmul_source_op_handler",
        ":op_regularizer_manager",
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/tensorflow",
    ],
)

py_library(
    name = "logistic_sigmoid_source_op_handler",
    srcs = ["logistic_sigmoid_source_op_handler.py"],
    deps = [
        "//third_party/py/morph_net/framework:grouping_op_handler",
        "//third_party/py/morph_net/framework:tpu_util",
        "//third_party/py/morph_net/op_regularizers:prob_gating_regularizer",
    ],
)

py_test(
    name = "logistic_sigmoid_source_op_handler_test",
    srcs = ["logistic_sigmoid_source_op_handler_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":logistic_sigmoid_source_op_handler",
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "//testing/pybase:parameterized",
        "//third_party/py/mock",
        "//third_party/py/morph_net/framework:op_regularizer_manager",
        "//third_party/py/morph_net/probabilistic_gating:activation_gating",
        "//third_party/py/scipy",
        "//third_party/py/tensorflow",
        "//third_party/tensorflow/contrib/framework:framework_py",
        "//third_party/tensorflow/contrib/layers:layers_py",
    ],
)

py_library(
    name = "probabilistic_grouping_regularizer",
    srcs = ["probabilistic_grouping_regularizer.py"],
    deps = [
        "//third_party/py/morph_net/framework:generic_regularizers",
        "//third_party/py/tensorflow",
    ],
)

py_test(
    name = "probabilistic_grouping_regularizer_test",
    srcs = ["probabilistic_grouping_regularizer_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":probabilistic_grouping_regularizer",
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/morph_net/framework:grouping_regularizers",
        "//third_party/py/morph_net/testing:op_regularizer_stub",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
    ],
)

py_test(
    name = "op_handler_decorator_test",
    srcs = ["op_handler_decorator_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":conv_source_op_handler",
        ":generic_regularizers",
        ":op_handler_decorator",
        ":op_regularizer_manager",
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
    ],
)

pytype_library(
    name = "op_handler_decorator",
    srcs = ["op_handler_decorator.py"],
    deps = [":op_handler"],
)

pytype_library(
    name = "tpu_util",
    srcs = ["tpu_util.py"],
    deps = [
        "//third_party/py/contextlib2",
        "//third_party/py/tensorflow",
    ],
)

pytype_library(
    name = "op_handlers",
    srcs = ["op_handlers.py"],
    strict_deps = True,
    deps = [
        ":concat_op_handler",
        ":depthwise_convolution_op_handler",
        ":grouping_op_handler",
        ":leaf_op_handler",
        ":output_non_passthrough_op_handler",
    ],
)

py_library(
    name = "concat_and_slice_regularizers",
    srcs = ["concat_and_slice_regularizers.py"],
    deps = [
        ":generic_regularizers",
        "//third_party/py/tensorflow",
    ],
)

py_test(
    name = "concat_and_slice_regularizers_test",
    srcs = ["concat_and_slice_regularizers_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":concat_and_slice_regularizers",
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "//third_party/py/morph_net/testing:op_regularizer_stub",
        "//third_party/py/tensorflow",
    ],
)

py_library(
    name = "grouping_regularizers",
    srcs = ["grouping_regularizers.py"],
    deps = [
        ":generic_regularizers",
        "//third_party/py/tensorflow",
    ],
)

py_test(
    name = "grouping_regularizers_test",
    srcs = ["grouping_regularizers_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":grouping_regularizers",
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/morph_net/testing:op_regularizer_stub",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
    ],
)

py_strict_test(
    name = "op_handlers_test",
    srcs = ["op_handlers_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":op_handlers",
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "//third_party/py/tensorflow",
    ],
)

py_strict_test(
    name = "tpu_util_test",
    srcs = ["tpu_util_test.py"],
    python_version = "PY3",
    deps = [
        ":tpu_util",
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/tensorflow",
        "//third_party/tensorflow/contrib/slim",
    ],
)
