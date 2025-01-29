load("//devtools/python/blaze:pytype.bzl", "pytype_library")
load("//third_party/bazel_rules/rules_python/python:py_library.bzl", "py_library")
load("//third_party/bazel_rules/rules_python/python:py_test.bzl", "py_test")

package(
    default_applicable_licenses = ["//third_party/py/morph_net:license"],
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

pytype_library(
    name = "structure_exporter",
    srcs = ["structure_exporter.py"],
    deps = [
        "//third_party/py/morph_net/framework:op_regularizer_manager",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
    ],
)

py_test(
    name = "structure_exporter_test",
    srcs = ["structure_exporter_test.py"],
    deps = [
        ":structure_exporter",
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "//third_party/py/absl/flags",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/morph_net/framework:generic_regularizers",
        "//third_party/py/morph_net/framework:op_regularizer_manager",
        "//third_party/py/tensorflow",
        "//third_party/tensorflow/contrib/layers:layers_py",
    ],
)

py_library(
    name = "configurable_ops",
    srcs = [
        "configurable_ops.py",
    ],
    deps = [
        ":structure_exporter",
        "//third_party/py/tensorflow",
        "//third_party/tensorflow/contrib/framework:framework_py",
        "//third_party/tensorflow/contrib/layers:layers_py",
        "//third_party/tensorflow/contrib/slim",
    ],
)

py_test(
    name = "configurable_ops_test",
    size = "medium",
    srcs = [
        "configurable_ops_test.py",
    ],
    tags = ["notap"],
    deps = [
        ":configurable_ops",
        ":test_module",
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "//third_party/py/absl/flags",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/tensorflow",
        "//third_party/tensorflow/contrib/framework:framework_py",
        "//third_party/tensorflow/contrib/layers:layers_py",
        "//third_party/tensorflow/contrib/slim",
        "//third_party/tensorflow/contrib/slim:nets",
    ],
)

py_library(
    name = "test_module",
    srcs = ["test_module.py"],
    deps = ["//third_party/py/tensorflow"],
)
