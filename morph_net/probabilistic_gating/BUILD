# load("//devtools/python/blaze:strict.bzl", "py_strict_test")

load("//third_party/bazel_rules/rules_python/python:py_library.bzl", "py_library")
load("//third_party/bazel_rules/rules_python/python:py_test.bzl", "py_test")

package(
    default_applicable_licenses = ["//third_party/py/morph_net:license"],
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

py_library(
    name = "activation_gating",
    srcs = ["activation_gating.py"],
    deps = [
        "//third_party/py/morph_net/framework:tpu_util",
        "//third_party/py/tensorflow",
        "//third_party/py/tensorflow_probability",
        "//third_party/tensorflow/contrib/layers:layers_py",
        "//third_party/tensorflow/python/framework:for_generated_wrappers",
    ],
)

py_test(
    name = "activation_gating_test",
    srcs = ["activation_gating_test.py"],
    deps = [
        ":activation_gating",
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
        "//third_party/tensorflow/contrib/layers:layers_py",
    ],
)
