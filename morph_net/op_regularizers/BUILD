load("//third_party/bazel_rules/rules_python/python:py_library.bzl", "py_library")
load("//third_party/bazel_rules/rules_python/python:py_test.bzl", "py_test")

package(
    default_applicable_licenses = ["//third_party/py/morph_net:license"],
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

py_library(
    name = "gamma_l1_regularizer",
    srcs = ["gamma_l1_regularizer.py"],
    deps = [
        "//third_party/py/morph_net/framework:generic_regularizers",
        "//third_party/py/morph_net/framework:tpu_util",
        "//third_party/py/tensorflow:tensorflow_google",
    ],
)

py_library(
    name = "group_lasso_regularizer",
    srcs = ["group_lasso_regularizer.py"],
    deps = [
        "//third_party/py/morph_net/framework:generic_regularizers",
        "//third_party/py/morph_net/framework:tpu_util",
        "//third_party/py/tensorflow",
    ],
)

py_test(
    name = "group_lasso_regularizer_test",
    srcs = ["group_lasso_regularizer_test.py"],
    deps = [
        ":group_lasso_regularizer",
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
        "//third_party/tensorflow/contrib/framework:framework_py",
        "//third_party/tensorflow/contrib/layers:layers_py",
    ],
)

py_library(
    name = "prob_gating_regularizer",
    srcs = ["prob_gating_regularizer.py"],
    deps = [
        "//third_party/py/morph_net/framework:generic_regularizers",
        "//third_party/py/morph_net/framework:tpu_util",
        "//third_party/py/tensorflow",
    ],
)
