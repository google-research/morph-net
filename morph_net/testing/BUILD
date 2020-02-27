package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "op_regularizer_stub",
    testonly = 1,
    srcs = ["op_regularizer_stub.py"],
    deps = [
        "//third_party/py/morph_net/framework:generic_regularizers",
        "//third_party/py/tensorflow",
        "//third_party/tensorflow/contrib/layers:layers_py",
    ],
)

py_library(
    name = "grouping_concat_model_stub",
    testonly = 1,
    srcs = ["grouping_concat_model_stub.py"],
    deps = [
        ":op_regularizer_stub",
        "//third_party/py/tensorflow",
        "//third_party/tensorflow/contrib/layers:layers_py",
    ],
)

py_library(
    name = "add_concat_model_stub",
    testonly = 1,
    srcs = ["add_concat_model_stub.py"],
    deps = [
        ":op_regularizer_stub",
        "//third_party/py/tensorflow",
        "//third_party/tensorflow/contrib/layers:layers_py",
    ],
)

py_library(
    name = "ladder_model_stub",
    testonly = 1,
    srcs = ["ladder_model_stub.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":op_regularizer_stub",
        "//third_party/py/tensorflow",
        "//third_party/tensorflow/contrib/layers:layers_py",
    ],
)

py_library(
    name = "dummy_decorator",
    testonly = 1,
    srcs = ["dummy_decorator.py"],
    deps = [
        "//third_party/py/morph_net/framework:generic_regularizers",
    ],
)
