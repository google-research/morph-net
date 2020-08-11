load("//devtools/python/blaze:pytype.bzl", "pytype_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

pytype_library(
    name = "structure_exporter",
    srcs = ["structure_exporter.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//third_party/py/morph_net/framework:op_regularizer_manager",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
    ],
)

py_test(
    name = "structure_exporter_test",
    srcs = ["structure_exporter_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
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
        "//third_party/py/enum",
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
    python_version = "PY3",
    srcs_version = "PY3",
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
