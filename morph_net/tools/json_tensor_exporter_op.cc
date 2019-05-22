#include "third_party/tensorflow/core/framework/op.h"


REGISTER_OP("JsonTensorExporter")
    .Attr("filename: string")
    .Attr("T: {float, double, int32, bool}")
    .Attr("N: int")
    .Attr("keys: list(string)")
    .Input("values: N * T")
    .Input("save: bool")
    .Doc(R"doc(
Saves the content of tensors on file as JSON dictionary.

filename: Filename to which the JSON is to be saved.
N: Number of tensors expected.
keys: The list of keys of the dictionary. Must be of length N.
values: A list of tensors, will be the respective values. The order of the
  values is expected to match that of the keys. Must be of length N. Currently
  only vectors and scalars (rank 1 and 0) are supported.
save: If false, the op would be a no-op. This mechanism is introduced because
  tf.cond can execute both the if and the else, and we don't want to write files
  unnecessarily.
)doc");
