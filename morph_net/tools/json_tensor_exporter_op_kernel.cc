#include <string>

#include "file/base/file.h"
#include "file/base/helpers.h"

#include "file/base/options.h"
#include "third_party/jsoncpp/json.h"
#include "third_party/tensorflow/core/framework/op_kernel.h"
#include "third_party/tensorflow/core/framework/tensor.h"
#include "third_party/tensorflow/core/framework/tensor_shape.h"
#include "third_party/tensorflow/core/lib/core/errors.h"


namespace morph_net {

using ::tensorflow::errors::InvalidArgument;
using ::tensorflow::OpInputList;
using ::tensorflow::OpKernelConstruction;
using ::tensorflow::OpKernel;
using ::tensorflow::OpKernelContext;
using ::tensorflow::Tensor;


template <typename T>
class JsonTensorExporterOpKernel : public OpKernel {
 public:
  explicit JsonTensorExporterOpKernel(OpKernelConstruction* context)
      : OpKernel(context) {
    int number_of_keys;
    OP_REQUIRES_OK(context, context->GetAttr("N", &number_of_keys));
    OP_REQUIRES_OK(context, context->GetAttr("keys", &keys_));
    OP_REQUIRES_OK(context, context->GetAttr("filename", &filename_));

    OP_REQUIRES(context, keys_.size() == number_of_keys,
                InvalidArgument("Number of keys (", keys_.size(), ") must match"
                                " N (", number_of_keys, ")."));

    OP_REQUIRES_OK(context, WriteFile(""));
  }

  void Compute(OpKernelContext* context) override {
    OpInputList values;
    const Tensor* save;
    OP_REQUIRES_OK(context, context->input_list("values", &values));
    OP_REQUIRES_OK(context, context->input("save", &save));
    if (!save->scalar<bool>()()) return;

    CHECK_EQ(values.size(), keys_.size());  // Enforced by REGISTER_OP

    Json::Value json;
    int ikey = 0;
    for (const Tensor& tensor : values) {
       OP_REQUIRES(context, tensor.dims() <= 1, InvalidArgument(
           "Only scalars and vectors are currnetly supported, but a tensor "
           "with rank ", tensor.dims(), "was found."));

      const string& key = keys_[ikey++];
      if (tensor.dims() == 0) {  // Scalar
        json[key] = tensor.scalar<T>()();
        continue;
      }

      // Vector
      for (int ielement = 0; ielement < tensor.NumElements(); ++ielement) {
        json[key][ielement] = tensor.vec<T>()(ielement);
      }
    }

    Json::StyledWriter writer;
    OP_REQUIRES_OK(context, WriteFile(writer.write(json)));
  }

 private:
  ::tensorflow::Status WriteFile(const string& content) const {
    ::util::Status status =
        ::file::SetContents(filename_, content, ::file::Defaults());
    if (status.ok()){
      return ::tensorflow::Status::OK();
    }
    return InvalidArgument("Unable to write to file ", filename_,
                           ". Error message: ", status.error_message());
  }

  std::vector<string> keys_;
  string filename_;
};

REGISTER_KERNEL_BUILDER(Name("JsonTensorExporter")
                            .Device(::tensorflow::DEVICE_CPU)
                            .TypeConstraint<int32>("T"),
                        JsonTensorExporterOpKernel<int32>);

REGISTER_KERNEL_BUILDER(Name("JsonTensorExporter")
                            .Device(::tensorflow::DEVICE_CPU)
                            .TypeConstraint<float>("T"),
                        JsonTensorExporterOpKernel<float>);

REGISTER_KERNEL_BUILDER(Name("JsonTensorExporter")
                            .Device(::tensorflow::DEVICE_CPU)
                            .TypeConstraint<double>("T"),
                        JsonTensorExporterOpKernel<double>);

REGISTER_KERNEL_BUILDER(Name("JsonTensorExporter")
                            .Device(::tensorflow::DEVICE_CPU)
                            .TypeConstraint<bool>("T"),
                        JsonTensorExporterOpKernel<bool>);

}  // namespace morph_net
