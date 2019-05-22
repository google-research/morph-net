#include <string>

#include "file/base/file.h"
#include "file/base/helpers.h"
#include "file/base/path.h"

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "third_party/jsoncpp/json.h"
#include "third_party/tensorflow/core/framework/fake_input.h"
#include "third_party/tensorflow/core/framework/node_def_builder.h"
#include "third_party/tensorflow/core/framework/tensor.h"
#include "third_party/tensorflow/core/framework/tensor_testutil.h"
#include "third_party/tensorflow/core/kernels/ops_testutil.h"
#include "third_party/tensorflow/core/lib/core/status_test_util.h"

namespace morph_net {

using ::tensorflow::DT_INT32;
using ::tensorflow::FakeInput;
using ::tensorflow::NodeDefBuilder;
using ::tensorflow::OpsTestBase;
using ::tensorflow::Status;
using ::tensorflow::TensorShape;
using ::testing::ElementsAre;


std::vector<int> ToVector(const Json::Value& json) {
  std::vector<int> v;
  for (const Json::Value& item : json) {
    v.push_back(item.asInt());
  }
  return v;
}


class JsonTensorExporterTest : public OpsTestBase {};

TEST_F(JsonTensorExporterTest, Success) {
  const int kLength = 3;
  const string filename = ::file::JoinPath(FLAGS_test_tmpdir, "success.json");
  TF_ASSERT_OK(
       NodeDefBuilder("exporter", "JsonTensorExporter")
      .Attr("T", DT_INT32)
      .Attr("N", kLength)
      .Attr("keys", {"k1", "k2", "k3"})
      .Attr("filename", filename)
      .Input(FakeInput(kLength, ::tensorflow::DT_INT32))
      .Input(FakeInput(::tensorflow::DT_BOOL))
      .Finalize(node_def()));

  TF_ASSERT_OK(InitOp());
  // The initialization of the op creates an empty file at `filename`. We delete
  // both to verify it was created, and to clean it up for the next steps of the
  // test.
  ASSERT_OK(::file::Delete(filename, ::file::Defaults()));

  AddInputFromArray<int>(TensorShape({3}), {3, 5, 7});
  AddInputFromArray<int>(TensorShape({2}), {6, 4});
  AddInputFromArray<int>(TensorShape({}), {9});

  // Set the `save` flag initially to false - so the op should be a no-op.
  AddInputFromArray<bool>(TensorShape({}), {false});
  TF_ASSERT_OK(RunOpKernel());
  // Verify that indeed no file was created.
  EXPECT_EQ(absl::StatusCode::kNotFound,
            ::file::Exists(filename, ::file::Defaults()).code());

  // Flip the `save` flag to true and test the content of the savef file.
  tensors_[3]->scalar<bool>()() = true;
  TF_ASSERT_OK(RunOpKernel());

  string contents;
  ASSERT_OK(::file::GetContents(filename, &contents, ::file::Defaults()));
  Json::Reader reader;
  Json::Value json;
  reader.parse(contents, json);
  EXPECT_THAT(json.getMemberNames(), ElementsAre("k1", "k2", "k3"));
  EXPECT_TRUE(json["k1"].isArray());
  EXPECT_THAT(ToVector(json["k1"]), ElementsAre(3, 5, 7));
  EXPECT_TRUE(json["k2"].isArray());
  EXPECT_THAT(ToVector(json["k2"]), ElementsAre(6, 4));
  EXPECT_EQ(9, json["k3"].asInt());
}

TEST_F(JsonTensorExporterTest, WrongNumberOfKeys) {
  const int kLength = 3;
  const string filename = ::file::JoinPath(FLAGS_test_tmpdir, "failure.json");
  TF_ASSERT_OK(
       NodeDefBuilder("exporter", "JsonTensorExporter")
      .Attr("T", DT_INT32)
      .Attr("N", kLength)
      .Attr("keys", {"k1", "k2"})  // Two keys only, even though kLength = 3.
      .Attr("filename", filename)
      .Input(FakeInput(kLength, ::tensorflow::DT_INT32))
      .Input(FakeInput(::tensorflow::DT_BOOL))
      .Finalize(node_def()));

  EXPECT_FALSE(InitOp().ok());
}

TEST_F(JsonTensorExporterTest, BadFileName) {
  const int kLength = 3;
  const string filename = "**bad";
  TF_ASSERT_OK(
       NodeDefBuilder("exporter", "JsonTensorExporter")
      .Attr("T", DT_INT32)
      .Attr("N", kLength)
      .Attr("keys", {"k1", "k2", "k3"})
      .Attr("filename", filename)
      .Input(FakeInput(kLength, ::tensorflow::DT_INT32))
      .Input(FakeInput(::tensorflow::DT_BOOL))
      .Finalize(node_def()));

  EXPECT_FALSE(InitOp().ok());
}

}  // namespace morph_net
