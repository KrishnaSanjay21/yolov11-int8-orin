// DFL plugin implementation.  // RUN ON DEVICE
#include "DFLPlugin.h"

#include <cassert>
#include <cstring>

using namespace nvinfer1;

namespace qint {

namespace {
template <typename T>
void writeToBuffer(char*& buffer, const T& val) {
  std::memcpy(buffer, &val, sizeof(T));
  buffer += sizeof(T);
}
template <typename T>
T readFromBuffer(const char*& buffer) {
  T val;
  std::memcpy(&val, buffer, sizeof(T));
  buffer += sizeof(T);
  return val;
}
}  // namespace

DFLPlugin::DFLPlugin(const void* data, size_t length) {
  const char* d = static_cast<const char*>(data);
  mRegMax = readFromBuffer<int>(d);
  assert(d == static_cast<const char*>(data) + length);
}

IPluginV2DynamicExt* DFLPlugin::clone() const noexcept {
  auto* p = new DFLPlugin(mRegMax);
  p->setPluginNamespace(mNamespace.c_str());
  return p;
}

DimsExprs DFLPlugin::getOutputDimensions(int, const DimsExprs* inputs, int,
                                         IExprBuilder& expr) noexcept {
  // input: (N, 4*regMax, A) -> output: (N, 4, A)
  DimsExprs out;
  out.nbDims = 3;
  out.d[0] = inputs[0].d[0];
  out.d[1] = expr.constant(4);
  out.d[2] = inputs[0].d[2];
  return out;
}

bool DFLPlugin::supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int,
                                          int) noexcept {
  // FP32 linear for both input (pos 0) and output (pos 1). The DFL op is kept in
  // FP32 on purpose — softmax is quantization-sensitive; see DECISIONS.md.
  return inOut[pos].type == DataType::kFLOAT && inOut[pos].format == TensorFormat::kLINEAR;
}

int DFLPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc*,
                       const void* const* inputs, void* const* outputs, void*,
                       cudaStream_t stream) noexcept {
  const int batch = inputDesc[0].dims.d[0];
  const int channels = inputDesc[0].dims.d[1];
  const int anchors = inputDesc[0].dims.d[2];
  dflForward(static_cast<const float*>(inputs[0]), static_cast<float*>(outputs[0]), batch,
             channels, anchors, mRegMax, stream);
  return cudaGetLastError() != cudaSuccess ? 1 : 0;
}

DataType DFLPlugin::getOutputDataType(int, const DataType*, int) const noexcept {
  return DataType::kFLOAT;
}

void DFLPlugin::serialize(void* buffer) const noexcept {
  char* d = static_cast<char*>(buffer);
  writeToBuffer(d, mRegMax);
}

// ---- Creator ----------------------------------------------------------------
DFLPluginCreator::DFLPluginCreator() {
  mFields.clear();
  mFields.emplace_back(PluginField{"reg_max", nullptr, PluginFieldType::kINT32, 1});
  mFC.nbFields = static_cast<int>(mFields.size());
  mFC.fields = mFields.data();
}

IPluginV2* DFLPluginCreator::createPlugin(const char*, const PluginFieldCollection* fc) noexcept {
  int regMax = kRegMax;
  for (int i = 0; i < fc->nbFields; ++i) {
    if (std::strcmp(fc->fields[i].name, "reg_max") == 0) {
      regMax = *static_cast<const int*>(fc->fields[i].data);
    }
  }
  auto* p = new DFLPlugin(regMax);
  p->setPluginNamespace(mNamespace.c_str());
  return p;
}

IPluginV2* DFLPluginCreator::deserializePlugin(const char*, const void* serialData,
                                               size_t serialLength) noexcept {
  auto* p = new DFLPlugin(serialData, serialLength);
  p->setPluginNamespace(mNamespace.c_str());
  return p;
}

REGISTER_TENSORRT_PLUGIN(DFLPluginCreator);

}  // namespace qint
