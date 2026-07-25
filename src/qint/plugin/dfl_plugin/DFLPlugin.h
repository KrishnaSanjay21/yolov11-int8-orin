// Fused DFL (Distribution Focal Loss) decode plugin for TensorRT.  // RUN ON DEVICE
//
// Fuses softmax-over-bins + expectation into a single kernel.
//   input  : (N, 4*REG_MAX, A) float logits
//   output : (N, 4, A)         float expected distances
//
// Validated numerically against qint.plugin.dfl_reference (see scripts/validate_plugin.py).
#ifndef QINT_DFL_PLUGIN_H
#define QINT_DFL_PLUGIN_H

#include <NvInferRuntime.h>
#include <cstdint>
#include <string>
#include <vector>

namespace qint {

constexpr int kRegMax = 16;  // YOLOv11 DFL bins; keep in sync with dfl_reference.REG_MAX

// Launch wrapper implemented in dfl_kernel.cu
void dflForward(const float* input, float* output, int batch, int channels,
                int anchors, int regMax, cudaStream_t stream);

class DFLPlugin : public nvinfer1::IPluginV2DynamicExt {
 public:
  DFLPlugin() = default;
  explicit DFLPlugin(int regMax) : mRegMax(regMax) {}
  DFLPlugin(const void* data, size_t length);

  // IPluginV2DynamicExt
  nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
  nvinfer1::DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs,
                                          int nbInputs,
                                          nvinfer1::IExprBuilder& expr) noexcept override;
  bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs,
                                 int nbOutputs) noexcept override;
  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
                       const nvinfer1::DynamicPluginTensorDesc* out,
                       int nbOutputs) noexcept override {}
  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
                          const nvinfer1::PluginTensorDesc* outputs,
                          int nbOutputs) const noexcept override { return 0; }
  int enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
              const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs,
              void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

  // IPluginV2Ext
  nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
                                       int nbInputs) const noexcept override;

  // IPluginV2
  const char* getPluginType() const noexcept override { return "DFL"; }
  const char* getPluginVersion() const noexcept override { return "1"; }
  int getNbOutputs() const noexcept override { return 1; }
  int initialize() noexcept override { return 0; }
  void terminate() noexcept override {}
  size_t getSerializationSize() const noexcept override { return sizeof(int); }
  void serialize(void* buffer) const noexcept override;
  void destroy() noexcept override { delete this; }
  void setPluginNamespace(const char* ns) noexcept override { mNamespace = ns; }
  const char* getPluginNamespace() const noexcept override { return mNamespace.c_str(); }

 private:
  int mRegMax{kRegMax};
  std::string mNamespace;
};

class DFLPluginCreator : public nvinfer1::IPluginCreator {
 public:
  DFLPluginCreator();
  const char* getPluginName() const noexcept override { return "DFL"; }
  const char* getPluginVersion() const noexcept override { return "1"; }
  const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override { return &mFC; }
  nvinfer1::IPluginV2* createPlugin(const char* name,
                                    const nvinfer1::PluginFieldCollection* fc) noexcept override;
  nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serialData,
                                         size_t serialLength) noexcept override;
  void setPluginNamespace(const char* ns) noexcept override { mNamespace = ns; }
  const char* getPluginNamespace() const noexcept override { return mNamespace.c_str(); }

 private:
  nvinfer1::PluginFieldCollection mFC{};
  std::vector<nvinfer1::PluginField> mFields;
  std::string mNamespace;
};

}  // namespace qint
#endif  // QINT_DFL_PLUGIN_H
