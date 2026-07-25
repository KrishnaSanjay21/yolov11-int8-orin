// Fused DFL kernel.  // RUN ON DEVICE
//
// One thread per (batch, side, anchor). Each thread performs, over REG_MAX bins:
//   1) max-reduction for numerical stability,
//   2) sum(exp(x-max)) and sum(exp(x-max)*j) in a single pass,
//   3) output = weighted_sum / exp_sum.
// This mirrors qint.plugin.dfl_reference.dfl_fused_reference exactly.
#include <cuda_runtime.h>
#include <math.h>

namespace qint {

__global__ void dflKernel(const float* __restrict__ input, float* __restrict__ output,
                          int batch, int channels, int anchors, int regMax) {
  // global thread index over N*4*A
  const long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const long total = static_cast<long>(batch) * 4 * anchors;
  if (idx >= total) return;

  const int a = idx % anchors;
  const int side = (idx / anchors) % 4;
  const int n = idx / (anchors * 4);

  // input layout: [n, c, a] with c in [0, channels); side occupies bins [side*regMax, +regMax)
  const long base = (static_cast<long>(n) * channels + side * regMax) * anchors + a;
  const long stride = anchors;  // step between consecutive bins for fixed (n, a)

  // pass 1: max
  float m = -INFINITY;
  for (int j = 0; j < regMax; ++j) {
    float v = input[base + static_cast<long>(j) * stride];
    m = fmaxf(m, v);
  }
  // pass 2: sum(exp) and sum(exp*j)
  float esum = 0.f;
  float wsum = 0.f;
  for (int j = 0; j < regMax; ++j) {
    float e = __expf(input[base + static_cast<long>(j) * stride] - m);
    esum += e;
    wsum += e * static_cast<float>(j);
  }
  const long outIdx = (static_cast<long>(n) * 4 + side) * anchors + a;
  output[outIdx] = wsum / esum;
}

void dflForward(const float* input, float* output, int batch, int channels, int anchors,
                int regMax, cudaStream_t stream) {
  const long total = static_cast<long>(batch) * 4 * anchors;
  const int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  dflKernel<<<blocks, threads, 0, stream>>>(input, output, batch, channels, anchors, regMax);
}

}  // namespace qint
