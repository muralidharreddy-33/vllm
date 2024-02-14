/*
Adapted from https://github.com/mit-han-lab/llm-awq
@article{lin2023awq,
  title={AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration},
  author={Lin, Ji and Tang, Jiaming and Tang, Haotian and Yang, Shang and Dang, Xingyu and Han, Song},
  journal={arXiv},
  year={2023}
}
 */


#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
// #include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "dequantize.cuh"


namespace vllm {
namespace awq {

// Pack two half values.
static inline __device__ __host__ unsigned
__pack_half2(const half x, const half y) {
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}

__device__ __forceinline__ int make_divisible(int c, int divisor){
  return (c + divisor - 1) / divisor;
}

template <int G>
__global__ void __launch_bounds__(128) gemm_forward_4bit_cuda_m128n64k32(int split_k_iters, half* __restrict__ A, int* __restrict__ B, half* __restrict__ scaling_factors, int* zeros, int M, int IC, int OC, half* __restrict__ C)
{
  float C_warp[64];
  __shared__ half A_shared[128 * (32 + 8)];
  __shared__ half B_shared[64 * (32 + 8)];

  // __shared__ half scaling_factors_shared[64];
  // __shared__ half zeros_shared[64];

  int j_factors1 = ((OC + 64 - 1) / 64);

  int blockIdx_y = blockIdx.x % ((M + 128 - 1) / 128 * j_factors1);
  int blockIdx_z = blockIdx.x / ((M + 128 - 1) / 128 * j_factors1);

  half A_shared_warp[32];
  half B_shared_warp[16];
  for (int i_0_3_init = 0; i_0_3_init < 4; ++i_0_3_init) {
    for (int j_0_4_init = 0; j_0_4_init < 2; ++j_0_4_init) {
      for (int i = 0; i < 8; ++i) {
        C_warp[((i_0_3_init * 16) + (j_0_4_init * 8)) + i] = 0.0;
      }
    }
  }

  static constexpr int row_stride_warp = 32 * 8 / 32;
  static constexpr int row_stride_A = 4 * 32 * 8 / 32;
  static constexpr int row_stride = 4 * 32 * 8 / 32;
  const int make_divisible_multipler = 128 / G;
  const int zeros_w = make_divisible(make_divisible(IC / G, 8), make_divisible_multipler) * make_divisible_multipler;
  const int sf_w = zeros_w * 8;

  int ld_A_row = (blockIdx_y / j_factors1 * 128 + threadIdx.y * row_stride_warp + threadIdx.x * 8 / 32);     // threadIdx.y is warp_id

  half* A_ptr = A
                + (((int)blockIdx_y) / j_factors1 * 128 + (((int)threadIdx.y) * row_stride_warp) + ((int)threadIdx.x) / (32 / 8)) * IC
                + (((int)threadIdx.x) % (32 / 8)) * 8;

  int* B_ptr = B
            + ((int)threadIdx.y) * (IC / 8) * 8
            + (((int)threadIdx.x) / (32 / 8)) * (IC / 8)
            + (((int)blockIdx_y) % j_factors1) * 64 * (IC / 8)
            + (((int)threadIdx.x) % (32 / 8)) * 1;

// Why * 1 in the above line?

  half* A_shared_ptr = A_shared
                    + ((int)threadIdx.y) * row_stride_warp * (32 + 8)
                    + (((int)threadIdx.x) / (32 / 8)) * (32 + 8)
                    + (((int)threadIdx.x) % (32 / 8) ) * 8;

  half* B_shared_ptr = B_shared
                    + ((int)threadIdx.y) * (row_stride / 4) * (32 + 8)
                    + (((int)threadIdx.x) / (32 / 8)) * (32 + 8)
                    + (((int)threadIdx.x) % (32 / 8)) * 8;


  int* zeros_ptr = zeros
                + ((int)threadIdx.y) * zeros_w * 8
                + (((int)threadIdx.x) / (32 / 8)) * zeros_w
                + (((int)blockIdx_y) % j_factors1) * 64 * zeros_w
                // this term is zero
                + (((int)threadIdx.x) % (32 / 8)) / G ;

  half* scaling_factors_ptr = scaling_factors
                            + ((int)threadIdx.y) * sf_w * 8
                            + (((int)threadIdx.x) / (32 / 8)) * sf_w
                            + (((int)blockIdx_y) % j_factors1) * (64) * sf_w
                            // this term is zero
                            + (((int)threadIdx.x) % (32 / 8)) * 8 / G;


  // Haotian: TBD, check, May 29 11:46 AM PST
  half* C_ptr = C
              + blockIdx_z * M * OC        // blockIdx_z -> split_k dim
              + (((int)blockIdx_y) % j_factors1) * 64
              + (((int)threadIdx.y) / 2) * 32
              + (((int)threadIdx.x) % 4) * 2;

  // preload s.f. and zeros
  int k_bound = make_divisible(IC / 32, split_k_iters); // (IC / 32 + split_k_iters - 1) / split_k_iters;
  if ((k_bound - 1) * 32 + blockIdx_z >= IC) k_bound -= 1;

  // TODO (Haotian): load scales and zero points to smem

  for (int _k_0_0 = 0; _k_0_0 < k_bound; ++_k_0_0) {
    int k_0_0 = _k_0_0 * split_k_iters + blockIdx_z;
    __syncthreads();
    // TODO: Haotian: Here we assume M % cta_M = 0.
    for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < 4; ++ax0_ax1_fused_0)
    {
      if (ld_A_row + ax0_ax1_fused_0 * row_stride_A < M)
      {
        *(uint4*)(A_shared_ptr + ax0_ax1_fused_0 * row_stride_A * 40) = *(uint4*)(A_ptr + (ax0_ax1_fused_0 * row_stride_A * IC) + (k_0_0 * 32));
      }
      else
      {
        *(uint4*)(A_shared_ptr + ax0_ax1_fused_0 * row_stride_A * 40) = make_uint4(0, 0, 0, 0);
      }
    }


    int* zeros_ptr_local = zeros_ptr + k_0_0 * 32 / G / 8;
    half* scaling_factors_ptr_local = scaling_factors_ptr + k_0_0 * 32 / G;

    // uint4 B_loaded_scale = make_uint4(0, 0, 0, 0);
    int* B_ptr_local = B_ptr + k_0_0 * (32 / 8);

    for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < 2; ++ax0_ax1_fused_0) {

      // B: 32 x 136 (128+8) float16
      // each warp: 32 x 4
      // each thr: read 32 bit -> convert to 8xFP16 (a UINT4) -> scale and minus zero -> WB UINT4
      // row stride in shared memory: (NWARPS * 32 * 8 / cta_N)
      int B_loaded_current = *(B_ptr_local + ax0_ax1_fused_0 * row_stride * (IC / 8));
      int zeros_loaded = *(zeros_ptr_local + ax0_ax1_fused_0 * row_stride * zeros_w);
      zeros_loaded >>= ((k_0_0 * 32 / G) % 8) * 4;
      float current_zeros = (float)(zeros_loaded & 0xF);
      half scaling_factors_loaded = *(scaling_factors_ptr_local + ax0_ax1_fused_0 * row_stride * sf_w);
      half B_loaded_fp16[8];
      #pragma unroll
      for (int ic_1 = 0; ic_1 < 8; ic_1++){
        float current_single_weight_fp = (float)(B_loaded_current & 0xF);
        half dequantized_weight = __float2half(__half2float(scaling_factors_loaded) * (current_single_weight_fp - current_zeros));
        B_loaded_current = B_loaded_current >> 4;
        B_loaded_fp16[ic_1] = dequantized_weight;
      }
      // write back
      *(uint4*)(B_shared_ptr + ax0_ax1_fused_0 * row_stride * (32 + 8)) = *reinterpret_cast<uint4*>(B_loaded_fp16);
    }
    __syncthreads();
    for (int k_0_1 = 0; k_0_1 < 2; ++k_0_1) {
      for (int ax0_0 = 0; ax0_0 < 4; ++ax0_0) {
        {
          unsigned int addr;
          __asm__ __volatile__(
            "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
            : "=r"(addr)
            : "l"((void *)((&(A_shared[((((((int)threadIdx.y) & 1) * 2560) + (ax0_0 * 640)) + (k_0_1 * 16))])) + (((((int)threadIdx.x) & 15) * 40) + ((((int)threadIdx.x) >> 4) * 8))))
          );
          __asm__ __volatile__(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
            "{%0, %1, %2, %3}, [%4];\n"
            : "=r"(((unsigned *)(A_shared_warp + (ax0_0 * 8)))[0]), "=r"(((unsigned *)(A_shared_warp + (ax0_0 * 8)))[1]), "=r"(((unsigned *)(A_shared_warp + (ax0_0 * 8)))[2]), "=r"(((unsigned *)(A_shared_warp + (ax0_0 * 8)))[3])
            : "r"(addr)
          );
        }
      }

      for (int ax0_0_1 = 0; ax0_0_1 < 2; ++ax0_0_1) {
        {
          unsigned int addr;
          __asm__ __volatile__(
            "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
            : "=r"(addr)
            : "l"((void *)((&(B_shared[((((((int)threadIdx.y) >> 1) * 1280) + (ax0_0_1 * 640)) + (k_0_1 * 16))])) + ((((((int)threadIdx.x) >> 4) * 320) + ((((int)threadIdx.x) & 7) * 40)) + (((((int)threadIdx.x) & 15) >> 3) * 8))))
          );
          __asm__ __volatile__(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
            "{%0, %1, %2, %3}, [%4];\n"
            : "=r"(((unsigned *)(B_shared_warp + (ax0_0_1 * 8)))[0]), "=r"(((unsigned *)(B_shared_warp + (ax0_0_1 * 8)))[1]), "=r"(((unsigned *)(B_shared_warp + (ax0_0_1 * 8)))[2]), "=r"(((unsigned *)(B_shared_warp + (ax0_0_1 * 8)))[3])
            : "r"(addr)
          );
        }
      }

      for (int i_0_3 = 0; i_0_3 < 4; ++i_0_3) {
        for (int j_0_4 = 0; j_0_4 < 2; ++j_0_4) {

          {
            __asm__ __volatile__(
              "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
              "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
              :  "=f"(((float *)(C_warp + ((i_0_3 * 16) + (j_0_4 * 8))))[0]), "=f"(((float *)(C_warp + ((i_0_3 * 16) + (j_0_4 * 8))))[1]), "=f"(((float *)(C_warp + ((i_0_3 * 16) + (j_0_4 * 8))))[2]), "=f"(((float *)(C_warp + ((i_0_3 * 16) + (j_0_4 * 8))))[3])
              : "r"(((unsigned *)(A_shared_warp + (i_0_3 * 8)))[0]), "r"(((unsigned *)(A_shared_warp + (i_0_3 * 8)))[1]), "r"(((unsigned *)(A_shared_warp + (i_0_3 * 8)))[2]), "r"(((unsigned *)(A_shared_warp + (i_0_3 * 8)))[3]), "r"(((unsigned *)(B_shared_warp + (j_0_4 * 8)))[0]), "r"(((unsigned *)(B_shared_warp + (j_0_4 * 8)))[1]), "f"(((float *)(C_warp + ((i_0_3 * 16) + (j_0_4 * 8))))[0]), "f"(((float *)(C_warp + ((i_0_3 * 16) + (j_0_4 * 8))))[1]), "f"(((float *)(C_warp + ((i_0_3 * 16) + (j_0_4 * 8))))[2]), "f"(((float *)(C_warp + ((i_0_3 * 16) + (j_0_4 * 8))))[3]));
          }

          {
            __asm__ __volatile__(
              "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
              "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
              :  "=f"(((float *)(C_warp + (((i_0_3 * 16) + (j_0_4 * 8)) + 4)))[0]), "=f"(((float *)(C_warp + (((i_0_3 * 16) + (j_0_4 * 8)) + 4)))[1]), "=f"(((float *)(C_warp + (((i_0_3 * 16) + (j_0_4 * 8)) + 4)))[2]), "=f"(((float *)(C_warp + (((i_0_3 * 16) + (j_0_4 * 8)) + 4)))[3])
              : "r"(((unsigned *)(A_shared_warp + (i_0_3 * 8)))[0]), "r"(((unsigned *)(A_shared_warp + (i_0_3 * 8)))[1]), "r"(((unsigned *)(A_shared_warp + (i_0_3 * 8)))[2]), "r"(((unsigned *)(A_shared_warp + (i_0_3 * 8)))[3]), "r"(((unsigned *)(B_shared_warp + ((j_0_4 * 8) + 4)))[0]), "r"(((unsigned *)(B_shared_warp + ((j_0_4 * 8) + 4)))[1]), "f"(((float *)(C_warp + (((i_0_3 * 16) + (j_0_4 * 8)) + 4)))[0]), "f"(((float *)(C_warp + (((i_0_3 * 16) + (j_0_4 * 8)) + 4)))[1]), "f"(((float *)(C_warp + (((i_0_3 * 16) + (j_0_4 * 8)) + 4)))[2]), "f"(((float *)(C_warp + (((i_0_3 * 16) + (j_0_4 * 8)) + 4)))[3]));
          }
        }
      }
    }
  }

// Haotian: Here (May 29 11:46AM PST)
// TODO: Shang: Hoist loop invariance.
  for (int ax0_0_2 = 0; ax0_0_2 < 4; ++ax0_0_2) {
    for (int ax1_0 = 0; ax1_0 < 2; ++ax1_0) {
      for (int local_id = 0; local_id < 8; ++local_id) {
        int row_offset = (((int)blockIdx_y) / j_factors1) * 128 + (threadIdx.y % 2) * 64 + ax0_0_2 * 16 + (local_id % 4) / 2 * 8 + ((int)threadIdx.x) / 4;
        if (row_offset < M)
        {
          *(C_ptr + ax1_0 * 16 + row_offset * OC + (local_id / 4) * 8 + local_id % 2) = __float2half(C_warp[(ax0_0_2 * 16) + (ax1_0 * 8) + local_id]);
        }
      }
    }
  }
}

__global__ void __launch_bounds__(64) dequantize_weights(
    int* __restrict__ B,
    half* __restrict__ scaling_factors,
    int* __restrict__ zeros,
    half* __restrict__ C,
    int G
)
{
  static constexpr uint32_t ZERO = 0x0;
  half B_shared[32 * (128 + 8)];

  half* B_shared_ptr2 = B_shared;

  int N = blockDim.x * gridDim.x;  // 2
  int col = (blockIdx.x * blockDim.x + threadIdx.x);
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int index1 = 8 * col + 8 * row * N;
  half* C_ptr2 = C + index1;

  int index2 = col + row * N;
  int* B_ptr2 = B + index2;

  int index3 = col + (int)(row / G) * N;
  int* zeros_ptr2 = zeros + index3;
  int index4 = 8 * col + (int)(row / G) * N * 8;
  half* scaling_factors_ptr2 = scaling_factors + index4;

  uint32_t zeros_loaded = *(uint32_t*)(zeros_ptr2);
  uint4 B_loaded_zero = dequantize_s4_to_fp16x2(zeros_loaded);
  uint4 B_loaded_scale = *(uint4*)(scaling_factors_ptr2);

  uint32_t B_loaded = *(uint32_t*)B_ptr2;
  uint4 B_loaded_fp16 = dequantize_s4_to_fp16x2(B_loaded);
  asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.x) : "r"(B_loaded_fp16.x), "r"(B_loaded_zero.x));
  asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(B_loaded_fp16.x) : "r"(B_loaded_fp16.x), "r"(B_loaded_scale.x), "r"(ZERO));
  asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.y) : "r"(B_loaded_fp16.y), "r"(B_loaded_zero.y));
  asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(B_loaded_fp16.y) : "r"(B_loaded_fp16.y), "r"(B_loaded_scale.y), "r"(ZERO));
  asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.z) : "r"(B_loaded_fp16.z), "r"(B_loaded_zero.z));
  asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(B_loaded_fp16.z) : "r"(B_loaded_fp16.z), "r"(B_loaded_scale.z), "r"(ZERO));
  asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.w) : "r"(B_loaded_fp16.w), "r"(B_loaded_zero.w));
  asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(B_loaded_fp16.w) : "r"(B_loaded_fp16.w), "r"(B_loaded_scale.w), "r"(ZERO));

  *(uint4*)B_shared_ptr2 = B_loaded_fp16;

  for (int i = 0; i < 8; ++i) {
    *(C_ptr2 + i) = B_shared[i];
  }
}

} // namespace awq
} // namespace vllm

torch::Tensor awq_dequantize(
    torch::Tensor _kernel,
    torch::Tensor _scaling_factors,
    torch::Tensor _zeros,
    int split_k_iters,
    int thx,
    int thy)
{
    int in_c = _kernel.size(0);
    int qout_c = _kernel.size(1);
    int out_c = qout_c * 8;
    int G = in_c / _scaling_factors.size(0);

    int x_thread = thx;
    int y_thread = thy;

    int x_blocks = 1;
    int y_blocks = 1;
    if (thx==0) {
      x_thread = qout_c;
    }
    if (thy==0) {
      y_thread = in_c;
    }
    if (thx==0 && thy==0) {
      x_thread = 8;
      y_thread = 8;
      x_blocks = (int)(qout_c / 8);
      y_blocks = (int)(in_c / 8);
    }

    const at::cuda::OptionalCUDAGuard device_guard(device_of(_scaling_factors));

    auto options = torch::TensorOptions().dtype(_scaling_factors.dtype()).device(_scaling_factors.device());
    at::Tensor _de_kernel = torch::empty({in_c, out_c}, options);

    auto kernel = reinterpret_cast<int*>(_kernel.data_ptr<int>());
    auto de_kernel = reinterpret_cast<half*>(_de_kernel.data_ptr<at::Half>());
    auto scaling_factors = reinterpret_cast<half*>(_scaling_factors.data_ptr<at::Half>());
    auto zeros = reinterpret_cast<int*>(_zeros.data_ptr<int>());

    dim3 num_blocks(x_blocks, y_blocks);
    dim3 threads_per_block(x_thread, y_thread);

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    vllm::awq::dequantize_weights<<<num_blocks, threads_per_block, 0, stream>>>(
        kernel, scaling_factors, zeros, de_kernel, G);

    return _de_kernel;
}

// in_feats: M, IC [float16]
// kernel: IC, OC // 8 [int32] -> cast to IC, OC [uint4b]
// scaling_factors: IC // G, OC [float16]
// zeros: IC // G, OC // 8 [int32] -> cast to IC // G, OC [uint4b]
// assume that batch_size < 16 for now

torch::Tensor awq_gemm(
    torch::Tensor _in_feats,
    torch::Tensor _kernel,
    torch::Tensor _scaling_factors,
    torch::Tensor _zeros,
    int group_size,
    int split_k_iters)
{
    int num_in_feats = _in_feats.size(0);
    int num_in_channels = _in_feats.size(1);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(_in_feats));

    auto options = torch::TensorOptions().dtype(_in_feats.dtype()).device(_in_feats.device());
    at::Tensor _out_feats = torch::empty({split_k_iters, num_in_feats, _kernel.size(1) * 8}, options);
    int num_out_feats = _out_feats.size(-2);
    int num_out_channels = _out_feats.size(-1);

    auto in_feats = reinterpret_cast<half*>(_in_feats.data_ptr<at::Half>());
    auto kernel = reinterpret_cast<int*>(_kernel.data_ptr<int>());
    auto out_feats = reinterpret_cast<half*>(_out_feats.data_ptr<at::Half>());
    auto scaling_factors = reinterpret_cast<half*>(_scaling_factors.data_ptr<at::Half>());
    auto zeros = reinterpret_cast<int*>(_zeros.data_ptr<int>());
    // auto half_zeros = reinterpret_cast<half*>(zeros);
    // int group_size = num_in_channels / _scaling_factors.size(0);

    // blockIdx_x: i_factors[0] * j_factors[0]
    // blockIdx_y: i_factors[1] * j_factors[1]

    if (num_out_channels % 64 != 0)
        throw std::invalid_argument("OC is not multiple of cta_N = 64");
    if (num_out_channels % 8 != 0)
        throw std::invalid_argument("OC is not multiple of pack_num = 8");
    int j_factors1 = num_out_channels / 64 / 1;
    dim3 num_blocks((num_out_feats + 128 - 1) / 128 * j_factors1 * split_k_iters);

    // threadIdx.x: 32
    // threadIdx.y: i_factors[2] * j_factors[2]
    dim3 threads_per_block(32, 4);
    if (group_size == 128)
    {
      vllm::awq::gemm_forward_4bit_cuda_m128n64k32<128><<<num_blocks, threads_per_block>>>(
        split_k_iters, in_feats, kernel, scaling_factors, zeros, num_in_feats, num_in_channels, num_out_channels, out_feats);
    }
    else if (group_size == 64)
    {
      vllm::awq::gemm_forward_4bit_cuda_m128n64k32<64><<<num_blocks, threads_per_block>>>(
        split_k_iters, in_feats, kernel, scaling_factors, zeros, num_in_feats, num_in_channels, num_out_channels, out_feats);
    }
    else
    {
      throw std::invalid_argument("Group size temporarily not supported.");
    }
    return _out_feats.sum(0);
}
