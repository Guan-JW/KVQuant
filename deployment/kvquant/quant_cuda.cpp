#include <torch/all.h>
#include <torch/python.h>
#include <c10/cuda/CUDAGuard.h>

#include "nvcomp/lz4.h"
#include "BatchData.h"

#include <chrono> // Include the chrono header for timing
#include <iomanip> // For setprecision

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // For automatic conversion between C++ and Python containers

#define PRINT
namespace py = pybind11;

void vecquant4appendvecK_cuda(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, int kcachelen
);
void vecquant4appendvecK(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, int kcachelen
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(mat));
  vecquant4appendvecK_cuda(mat, lookup_table, newvec, kcachelen);
}

void vecquant3appendvecK_cuda(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, int kcachelen
);
void vecquant3appendvecK(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, int kcachelen
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(mat));
  vecquant3appendvecK_cuda(mat, lookup_table, newvec, kcachelen);
}

void vecquant2appendvecK_cuda(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, int kcachelen
);
void vecquant2appendvecK(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, int kcachelen
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(mat));
  vecquant2appendvecK_cuda(mat, lookup_table, newvec, kcachelen);
}

void vecquant4appendvecKsparse_cuda(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, torch::Tensor outliers_rescaled, torch::Tensor outlier_threshold_lower, torch::Tensor outlier_threshold_upper, int kcachelen
);
void vecquant4appendvecKsparse(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, torch::Tensor outliers_rescaled, torch::Tensor outlier_threshold_lower, torch::Tensor outlier_threshold_upper, int kcachelen
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(mat));
  vecquant4appendvecKsparse_cuda(mat, lookup_table, newvec, outliers_rescaled, outlier_threshold_lower, outlier_threshold_upper, kcachelen);
}

void vecquant4appendvecKsparseParallel_cuda(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, torch::Tensor outliers_rescaled, torch::Tensor outlier_threshold_lower, torch::Tensor outlier_threshold_upper
);
void vecquant4appendvecKsparseParallel(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, torch::Tensor outliers_rescaled, torch::Tensor outlier_threshold_lower, torch::Tensor outlier_threshold_upper
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(mat));
  vecquant4appendvecKsparseParallel_cuda(mat, lookup_table, newvec, outliers_rescaled, outlier_threshold_lower, outlier_threshold_upper);
}

void vecquant3appendvecKsparse_cuda(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, torch::Tensor outliers_rescaled, torch::Tensor outlier_threshold_lower, torch::Tensor outlier_threshold_upper, int kcachelen
);
void vecquant3appendvecKsparse(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, torch::Tensor outliers_rescaled, torch::Tensor outlier_threshold_lower, torch::Tensor outlier_threshold_upper, int kcachelen
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(mat));
  vecquant3appendvecKsparse_cuda(mat, lookup_table, newvec, outliers_rescaled, outlier_threshold_lower, outlier_threshold_upper, kcachelen);
}

void vecquant3appendvecKsparseParallel_cuda(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, torch::Tensor outliers_rescaled, torch::Tensor outlier_threshold_lower, torch::Tensor outlier_threshold_upper
);
void vecquant3appendvecKsparseParallel(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, torch::Tensor outliers_rescaled, torch::Tensor outlier_threshold_lower, torch::Tensor outlier_threshold_upper
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(mat));
  vecquant3appendvecKsparseParallel_cuda(mat, lookup_table, newvec, outliers_rescaled, outlier_threshold_lower, outlier_threshold_upper);
}

void vecquant2appendvecKsparse_cuda(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, torch::Tensor outliers_rescaled, torch::Tensor outlier_threshold_lower, torch::Tensor outlier_threshold_upper, int kcachelen
);
void vecquant2appendvecKsparse(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, torch::Tensor outliers_rescaled, torch::Tensor outlier_threshold_lower, torch::Tensor outlier_threshold_upper, int kcachelen
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(mat));
  vecquant2appendvecKsparse_cuda(mat, lookup_table, newvec, outliers_rescaled, outlier_threshold_lower, outlier_threshold_upper, kcachelen);
}

void vecquant2appendvecKsparseParallel_cuda(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, torch::Tensor outliers_rescaled, torch::Tensor outlier_threshold_lower, torch::Tensor outlier_threshold_upper
);
void vecquant2appendvecKsparseParallel(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, torch::Tensor outliers_rescaled, torch::Tensor outlier_threshold_lower, torch::Tensor outlier_threshold_upper
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(mat));
  vecquant2appendvecKsparseParallel_cuda(mat, lookup_table, newvec, outliers_rescaled, outlier_threshold_lower, outlier_threshold_upper);
}


void vecquant4appendvecV_cuda(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, int vcachelen
);
void vecquant4appendvecV(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, int vcachelen
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(mat));
  vecquant4appendvecV_cuda(mat, lookup_table, newvec, vcachelen);
}

void vecquant3appendvecV_cuda(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, int vcachelen
);
void vecquant3appendvecV(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, int vcachelen
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(mat));
  vecquant3appendvecV_cuda(mat, lookup_table, newvec, vcachelen);
}

void vecquant2appendvecV_cuda(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, int vcachelen
);
void vecquant2appendvecV(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, int vcachelen
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(mat));
  vecquant2appendvecV_cuda(mat, lookup_table, newvec, vcachelen);
}

void vecquant4appendvecVsparse_cuda(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, float zeropoint, float outlier_threshold_lower, float outlier_threshold_upper, int vcachelen
);
void vecquant4appendvecVsparse(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, float zeropoint, float outlier_threshold_lower, float outlier_threshold_upper, int vcachelen
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(mat));
  vecquant4appendvecVsparse_cuda(mat, lookup_table, newvec, zeropoint, outlier_threshold_lower, outlier_threshold_upper, vcachelen);
}

void vecquant4appendvecVsparseParallel_cuda(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, torch::Tensor outlier_threshold_lower, torch::Tensor outlier_threshold_upper
);
void vecquant4appendvecVsparseParallel(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, torch::Tensor outlier_threshold_lower, torch::Tensor outlier_threshold_upper
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(mat));
  vecquant4appendvecVsparseParallel_cuda(mat, lookup_table, newvec, outlier_threshold_lower, outlier_threshold_upper);
}


void vecquant3appendvecVsparse_cuda(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, float zeropoint, float outlier_threshold_lower, float outlier_threshold_upper, int vcachelen
);
void vecquant3appendvecVsparse(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, float zeropoint, float outlier_threshold_lower, float outlier_threshold_upper, int vcachelen
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(mat));
  vecquant3appendvecVsparse_cuda(mat, lookup_table, newvec, zeropoint, outlier_threshold_lower, outlier_threshold_upper, vcachelen);
}

void vecquant3appendvecVsparseParallel_cuda(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, torch::Tensor outlier_threshold_lower, torch::Tensor outlier_threshold_upper
);
void vecquant3appendvecVsparseParallel(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, torch::Tensor outlier_threshold_lower, torch::Tensor outlier_threshold_upper
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(mat));
  vecquant3appendvecVsparseParallel_cuda(mat, lookup_table, newvec, outlier_threshold_lower, outlier_threshold_upper);
}

void vecquant2appendvecVsparse_cuda(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, float zeropoint, float outlier_threshold_lower, float outlier_threshold_upper, int vcachelen
);
void vecquant2appendvecVsparse(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, float zeropoint, float outlier_threshold_lower, float outlier_threshold_upper, int vcachelen
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(mat));
  vecquant2appendvecVsparse_cuda(mat, lookup_table, newvec, zeropoint, outlier_threshold_lower, outlier_threshold_upper, vcachelen);
}

void vecquant2appendvecVsparseParallel_cuda(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, torch::Tensor outlier_threshold_lower, torch::Tensor outlier_threshold_upper
);
void vecquant2appendvecVsparseParallel(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, torch::Tensor outlier_threshold_lower, torch::Tensor outlier_threshold_upper
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(mat));
  vecquant2appendvecVsparseParallel_cuda(mat, lookup_table, newvec, outlier_threshold_lower, outlier_threshold_upper);
}


void vecquant4matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table, int kcachelen, float theta, int pos_offset
);
void vecquant4matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table, int kcachelen, float theta, int pos_offset
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant4matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt_cuda(vec, mat, mul, lookup_table, kcachelen, theta, pos_offset);
}

void vecquant4matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt2_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table, int kcachelen,
  torch::Tensor outliers, torch::Tensor outlier_indices, float theta, int pos_offset
);
void vecquant4matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt2(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table, int kcachelen,
  torch::Tensor outliers, torch::Tensor outlier_indices, float theta, int pos_offset
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant4matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt2_cuda(vec, mat, mul, lookup_table, kcachelen, outliers, outlier_indices, theta, pos_offset);
}

void vecquant4matmul_nuq_perchannel_transposed_mha_batched_fused_opt_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table, int vcachelen
);
void vecquant4matmul_nuq_perchannel_transposed_mha_batched_fused_opt(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table, int vcachelen
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant4matmul_nuq_perchannel_transposed_mha_batched_fused_opt_cuda(vec, mat, mul, lookup_table, vcachelen);
}

void vecquant4matmul_nuq_perchannel_transposed_mha_batched_fused_opt2_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table, int vcachelen,
  torch::Tensor outliers, torch::Tensor outlier_indices
);
void vecquant4matmul_nuq_perchannel_transposed_mha_batched_fused_opt2(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table, int vcachelen,
  torch::Tensor outliers, torch::Tensor outlier_indices
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant4matmul_nuq_perchannel_transposed_mha_batched_fused_opt2_cuda(vec, mat, mul, lookup_table, vcachelen, outliers, outlier_indices);
}



void vecquant3matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table, int kcachelen, float theta, int pos_offset
);
void vecquant3matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table, int kcachelen, float theta, int pos_offset
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant3matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt_cuda(vec, mat, mul, lookup_table, kcachelen, theta, pos_offset);
}

void vecquant3matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt2_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table, int kcachelen,
  torch::Tensor outliers, torch::Tensor outlier_indices, float theta, int pos_offset
);
void vecquant3matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt2(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table, int kcachelen,
  torch::Tensor outliers, torch::Tensor outlier_indices, float theta, int pos_offset
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant3matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt2_cuda(vec, mat, mul, lookup_table, kcachelen, outliers, outlier_indices, theta, pos_offset);
}

void vecquant3matmul_nuq_perchannel_transposed_mha_batched_fused_opt_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table, int vcachelen
);
void vecquant3matmul_nuq_perchannel_transposed_mha_batched_fused_opt(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table, int vcachelen
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant3matmul_nuq_perchannel_transposed_mha_batched_fused_opt_cuda(vec, mat, mul, lookup_table, vcachelen);
}

void vecquant3matmul_nuq_perchannel_transposed_mha_batched_fused_opt2_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table, int vcachelen,
  torch::Tensor outliers, torch::Tensor outlier_indices
);
void vecquant3matmul_nuq_perchannel_transposed_mha_batched_fused_opt2(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table, int vcachelen,
  torch::Tensor outliers, torch::Tensor outlier_indices
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant3matmul_nuq_perchannel_transposed_mha_batched_fused_opt2_cuda(vec, mat, mul, lookup_table, vcachelen, outliers, outlier_indices);
}


void vecquant2matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table, int kcachelen, float theta, int pos_offset
);
void vecquant2matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table, int kcachelen, float theta, int pos_offset
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant2matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt_cuda(vec, mat, mul, lookup_table, kcachelen, theta, pos_offset);
}

void vecquant2matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt2_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table, int kcachelen,
  torch::Tensor outliers, torch::Tensor outlier_indices, float theta, int pos_offset
);
void vecquant2matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt2(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table, int kcachelen,
  torch::Tensor outliers, torch::Tensor outlier_indices, float theta, int pos_offset
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant2matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt2_cuda(vec, mat, mul, lookup_table, kcachelen, outliers, outlier_indices, theta, pos_offset);
}

void vecquant2matmul_nuq_perchannel_transposed_mha_batched_fused_opt_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table, int vcachelen
);
void vecquant2matmul_nuq_perchannel_transposed_mha_batched_fused_opt(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table, int vcachelen
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant2matmul_nuq_perchannel_transposed_mha_batched_fused_opt_cuda(vec, mat, mul, lookup_table, vcachelen);
}

void vecquant2matmul_nuq_perchannel_transposed_mha_batched_fused_opt2_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table, int vcachelen,
  torch::Tensor outliers, torch::Tensor outlier_indices
);
void vecquant2matmul_nuq_perchannel_transposed_mha_batched_fused_opt2(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table, int vcachelen,
  torch::Tensor outliers, torch::Tensor outlier_indices
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant2matmul_nuq_perchannel_transposed_mha_batched_fused_opt2_cuda(vec, mat, mul, lookup_table, vcachelen, outliers, outlier_indices);
}

void vecquant4matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt2_orig_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table, int kcachelen,
  torch::Tensor rows, torch::Tensor cols, torch::Tensor startrows,
  torch::Tensor spmat, int num_rows, int num_threads, int nnz,
  float rope_theta, int pos_offset
);
void vecquant4matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt2_orig(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table, int kcachelen,
  torch::Tensor rows, torch::Tensor cols, torch::Tensor startrows,
  torch::Tensor spmat, int num_rows, int num_threads, int nnz,
  float rope_theta, int pos_offset
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant4matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt2_orig_cuda(vec, mat, mul, lookup_table, kcachelen, rows, cols, startrows, spmat, num_rows, num_threads, nnz, rope_theta, pos_offset);
}

void vecquant4matmul_nuq_perchannel_transposed_mha_batched_fused_opt2_orig_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table, int vcachelen,
  torch::Tensor rows, torch::Tensor cols, torch::Tensor startcols,
  torch::Tensor spmat, int num_rows, int num_threads, int nnz
);
void vecquant4matmul_nuq_perchannel_transposed_mha_batched_fused_opt2_orig(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table, int vcachelen,
  torch::Tensor rows, torch::Tensor cols, torch::Tensor startcols,
  torch::Tensor spmat, int num_rows, int num_threads, int nnz
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant4matmul_nuq_perchannel_transposed_mha_batched_fused_opt2_orig_cuda(vec, mat, mul, lookup_table, vcachelen, rows, cols, startcols, spmat, num_rows, num_threads, nnz);
}

std::vector<torch::Tensor> vecquant4appendvecKsparseorig_cuda(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, torch::Tensor zeropoint, torch::Tensor row, torch::Tensor col, torch::Tensor val, torch::Tensor start_rows, torch::Tensor outlier_threshold_lower, torch::Tensor outlier_threshold_upper, int kcachelen
);
std::vector<torch::Tensor> vecquant4appendvecKsparseorig(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, torch::Tensor zeropoint, torch::Tensor row, torch::Tensor col, torch::Tensor val, torch::Tensor start_rows, torch::Tensor outlier_threshold_lower, torch::Tensor outlier_threshold_upper, int kcachelen
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(mat));
  return vecquant4appendvecKsparseorig_cuda(mat, lookup_table, newvec, zeropoint, row, col, val, start_rows, outlier_threshold_lower, outlier_threshold_upper, kcachelen);
}

std::vector<torch::Tensor> vecquant4appendvecVsparseorig_cuda(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, float zeropoint, torch::Tensor row, torch::Tensor col, torch::Tensor val, torch::Tensor start_cols, float outlier_threshold_lower, float outlier_threshold_upper, int vcachelen
);
std::vector<torch::Tensor> vecquant4appendvecVsparseorig(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, float zeropoint, torch::Tensor row, torch::Tensor col, torch::Tensor val, torch::Tensor start_cols, float outlier_threshold_lower, float outlier_threshold_upper, int vcachelen
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(mat));
  return vecquant4appendvecVsparseorig_cuda(mat, lookup_table, newvec, zeropoint, row, col, val, start_cols, outlier_threshold_lower, outlier_threshold_upper, vcachelen);
}


BatchData execute_lz4compress(char* device_input_data, 
      const size_t in_bytes, const size_t batch_size, const size_t chunk_size) {
  // Start compression logic 
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // build up metadata
  BatchData input_data(device_input_data, in_bytes, batch_size, chunk_size, stream);
#ifdef PRINT
  std::cout << "in_bytes: " << in_bytes << "; batch_size: " << batch_size << "; chunk_size: " << chunk_size << std::endl;
#endif
  // record time
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start, stream);

  // Compress on the GPU using batched API
  size_t comp_temp_bytes;
  nvcompStatus_t status = nvcompBatchedLZ4CompressGetTempSize(
      input_data.size(),
      chunk_size,
      nvcompBatchedLZ4DefaultOpts,
      &comp_temp_bytes);
  if( status != nvcompSuccess){
    throw std::runtime_error("ERROR: nvcompBatchedLZ4CompressGetTempSize() not successful");
  }
  #ifdef PRINT
  std::cout << "comp_temp_bytes: " << comp_temp_bytes << std::endl;
#endif

  void* d_comp_temp;
  CUDA_CHECK(cudaMalloc(&d_comp_temp, comp_temp_bytes));

  size_t max_out_bytes;
  status = nvcompBatchedLZ4CompressGetMaxOutputChunkSize(
      chunk_size, nvcompBatchedLZ4DefaultOpts, &max_out_bytes);
  if( status != nvcompSuccess){
    throw std::runtime_error("ERROR: nvcompBatchedLZ4CompressGetMaxOutputChunkSize() not successful");
  }
  #ifdef PRINT
  std::cout << "max_out_bytes: " << max_out_bytes << std::endl;
#endif

  BatchData compress_data(max_out_bytes, input_data.size(), stream);
  #ifdef PRINT
  std::cout << input_data.size() << std::endl;
  std::cout << compress_data.ptrs() << "; " << compress_data.sizes() << std::endl;
  #endif

  status = nvcompBatchedLZ4CompressAsync(
      input_data.ptrs(),
      input_data.sizes(),
      chunk_size,
      input_data.size(),
      d_comp_temp,
      comp_temp_bytes,
      compress_data.ptrs(),
      compress_data.sizes(),
      nvcompBatchedLZ4DefaultOpts,
      stream);
  if (status != nvcompSuccess) {
    throw std::runtime_error("nvcompBatchedLZ4CompressAsync() failed.");
  }
 

  cudaEventRecord(end, stream);
  // CUDA_CHECK(cudaStreamSynchronize(stream));

  // free compression memory
  cudaFree(d_comp_temp);

  float ms;
  cudaEventElapsedTime(&ms, start, end);
  
  // compute compression ratio
  size_t * host_compressed_bytes;
  cudaMallocHost((void**)&host_compressed_bytes, sizeof(size_t) * batch_size);

  cudaMemcpy(
    host_compressed_bytes,
    compress_data.sizes(),
    sizeof(size_t) * batch_size,
    cudaMemcpyDeviceToHost);
  size_t comp_bytes = 0;
  for (size_t i = 0; i < batch_size; i ++) {
    comp_bytes += host_compressed_bytes[i];
  }
#ifdef PRINT
  std::cout << "comp_size: " << comp_bytes
            << ", compressed ratio: " << std::fixed << std::setprecision(2)
            << (double)in_bytes / comp_bytes << std::endl;
  std::cout << "compression time (ms): " << ms << std::endl;
#endif

  cudaEventDestroy(start);
  cudaEventDestroy(end);
  cudaStreamDestroy(stream);

  return compress_data;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
 lz4compress_wrapper(torch::Tensor input_tensor) {
  // Ensure the input tensor is on the correct device, has the correct data type, and is contiguous
  // input_tensor = input_tensor.contiguous().to(torch::kCPU, torch::kUInt8);
  CHECK_INPUT(input_tensor);
  #ifdef PRINT
  std::cout << "Total number of elements in input_tensor: " << input_tensor.numel() << std::endl;
#endif

  // Cast the tensor data (with INT) to char* for byte-wise compression
  char* device_input_data = reinterpret_cast<char*>(input_tensor.data_ptr<int>());

  // Calculate the total byte size of the tensor's data
  size_t in_bytes = input_tensor.numel() * sizeof(int);

  const size_t chunk_size = 1 << 16;
  const size_t batch_size = (in_bytes + chunk_size - 1) / chunk_size;

  // We make it to only one chunk
  // const size_t batch_size = 1;
  // const size_t chunk_size = in_bytes;

  auto comp_data = execute_lz4compress(device_input_data, in_bytes, batch_size, chunk_size);
  // return execute_lz4compress(device_input_data, in_bytes, batch_size, chunk_size);

  auto ptrs_tensor = comp_data.ptrs_tensor();
  auto data_tensor = comp_data.data_tensor();
  auto sizes_tensor = comp_data.sizes_tensor();
  return std::make_tuple(ptrs_tensor, data_tensor, sizes_tensor);
}

torch::Tensor lz4decompress_wrapper(std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> tensors, int64_t length) {
#ifdef PRINT
  std::cout << "length: " << length << std::endl;
#endif
  const size_t chunk_size = 1 << 16;

  auto [ptrs_tensor, data_tensor, sizes_tensor] = tensors;

  BatchData compress_data(ptrs_tensor, data_tensor, sizes_tensor);
  // Decompression can be similarly performed on a batch of multiple compressed input chunks. 
  // As no metadata is stored with the compressed data, chunks can be re-arranged as well as decompressed 
  // with other chunks that originally were not compressed in the same batch.
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // CUDA events to measure decompression time
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  // If we didn't have the uncompressed sizes, we'd need to compute this information here. 
  // We demonstrate how to do this.
  size_t * uncomp_sizes;
  cudaMalloc((void**)&uncomp_sizes, sizeof(size_t) * compress_data.size());
  #ifdef PRINT
  std::cout << compress_data.ptrs() << "; " << std::endl;
  std::cout << compress_data.sizes() << "; " << std::endl;
  std::cout << uncomp_sizes << "; " << std::endl;
  #endif

  nvcompBatchedLZ4GetDecompressSizeAsync(
      compress_data.ptrs(),
      compress_data.sizes(),
      uncomp_sizes,
      compress_data.size(),
      stream);
  BatchData decomp_data(uncomp_sizes, compress_data.size(), chunk_size, stream);

  // Next, allocate the temporary buffer 
  size_t decomp_temp_bytes;
  nvcompStatus_t status = nvcompBatchedLZ4DecompressGetTempSize(
      compress_data.size(), chunk_size, &decomp_temp_bytes);
  if (status != nvcompSuccess) {
    throw std::runtime_error("nvcompBatchedLZ4DecompressGetTempSize() failed.");
  }
  void* d_decomp_temp;
  CUDA_CHECK(cudaMalloc(&d_decomp_temp, decomp_temp_bytes));
  
  nvcompStatus_t* d_status_ptrs;
  CUDA_CHECK(cudaMalloc(
      (void**)&d_status_ptrs, compress_data.size() * sizeof(nvcompStatus_t)));

  // Also allocate an array to store the actual_uncompressed_bytes.
  // Note that we could use nullptr for this. We already have the 
  // actual sizes computed during the call to nvcompBatchedLZ4GetDecompressSizeAsync.
  size_t* d_decomp_sizes;
  cudaMalloc(&d_decomp_sizes, sizeof(size_t) * compress_data.size());

  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Run decompression
  status = nvcompBatchedLZ4DecompressAsync(
      compress_data.ptrs(),
      compress_data.sizes(),
      decomp_data.sizes(),
      d_decomp_sizes,
      compress_data.size(),
      d_decomp_temp,
      decomp_temp_bytes,
      decomp_data.ptrs(),
      d_status_ptrs,
      stream);
  if( status != nvcompSuccess){
    throw std::runtime_error("ERROR: nvcompBatchedLZ4DecompressAsync() not successful");
  }

  CUDA_CHECK(cudaStreamSynchronize(stream));

  // float ms;
  // cudaEventElapsedTime(&ms, start, end);
  // std::cout << "decompression time(ms): " << ms << std::endl;

  // cudaFree(d_decomp_temp);

  // cudaEventDestroy(start);
  // cudaEventDestroy(end);
  // cudaStreamDestroy(stream);

  return decomp_data.data_char_to_int(length);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("vecquant4matmul_nuq_perchannel_transposed_mha_batched_fused_opt", &vecquant4matmul_nuq_perchannel_transposed_mha_batched_fused_opt, "4-bit value cache matrix-vector operation");
  m.def("vecquant4matmul_nuq_perchannel_transposed_mha_batched_fused_opt2", &vecquant4matmul_nuq_perchannel_transposed_mha_batched_fused_opt2, "4-bit value cache matrix-vector operation (including sparsity)");
  m.def("vecquant4matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt", &vecquant4matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt, "4-bit key cache matrix-vector operation");
  m.def("vecquant4matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt2", &vecquant4matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt2, "4-bit key cache matrix-vector operation (including sparsity)");
  m.def("vecquant4appendvecK", &vecquant4appendvecK, "Append 4-bit key vector to the key cache");
  m.def("vecquant4appendvecKsparse", &vecquant4appendvecKsparse, "Append 4-bit key vector to the key cache (including sparsity)");
  m.def("vecquant4appendvecKsparseParallel", &vecquant4appendvecKsparseParallel, "Append 4-bit key vectors to the key cache (including sparsity)");
  m.def("vecquant4appendvecV", &vecquant4appendvecV, "Append 4-bit value vector to the value cache");
  m.def("vecquant4appendvecVsparse", &vecquant4appendvecVsparse, "Append 4-bit value vector to the value cache (including sparsity)");
  m.def("vecquant4appendvecVsparseParallel", &vecquant4appendvecVsparseParallel, "Append 4-bit value vectors to the value cache (including sparsity)");
  m.def("vecquant3matmul_nuq_perchannel_transposed_mha_batched_fused_opt", &vecquant3matmul_nuq_perchannel_transposed_mha_batched_fused_opt, "3-bit value cache matrix-vector operation");
  m.def("vecquant3matmul_nuq_perchannel_transposed_mha_batched_fused_opt2", &vecquant3matmul_nuq_perchannel_transposed_mha_batched_fused_opt2, "3-bit value cache matrix-vector operation (including sparsity)");
  m.def("vecquant3matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt", &vecquant3matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt, "3-bit key cache matrix-vector operation");
  m.def("vecquant3matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt2", &vecquant3matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt2, "3-bit key cache matrix-vector operation (including sparsity)");
  m.def("vecquant3appendvecK", &vecquant3appendvecK, "Append 3-bit key vector to the key cache");
  m.def("vecquant3appendvecKsparse", &vecquant3appendvecKsparse, "Append 3-bit key vector to the key cache (including sparsity)");
  m.def("vecquant3appendvecKsparseParallel", &vecquant3appendvecKsparseParallel, "Append 3-bit key vectors to the key cache (including sparsity)");
  m.def("vecquant3appendvecV", &vecquant3appendvecV, "Append 3-bit value vector to the value cache");
  m.def("vecquant3appendvecVsparse", &vecquant3appendvecVsparse, "Append 3-bit value vector to the value cache (including sparsity)");
  m.def("vecquant3appendvecVsparseParallel", &vecquant3appendvecVsparseParallel, "Append 3-bit value vectors to the value cache (including sparsity)");
  m.def("vecquant2matmul_nuq_perchannel_transposed_mha_batched_fused_opt", &vecquant2matmul_nuq_perchannel_transposed_mha_batched_fused_opt, "2-bit value cache matrix-vector operation");
  m.def("vecquant2matmul_nuq_perchannel_transposed_mha_batched_fused_opt2", &vecquant2matmul_nuq_perchannel_transposed_mha_batched_fused_opt2, "2-bit value cache matrix-vector operation (including sparsity)");
  m.def("vecquant2matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt", &vecquant2matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt, "2-bit key cache matrix-vector operation");
  m.def("vecquant2matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt2", &vecquant2matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt2, "2-bit key cache matrix-vector operation (including sparsity)");
  m.def("vecquant2appendvecK", &vecquant2appendvecK, "Append 2-bit key vector to the key cache");
  m.def("vecquant2appendvecKsparse", &vecquant2appendvecKsparse, "Append 2-bit key vector to the key cache (including sparsity)");
  m.def("vecquant2appendvecKsparseParallel", &vecquant2appendvecKsparseParallel, "Append 2-bit key vectors to the key cache (including sparsity)");
  m.def("vecquant2appendvecV", &vecquant2appendvecV, "Append 2-bit value vector to the value cache");
  m.def("vecquant2appendvecVsparse", &vecquant2appendvecVsparse, "Append 2-bit value vector to the value cache (including sparsity)");
  m.def("vecquant2appendvecVsparseParallel", &vecquant2appendvecVsparseParallel, "Append 2-bit value vectors to the value cache (including sparsity)");
  m.def("vecquant4matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt2_orig", &vecquant4matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt2_orig, "4-bit key cache matrix-vector operation");
  m.def("vecquant4appendvecKsparseorig", &vecquant4appendvecKsparseorig, "Append 4-bit key vector to the key cache (including sparsity)");
  m.def("vecquant4matmul_nuq_perchannel_transposed_mha_batched_fused_opt2_orig", &vecquant4matmul_nuq_perchannel_transposed_mha_batched_fused_opt2_orig, "4-bit value cache matrix-vector operation");
  m.def("vecquant4appendvecVsparseorig", &vecquant4appendvecVsparseorig, "Append 4-bit value vector to the value cache (including sparsity)");
  m.def("lz4compress", &lz4compress_wrapper, "LZ4 compression of a PyTorch tensor");
  m.def("lz4decompress", &lz4decompress_wrapper, "LZ4 decompression of a PyTorch tensor");
  // py::class_<BatchData>(m, "BatchData")
  //       .def(py::init<char*, size_t, size_t, size_t, cudaStream_t>(),
  //           py::arg("device_data"), py::arg("in_bytes"), py::arg("batch_size"), py::arg("chunk_size"), py::arg("stream"))
  //       .def("data", &BatchData::data)
  //       .def("ptrs", &BatchData::ptrs)
  //       .def("sizes", &BatchData::sizes)
  //       .def("size", &BatchData::size);
}
