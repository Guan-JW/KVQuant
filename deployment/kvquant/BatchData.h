#pragma once
#include <torch/all.h>
#include <torch/python.h>
#include <c10/cuda/CUDAGuard.h>
#include "utils.h"

#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>

class BatchData
{
public:
    BatchData(
        char* device_data,
        const size_t in_bytes,
        const size_t batch_size,
        const size_t chunk_size,
        cudaStream_t stream) :
        m_ptrs(),
        m_sizes(),
        m_data(device_data),
        m_size(batch_size),
        max_out_bytes(0)
    {
        // Setup an array of pointers to the start of each chunk
        void** host_uncompressed_ptrs;
        cudaMallocHost((void**)&host_uncompressed_ptrs, sizeof(size_t) * batch_size);
        for (size_t i = 0; i < batch_size; ++i) {
            host_uncompressed_ptrs[i] = m_data + chunk_size * i;
        }

        cudaMalloc((void**)&m_ptrs, sizeof(size_t) * batch_size);
        cudaMemcpyAsync(m_ptrs, host_uncompressed_ptrs, sizeof(size_t) * batch_size, cudaMemcpyHostToDevice, stream);

        size_t* host_uncompressed_bytes;
        cudaMallocHost((void**)&host_uncompressed_bytes, sizeof(size_t) * batch_size);
        for (size_t i = 0; i < batch_size; ++i) {
            if (i + 1 < batch_size) {
                host_uncompressed_bytes[i] = chunk_size;
            } else {
                // last chunk may be smaller
                host_uncompressed_bytes[i] = in_bytes - (chunk_size*i);
            }
        }
        
        cudaMalloc((void**)&m_sizes, sizeof(size_t) * batch_size);
        cudaMemcpyAsync(m_sizes, host_uncompressed_bytes, sizeof(size_t) * batch_size, cudaMemcpyHostToDevice, stream);

        // Free host memory
        cudaStreamSynchronize(stream);
        cudaFreeHost(host_uncompressed_ptrs);
        cudaFreeHost(host_uncompressed_bytes);
    }

    BatchData(const size_t max_out_bytes, const size_t batch_size, cudaStream_t stream) :
        m_ptrs(),
        m_sizes(),
        m_data(),
        m_size(batch_size),
        max_out_bytes(max_out_bytes)
    {
        cudaMalloc((void**)&m_data, max_out_bytes * batch_size);

        // allocate output space on the device
        void ** host_compressed_ptrs;
        cudaMallocHost((void**)&host_compressed_ptrs, sizeof(size_t) * batch_size);
        for(size_t ix_chunk = 0; ix_chunk < batch_size; ++ix_chunk) {
            cudaMalloc(&host_compressed_ptrs[ix_chunk], max_out_bytes);
        }

        cudaMalloc((void**)&m_ptrs, sizeof(size_t) * batch_size);
        cudaMemcpyAsync(
            m_ptrs, host_compressed_ptrs, 
            sizeof(size_t) * batch_size,cudaMemcpyHostToDevice, stream);

        // allocate space for compressed chunk sizes to be written to
        cudaMalloc((void**)&m_sizes, sizeof(size_t) * batch_size);

        // Free host memory
        // cudaStreamSynchronize(stream);
        // cudaFreeHost(host_compressed_ptrs);
    }

    BatchData(torch::Tensor ptrs_tensor, 
        torch::Tensor data_tensor, torch::Tensor sizes_tensor) {
        CHECK_CUDA(ptrs_tensor);
        CHECK_CUDA(data_tensor);
        CHECK_CUDA(sizes_tensor);

        m_size = sizes_tensor.size(0);
        m_data = static_cast<char*>(data_tensor.data_ptr());
        m_sizes = static_cast<size_t*>(sizes_tensor.data_ptr());
        m_ptrs = static_cast<void**>(ptrs_tensor.data_ptr());
    }

    BatchData(size_t * uncomp_sizes, size_t batch_size,
                size_t chunk_size, cudaStream_t stream):
        m_sizes(uncomp_sizes),
        m_size(batch_size) 
    {
        // get total data size
        // Wrap the raw pointer in a thrust::device_ptr
        size_t * host_sizes;
        cudaMallocHost((void**)&host_sizes, sizeof(size_t) * batch_size);
        cudaMemcpy(host_sizes, uncomp_sizes, sizeof(size_t) * batch_size, cudaMemcpyDeviceToHost);
        size_t total_size = 0;
        for (size_t i = 0; i < batch_size; i ++) {
            total_size += host_sizes[i];
        }
        // thrust::device_ptr<size_t> dev_ptr(uncomp_sizes);
        // size_t total_size = thrust::reduce(thrust::device, dev_ptr, 
        //     dev_ptr + batch_size, 
        //     (size_t)0, 
        //     thrust::plus<size_t>());
        #ifdef PRINT
        std::cout << "total_size: " << total_size << std::endl;
        #endif

        
        cudaMalloc((void**)&m_data, total_size);

        // set pointers for each chunk
        void** host_uncompressed_ptrs;
        cudaMallocHost((void**)&host_uncompressed_ptrs, sizeof(size_t) * batch_size);
        for (size_t i = 0; i < batch_size; i ++) {
            host_uncompressed_ptrs[i] = data() + chunk_size * i;
        }

        cudaMalloc((void**)&m_ptrs, sizeof(size_t) * batch_size);
        cudaMemcpyAsync(m_ptrs, host_uncompressed_ptrs, sizeof(size_t) * batch_size, cudaMemcpyHostToDevice, stream);

    }

    char* data() 
    {
        return m_data;
    }

    void** ptrs()
    {
        return m_ptrs;
    }

    size_t* sizes()
    {
        return m_sizes;
    }

    size_t size() const
    {
        return m_size;
    }

    // Return data as a torch::Tensor
    torch::Tensor data_tensor() const {
        auto options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);    // kUInt8 for char
        return torch::from_blob(m_data, {static_cast<int64_t>(m_size)}, options);
    }

    // Return sizes as a torch::Tensor
    torch::Tensor sizes_tensor() const {
        auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA); // kInt64 for size_t
        return torch::from_blob(m_sizes, {static_cast<int64_t>(m_size)}, options);
    }

    torch::Tensor ptrs_tensor() const {
        auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA); // kInt64 for void*
        return torch::from_blob(m_ptrs, {static_cast<int64_t>(m_size)}, options);
    }

    torch::Tensor data_char_to_int(int64_t length) {
        auto options = torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA);
        return torch::from_blob(m_data, {length}, options);
    }


private:
    void ** m_ptrs;
    size_t * m_sizes;
    char * m_data;
    size_t m_size;  // batch size
    size_t max_out_bytes;
};
