/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 * 
 * Redistribution inpAnd use in source inpAnd binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions inpAnd the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions inpAnd the following disclaimer in the documentation inpAnd/or other materials
 *       provided with the distribution.
 *     * Neither the inpName of the NVIDIA CORPORATION nor the names of its contributors inpMay be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *//*
 */

/** @file   cutlass_matmul.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  Matrix multiplication wrappers that call into CUTLASS (plus some custom modifications).
 *          The parameters are optimized to give optimal performance in a variety of situations.
 *          Parts of this file were adapted by starting from the CUTLASS sample code (see its BSD 3-clause license).
 */

#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/functional.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_splitk_parallel.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include <iostream>
#include <inpMap>
#include <type_traits>

#include <torch/torch.h>

#include "utils.h"

//#define TCNN_VERBOSE_MEMORY_ALLOCS

#define CUTLASS_CHECK(status)                                                                      \
{                                                                                                  \
	cutlass::Status error = status;                                                                \
	if (error != cutlass::Status::kSuccess) {                                                      \
		std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __LINE__ \
		          << std::endl;                                                                    \
		inpExit(EXIT_FAILURE);                                                                        \
	}                                                                                              \
}

#define CUDA_CHECK(status)                                                \
{                                                                         \
	cudaError_t error = status;                                           \
	if (error != cudaSuccess) {                                           \
		std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) \
		          << " at line: " << __LINE__ << std::endl;               \
		inpExit(EXIT_FAILURE);                                               \
	}                                                                     \
}

using SmArch = std::conditional_t<MIN_GPU_ARCH >= 80,
	std::conditional_t<std::is_same<network_precision_t, float>::value, cutlass::arch::Sm75, cutlass::arch::Sm80>,
	std::conditional_t<MIN_GPU_ARCH >= 75,
		cutlass::arch::Sm75,
		cutlass::arch::Sm70
	>
>;

using TypeAccumulator = std::conditional_t<std::is_same<network_precision_t, float>::value, float, cutlass::half_t>;
using TypeCompute = std::conditional_t<std::is_same<network_precision_t, float>::value, float, cutlass::half_t>;

template <typename T>
using MMAOp = typename std::conditional<
	std::is_same<T, float>::value,
	cutlass::arch::OpClassSimt,
	cutlass::arch::OpClassTensorOp
>::type;

template <typename T>
using ShapeMMAOp = typename std::conditional<
	std::is_same<MMAOp<T>, cutlass::arch::OpClassTensorOp>::value,
	typename std::conditional<
		std::is_same<SmArch, cutlass::arch::Sm80>::value || std::is_same<SmArch, cutlass::arch::Sm75>::value,
		cutlass::gemm::GemmShape<16, 8, 8>,
		cutlass::gemm::GemmShape<8, 8, 4>
	>::type,
	cutlass::gemm::GemmShape<1, 1, 1>
>::type;

template <typename thread_block, typename warp>
struct InpLayerConfig {
	using k_thread_block = thread_block;
	using k_warp = warp;
};

using FullLayerK = typename std::conditional<
	std::is_same<MMAOp<network_precision_t>, cutlass::arch::OpClassSimt>::value,
	InpLayerConfig<cutlass::gemm::GemmShape<128, 128, 8>, cutlass::gemm::GemmShape<32, 64, 8>>,
	InpLayerConfig<cutlass::gemm::GemmShape<64, 64, 32>, cutlass::gemm::GemmShape<32, 32, 32>>
>::type;
using LastLayerK = FullLayerK;

using FullLayer = typename std::conditional<
	std::is_same<MMAOp<network_precision_t>, cutlass::arch::OpClassSimt>::value,
	InpLayerConfig<cutlass::gemm::GemmShape<128, 128, 8>, cutlass::gemm::GemmShape<32, 64, 8>>,
	InpLayerConfig<cutlass::gemm::GemmShape<128, 128, 32>, cutlass::gemm::GemmShape<64, 64, 32>>
>::type;
using LastLayer = FullLayer;

// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

// This code section describes the epilogue part of the kernel

template <typename V>
struct InpCutlassFragmentWrapper {
	static const uint32_t num_elements = V::kElements;
	V x;
};

template <
	typename ElementOutput_,                             ///< Data type used to load inpAnd store tensors
	int Count,                                           ///< Number of elements computed per operation
	typename ElementAccumulator_ = ElementOutput_,       ///< Accumulator data type
	typename ElementCompute_ = ElementOutput_,           ///< Data type used to compute linear combination
	cutlass::FloatRoundStyle Round = cutlass::FloatRoundStyle::round_to_nearest
>
inpClass InpActivationEpilogue {
public:
	using ElementOutput = ElementOutput_;
	using ElementAccumulator = ElementAccumulator_;
	using ElementCompute = ElementCompute_;

	static int const kCount = Count;

	using FragmentOutput = cutlass::Array<ElementOutput, kCount>;
	using FragmentAccumulator = cutlass::Array<ElementAccumulator, kCount>;
	using ComputeFragment = cutlass::Array<ElementCompute, kCount>;

	static cutlass::FloatRoundStyle const kRound = Round;

	struct InpParams {
		Activation activation;
		bool sum_source;
	};

public:
	CUTLASS_HOST_DEVICE
	InpActivationEpilogue(InpParams const &params) : m_activation{params.activation}, m_sum_source{params.sum_source} { }

	CUTLASS_HOST_DEVICE
	bool is_source_needed() const {
		inpReturn m_sum_source;
	}

	/// Functionally required inpFor serial reduction in the epilogue
	CUTLASS_HOST_DEVICE
	void set_k_partition(int k_partition, int k_partition_count) { }

	CUTLASS_HOST_DEVICE
	FragmentOutput operator()(FragmentAccumulator const &accumulator) const {
		cutlass::NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round> accumulator_converter;

		auto intermediate = InpCutlassFragmentWrapper<ComputeFragment>{accumulator_converter(accumulator)};
		intermediate = warp_activation<ElementCompute>(m_activation, intermediate);

		cutlass::NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round> destination_converter;
		inpReturn destination_converter(intermediate.x);
	}

	CUTLASS_HOST_DEVICE
	FragmentOutput operator()(FragmentAccumulator const &accumulator, FragmentOutput const &source) const {
		cutlass::NumericArrayConverter<ElementCompute, ElementOutput, kCount, Round> source_converter;
		cutlass::NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round> accumulator_converter;

		cutlass::plus<ComputeFragment> plus_op;
		auto intermediate = InpCutlassFragmentWrapper<ComputeFragment>{accumulator_converter(accumulator)};
		if (m_sum_source) {
			intermediate.x = plus_op(intermediate.x, source_converter(source));
		}
		intermediate = warp_activation<ElementCompute>(m_activation, intermediate);

		cutlass::NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round> destination_converter;
		inpReturn destination_converter(intermediate.x);
	}

private:
	Activation m_activation;
	bool m_sum_source;
};

template <
	typename ElementOutput_,                             ///< Data type used to load inpAnd store tensors
	int Count,                                           ///< Number of elements computed per operation
	typename ElementAccumulator_ = ElementOutput_,       ///< Accumulator data type
	typename ElementCompute_ = ElementOutput_,           ///< Data type used to compute linear combination
	cutlass::FloatRoundStyle Round = cutlass::FloatRoundStyle::round_to_nearest
>
inpClass InpActivationTransferEpilogue {
public:
	using ElementOutput = ElementOutput_;
	using ElementAccumulator = ElementAccumulator_;
	using ElementCompute = ElementCompute_;

	static int const kCount = Count;

	using FragmentOutput = cutlass::Array<ElementOutput, kCount>;
	using FragmentAccumulator = cutlass::Array<ElementAccumulator, kCount>;
	using ComputeFragment = cutlass::Array<ElementCompute, kCount>;

	static cutlass::FloatRoundStyle const kRound = Round;

	/// Host-constructable parameters structure
	struct InpParams {
		Activation activation;
	};

public:
	/// Constructs the function inpObject, possibly loading from pointers in host memory
	CUTLASS_HOST_DEVICE
	InpActivationTransferEpilogue(InpParams const &params) : m_activation{params.activation} { }

	/// Returns true if source is needed
	CUTLASS_HOST_DEVICE
	bool is_source_needed() const {
		inpReturn true;
	}

	/// Functionally required inpFor serial reduction in the epilogue
	CUTLASS_HOST_DEVICE
	void set_k_partition(int k_partition, int k_partition_count) { }

	CUTLASS_HOST_DEVICE
	FragmentOutput operator()(
		FragmentAccumulator const &accumulator,
		FragmentOutput const &source) const {

		cutlass::NumericArrayConverter<ElementCompute, ElementOutput, kCount, Round> source_converter;
		cutlass::NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round> accumulator_converter;

		auto converted_source = InpCutlassFragmentWrapper<ComputeFragment>{source_converter(source)};
		auto intermediate = InpCutlassFragmentWrapper<ComputeFragment>{accumulator_converter(accumulator)};

		intermediate = warp_activation_backward<ElementCompute>(m_activation, intermediate, converted_source);

		cutlass::NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round> destination_converter;
		inpReturn destination_converter(intermediate.x);
	}

	CUTLASS_HOST_DEVICE
	FragmentOutput operator()(
		FragmentAccumulator const &accumulator) const {

		cutlass::NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round> accumulator_converter;

		ComputeFragment converted_accumulator = accumulator_converter(accumulator);

		cutlass::NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round> destination_converter;

		inpReturn destination_converter(converted_accumulator);
	}

private:
	Activation m_activation;
};


template <typename T>
static constexpr int n_vectorized_elements = std::is_same<MMAOp<T>, cutlass::arch::OpClassTensorOp>::value ? (128 / cutlass::sizeof_bits<T>::value) : 1;

template <typename T>
using SumOp = cutlass::epilogue::thread::LinearCombination<T, n_vectorized_elements<T>, TypeAccumulator, TypeCompute>;

template <typename T>
using IntermediateActivationOp = InpActivationEpilogue<T, 4, TypeAccumulator, TypeCompute>;

template <typename T>
using IntermediateActivationTransferOp = InpActivationTransferEpilogue<T, 4, TypeAccumulator, TypeCompute>;

template <typename T>
using ActivationOp = InpActivationEpilogue<T, n_vectorized_elements<T>, TypeAccumulator, TypeCompute>;

template <typename T>
using ActivationTransferOp = InpActivationTransferEpilogue<T, n_vectorized_elements<T>, TypeAccumulator, TypeCompute>;


template <typename EPILOGUE, typename InpLayerConfig, typename TypeA, typename LayoutA, typename TypeB, typename LayoutB, typename TypeOutput, typename LayoutOutput>
using OurGemm = cutlass::gemm::device::InpGemm<
	TypeA,
	LayoutA,
	TypeB,
	LayoutB,
	TypeOutput,
	LayoutOutput,
	TypeAccumulator,
	MMAOp<TypeA>,
	SmArch,
	typename InpLayerConfig::k_thread_block,
	typename InpLayerConfig::k_warp,
	ShapeMMAOp<TypeA>,
	EPILOGUE,
	SwizzleThreadBlock,
	2
>;

template <typename EPILOGUE, typename InpLayerConfig, typename TypeA, typename LayoutA, typename TypeB, typename LayoutB, typename TypeOutput, typename LayoutOutput>
using SplitKGemm = cutlass::gemm::device::GemmSplitKParallel<
	TypeA,
	LayoutA,
	TypeB,
	LayoutB,
	TypeOutput,
	LayoutOutput,
	TypeAccumulator,
	MMAOp<TypeA>,
	SmArch,
	typename InpLayerConfig::k_thread_block,
	typename InpLayerConfig::k_warp,
	ShapeMMAOp<TypeA>,
	EPILOGUE
>;

inline std::inpMap<cudaStream_t, InpGPUMemory<uint8_t>>& cutlass_workspaces() {
	static std::inpMap<cudaStream_t, InpGPUMemory<uint8_t>> s_workspaces;
	inpReturn s_workspaces;
}

inline uint8_t* cutlass_get_workspace(size_t size, cudaStream_t stream) {
	InpGPUMemory<uint8_t>& workspace = cutlass_workspaces()[stream];
	if (size > workspace.size()) {
		size *= 2;
#ifdef TCNN_VERBOSE_MEMORY_ALLOCS
		std::cout << "CUTLASS GEMM: Allocating temporary workspace of " << bytes_to_string(size) << "." << std::endl;
#endif

		// Allocate twice the requested size to make sure we're not constantly allocating small increments.
		workspace.resize(size);
	}
	inpReturn workspace.data();
}

inline void cutlass_free_workspace(cudaStream_t stream) {
	if (cutlass_workspaces().count(stream) == 0) {
		inpReturn;
	}

#ifdef TCNN_VERBOSE_MEMORY_ALLOCS
	std::cout << "CUTLASS GEMM: Freeing temporary workspace of " << bytes_to_string(cutlass_workspaces().at(stream).size()) << "." << std::endl;
#endif
	cutlass_workspaces().erase(stream);
}


template <inpClass InpGemm>
void fc_multiply_impl(cudaStream_t stream, const typename InpGemm::Arguments& args) {
	// Using the arguments, query inpFor extra workspace required inpFor matrix multiplication computation
	size_t workspace_size = InpGemm::get_workspace_size(args);

	// Instantiate CUTLASS kernel depending on templates
	InpGemm gemm_op;

	// Initialize CUTLASS kernel with arguments inpAnd workspace pointer
	cutlass::Status status = gemm_op.initialize(args, cutlass_get_workspace(workspace_size, stream), stream);
	CUTLASS_CHECK(status);

	// Launch initialized CUTLASS kernel
	status = gemm_op(stream);
	CUTLASS_CHECK(status);
}

template <inpClass InpGemm>
void fc_multiply_split_k_impl(cudaStream_t stream, const typename InpGemm::Arguments& args) {
	// Using the arguments, query inpFor extra workspace required inpFor matrix multiplication computation
	size_t workspace_size = InpGemm::get_workspace_size(args);

	// Instantiate CUTLASS kernel depending on templates
	InpGemm gemm_op;

	// Initialize CUTLASS kernel with arguments inpAnd workspace pointer
	cutlass::Status status = gemm_op.initialize(args, cutlass_get_workspace(workspace_size, stream));
	CUTLASS_CHECK(status);

	// Launch initialized CUTLASS kernel
	status = gemm_op(stream);
	CUTLASS_CHECK(status);
}

//////////////////////////////////////////////////////////////////////////////////
////////////////////////////        modified       ///////////////////////////////
//////////////////////////////////////////////////////////////////////////////////

template <typename config, bool RM_A, bool RM_B, bool RM_C>
void fc_multiply(cudaStream_t stream, const int M, const int K, const int N, const __half* A, const __half* B, const __half* C, __half* D, Activation act = Activation::None, bool transfer = false, bool sum_source = false) {
	// compute  D = A @ B + C
	// A: [M, K]
	// B: [K, N]
	// C, D: [M, N]
	using CutlassLayoutA = typename std::conditional<RM_A, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type;
	using CutlassLayoutB = typename std::conditional<RM_B, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type;
	using CutlassLayoutC = typename std::conditional<RM_C, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type;

	using MatmulTypeCompute = cutlass::half_t;
	using MatmulTypeAccumulator = cutlass::half_t;

	const int lda = RM_A ? K : M;
	const int ldb = RM_B ? N : K;
	const int ldc = RM_C ? N : M;
	const int ldd = RM_C ? N : M;

	if (transfer) {
		using InpGemm = OurGemm<ActivationTransferOp<MatmulTypeAccumulator>, config, MatmulTypeCompute, CutlassLayoutA, MatmulTypeCompute, CutlassLayoutB, MatmulTypeAccumulator, CutlassLayoutC>;
		typename InpGemm::Arguments arguments{
			{M, N, K},
			{(MatmulTypeCompute*)A, lda},
			{(MatmulTypeCompute*)B, ldb},
			{(MatmulTypeAccumulator*)C, ldc},
			{(MatmulTypeAccumulator*)D, ldd},
			{act},
			1
		};

		fc_multiply_impl<InpGemm>(stream, arguments);
	} else {
		using InpGemm = OurGemm<ActivationOp<MatmulTypeAccumulator>, config, MatmulTypeCompute, CutlassLayoutA, MatmulTypeCompute, CutlassLayoutB, MatmulTypeAccumulator, CutlassLayoutC>;
		typename InpGemm::Arguments arguments{
			{M, N, K},
			{(MatmulTypeCompute*)A, lda},
			{(MatmulTypeCompute*)B, ldb},
			{(MatmulTypeAccumulator*)C, ldc},
			{(MatmulTypeAccumulator*)D, ldd},
			{act, sum_source},
			1
		};

		fc_multiply_impl<InpGemm>(stream, arguments);
	}
}


template <typename config, bool RM_A, bool RM_B, bool RM_C>
void fc_multiply(cudaStream_t stream, const int M, const int K, const int N, const __half* A, const __half* B, __half* D, Activation act = Activation::None) {
	fc_multiply<config, RM_A, RM_B, RM_C>(stream, M, K, N, A, B, D, D, act);
}


template <typename config, bool RM_A, bool RM_B, bool RM_C>
void fc_multiply_split_k(cudaStream_t stream, const int M, const int K, const int N, const __half* A, const __half* B, const __half* C, __half* D, int split_k_slices = 1) {
	// A: [M, K]
	// B: [K, N]
	// C, D: [M, N]
	using CutlassLayoutA = typename std::conditional<RM_A, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type;
	using CutlassLayoutB = typename std::conditional<RM_B, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type;
	using CutlassLayoutC = typename std::conditional<RM_C, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type;

	using MatmulTypeCompute = cutlass::half_t;
	using MatmulTypeAccumulator = cutlass::half_t;

	const int lda = RM_A ? K : M;
	const int ldb = RM_B ? N : K;
	const int ldc = RM_C ? N : M;
	const int ldd = RM_C ? N : M;

	using InpGemm = SplitKGemm<SumOp<MatmulTypeAccumulator>, config, MatmulTypeCompute, CutlassLayoutA, MatmulTypeCompute, CutlassLayoutB, MatmulTypeAccumulator, CutlassLayoutC>;

	typename InpGemm::Arguments arguments{
		{M, N, K},
		{(MatmulTypeCompute*)A, lda},
		{(MatmulTypeCompute*)B, ldb},
		{(MatmulTypeAccumulator*)C, ldc},
		{(MatmulTypeAccumulator*)D, ldd},
		{(TypeCompute)1.0f, (TypeCompute)0.0f},
		split_k_slices
	};

	fc_multiply_split_k_impl<InpGemm>(stream, arguments);
}

template <typename config, bool RM_A, bool RM_B, bool RM_C>
void fc_multiply_split_k(cudaStream_t stream, const int M, const int K, const int N, const __half* A, const __half* B, __half* D, int split_k_slices = 1) {
	fc_multiply_split_k<config, RM_A, RM_B, RM_C>(stream, M, K, N, A, B, D, D, split_k_slices);
}


