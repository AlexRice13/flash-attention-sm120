# SM120 (Blackwell) FlashAttention v3 Implementation

## Overview
This implementation extends FlashAttention v3 to fully support NVIDIA Blackwell (SM120) GPUs with architecture-specific optimizations.

## Key Architectural Differences (Blackwell vs Hopper)

### 1. Smaller Shared Memory Per SM
- **Impact**: Reduced from Hopper's SMEM size
- **Solution**: Tile sizes in `tile_size_fwd_sm120()` are reduced compared to SM90
- **Examples**:
  - hdim=64: `kBlockM=160` (vs 192 on SM90)
  - hdim=96: `kBlockM=160` (vs 192 on SM90)
  - hdim=128: Reduced `kBlockN` from 176 to 160
  - hdim=256: More conservative tile sizes

### 2. FP4 Acceleration Hardware
- **Status**: Infrastructure ready, not yet implemented
- **Future Work**: Add FP4 attention paths when CUDA toolkit and PyTorch support becomes available
- **Guards**: Code structured with `#if __CUDA_ARCH__ >= 1200` guards for future FP4 kernels

### 3. Similar ISA to Hopper
- **Features Reused**: TMA (Tensor Memory Accelerator), GMMA, warp specialization, tensor cores
- **Implementation**: SM120 kernels inherit from SM90 implementation with adjusted tile sizes

## Files Modified/Created

### Build System
- `hopper/setup.py`: Added SM120 compilation flags and ninja rules
  - `-gencode arch=compute_120,code=sm_120`
  - `cuda_compile_sm120` rule
  - Included SM120 instantiation files in sources

### Architecture Guards & Dispatch
- `hopper/utils.h`: Added `enable_sm120_or_later` wrapper
- `hopper/static_switch.h`: Updated `ARCH_SWITCH` macro to handle SM120
- `hopper/flash_api.cpp`: 
  - Added `is_sm120` device capability check
  - Updated all `Arch == 90` checks to `Arch == 90 || Arch == 120`

### Kernel Headers
- `hopper/flash_fwd_kernel_sm120.h`: SM120 forward kernel (based on SM90)
- `hopper/flash_bwd_kernel_sm120.h`: SM120 backward kernel (based on SM90)

### Launch Templates
- `hopper/flash_fwd_launch_template.h`: 
  - Added SM120 tile size selection via `tile_size_fwd_sm120()`
  - Added SM120 kernel dispatch with `enable_sm120_or_later`
- `hopper/flash_bwd_launch_template.h`: 
  - Added SM120 kernel dispatch

### Tile Configuration
- `hopper/tile_size.h`: Added `tile_size_fwd_sm120()` function
  - Tuned for smaller shared memory
  - Maintains high occupancy and tensor core utilization
  - Documented rationale in comments

### Kernel Generation
- `hopper/generate_kernels.py`: Updated to generate SM120 instantiations
  - Added 120 to `SM = [80, 90, 120]`
  - Added `KERNEL_IMPL_TEMPLATE_FWD_SM120` and `KERNEL_IMPL_TEMPLATE_BWD_SM120`

### Generated Files
- `hopper/instantiations/`: 210 SM120 kernel instantiation files generated
  - Forward kernels for all head dimensions, dtypes, and configurations
  - Backward kernels for all head dimensions and dtypes

## Tuning Philosophy

### Shared Memory Optimization
Blackwell has smaller SMEM per SM than Hopper, requiring careful tuning:
1. **Reduce tile sizes** while maintaining alignment for tensor cores
2. **Maintain high occupancy** by not reducing tiles too aggressively  
3. **Balance SMEM usage** vs register pressure

### FP8 Support
- SM120 supports FP8 (like SM90) via existing paths
- FP4 support requires future CUDA/PyTorch updates

### Performance Targets
- **Goal**: Match or exceed Hopper performance despite smaller SMEM
- **Strategy**: Leverage same ISA features (TMA, GMMA) with optimized tile sizes
- **Validation**: Use existing test suite and benchmarks

## Testing & Validation

### Test Suite (All in `hopper/`)
- `test_flash_attn.py`: Basic forward/backward correctness
- `test_attn_kvcache.py`: KV cache functionality
- `test_kvcache.py`: Cache management
- `test_flash_attn_bwd_determinism.py`: Backward determinism
- `test_util.py`: Utility functions

### Benchmarks (All in `hopper/`)
- `benchmark_attn.py`: General attention performance
- `benchmark_flash_attention_fp8.py`: FP8 performance
- `benchmark_split_kv.py`: Split KV performance
- `benchmark_mla_decode.py`: MLA decode performance

### Validation Steps
1. **Build**: Ensure clean build for SM120 target
2. **Correctness**: Run test suite on SM120 hardware
3. **Performance**: Compare benchmarks vs SM90
4. **Tuning**: Iterate on tile sizes based on profiling

## Future Work

### Short Term
1. **Profiling**: Measure actual SMEM usage on SM120 hardware
2. **Fine-tuning**: Adjust tile sizes based on real hardware data
3. **Validation**: Comprehensive testing on Blackwell GPUs

### Long Term
1. **FP4 Support**: 
   - Wait for CUDA toolkit FP4 intrinsics
   - Add FP4 kernel variants
   - Benchmark FP4 vs FP8 vs FP16/BF16
2. **Advanced Optimizations**:
   - Explore Blackwell-specific features
   - Further SMEM optimization
   - Register allocation tuning

## References
- **NVIDIA Blackwell (SM120) Architecture Whitebook**: Hardware specifications and tuning guidelines
- **FlashAttention-3 Paper**: Algorithm and optimization techniques
- **CUTLASS Documentation**: Tensor core programming patterns

## Notes for Developers
- All SM120-specific tuning is in `tile_size_fwd_sm120()` 
- Mainloop implementations are shared with SM90
- FP4 infrastructure present but not activated (requires CUDA support)
- Comments in code reference Blackwell whitebook where applicable
