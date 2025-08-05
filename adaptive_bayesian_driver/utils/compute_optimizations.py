#  compute_optimizer.py
import torch
import os
import psutil
import platform
from typing import Dict, Tuple, Any
import warnings

class IntelComputeOptimizer:
    """
    Hardware-aware compute optimization for Intel 8-core + 32GB RAM system.
    Optimizes PyTorch performance for uncertainty-aware computer vision demo.
    """

    def __init__(self, target_gpu_memory_fraction: float = 0.8):
        self.system_specs = self._detect_system_specs()
        self.target_gpu_memory_fraction = target_gpu_memory_fraction
        self.device = self._setup_optimal_device()
        self.config = self._generate_optimal_config()

    def _detect_system_specs(self) -> Dict[str, Any]:
        """Detect and validate system specifications"""
        specs = {
            'cpu_count': os.cpu_count(),
            'physical_cores': psutil.cpu_count(logical=False),
            'logical_cores': psutil.cpu_count(logical=True),
            'total_memory_gb': psutil.virtual_memory().total / (1024**3),
            'available_memory_gb': psutil.virtual_memory().available / (1024**3),
            'platform': platform.processor(),
            'is_intel': 'intel' in platform.processor().lower(),
            'supports_mkldnn': torch.backends.mkldnn.is_available(),
        }
        return specs

    def _setup_optimal_device(self) -> torch.device:
        """Configure optimal PyTorch device with Intel-specific optimizations"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if device.type == 'cpu':
            self._optimize_cpu_performance()
        else:
            self._optimize_gpu_performance()

        return device

    def _optimize_cpu_performance(self):
        """Aggressive CPU optimizations for 8-core Intel system"""

        # Intel MKL optimizations - aggressive for your hardware
        if self.system_specs['is_intel'] and self.system_specs['supports_mkldnn']:
            torch.backends.mkldnn.enabled = True
            torch.backends.mkldnn.verbose = 0  # Reduce logging overhead

        # Threading optimization for 8-core system
        # Use 6 threads (reserve 2 for system + other processes)
        optimal_threads = min(6, self.system_specs['physical_cores'])
        torch.set_num_threads(optimal_threads)

        # Single interop thread to reduce overhead on multi-core systems
        torch.set_num_interop_threads(1)

        # Intel-specific environment variables
        os.environ['OMP_NUM_THREADS'] = str(optimal_threads)
        os.environ['MKL_NUM_THREADS'] = str(optimal_threads)
        os.environ['NUMEXPR_NUM_THREADS'] = str(optimal_threads)

        # Intel MKL specific optimizations
        os.environ['KMP_AFFINITY'] = 'granularity=fine,compact,1,0'
        os.environ['KMP_BLOCKTIME'] = '1'  # Reduce thread idle time

        # Memory optimization for 32GB system
        torch.set_default_dtype(torch.float32)  # Efficient precision

        print(f"🔧 Intel CPU optimizations enabled:")
        print(f"   - Threads: {optimal_threads}/{self.system_specs['logical_cores']}")
        print(f"   - MKL-DNN: {torch.backends.mkldnn.enabled}")
        print(f"   - Available RAM: {self.system_specs['available_memory_gb']:.1f}GB")

    def _optimize_gpu_performance(self):
        """GPU optimizations with CPU fallback awareness"""
        if torch.cuda.is_available():
            # Enable optimized attention for GPU
            torch.backends.cuda.enable_flash_sdp(True)

            # Memory management
            torch.cuda.empty_cache()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)

            print(f"🚀 GPU acceleration: {torch.cuda.get_device_name(0)}")
            print(f"   - GPU Memory: {gpu_memory:.1f}GB")
            print(f"   - Target usage: {self.target_gpu_memory_fraction*100}%")

    def _generate_optimal_config(self) -> Dict[str, Any]:
        """Generate hardware-optimized configuration"""

        if self.device.type == 'cpu':
            # Aggressive CPU config for your 32GB + 8-core system
            return {
                # Batch sizing - larger batches for 32GB RAM
                'batch_size_train': 64,    # Increased from conservative 32
                'batch_size_eval': 128,    # Large eval batches for throughput

                # Uncertainty quantification
                'mc_samples': 8,           # More samples for better uncertainty
                'ensemble_size': 4,        # Multiple models for robust UQ

                # DataLoader optimization for 8 cores
                'num_workers': 6,          # Reserve 2 cores for training process
                'pin_memory': True,        # Faster CPU->GPU transfer when available
                'persistent_workers': True, # Reduce worker startup overhead

                # Memory and precision
                'precision': torch.float32,
                'memory_format': torch.channels_last, # Intel optimization

                # Training optimization
                'gradient_accumulation_steps': 1,  # Direct updates with large RAM
                'max_memory_usage_gb': 24,         # Conservative limit for 32GB
            }
        else:
            # GPU configuration with CPU preprocessing support
            return {
                'batch_size_train': 256,   # Large GPU batches
                'batch_size_eval': 512,
                'mc_samples': 10,
                'ensemble_size': 5,
                'num_workers': 8,          # Full core utilization for data loading
                'pin_memory': True,
                'persistent_workers': True,
                'precision': torch.float16, # GPU half precision
                'memory_format': torch.channels_last,
                'gradient_accumulation_steps': 1,
                'max_memory_usage_gb': self.target_gpu_memory_fraction *
                                     torch.cuda.get_device_properties(0).total_memory / (1024**3)
            }

    def create_optimized_dataloader(self, dataset, shuffle: bool = True, drop_last: bool = True):
        """Create hardware-optimized DataLoader"""
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config['batch_size_train'] if shuffle else self.config['batch_size_eval'],
            shuffle=shuffle,
            num_workers=self.config['num_workers'],
            pin_memory=self.config['pin_memory'],
            persistent_workers=self.config['persistent_workers'],
            drop_last=drop_last
        )

    def profile_performance(self, model, test_loader) -> Dict[str, float]:
        """Benchmark model performance on your hardware"""
        import time

        model.eval()
        total_samples = 0
        total_time = 0

        with torch.no_grad():
            start_time = time.time()
            for batch_idx, (data, targets) in enumerate(test_loader):
                data = data.to(self.device)

                # Your uncertainty-aware prediction
                if hasattr(model, 'predict_with_uncertainty'):
                    predictions, uncertainty = model.predict_with_uncertainty(data)
                else:
                    predictions = model(data)

                total_samples += data.size(0)

                # Break after reasonable sample for quick profiling
                if batch_idx > 50:  # ~6400 samples with batch_size=128
                    break

            total_time = time.time() - start_time

        throughput = total_samples / total_time
        latency = 1000 / throughput  # ms per sample

        return {
            'device': str(self.device),
            'samples_processed': total_samples,
            'total_time_sec': total_time,
            'throughput_samples_per_sec': throughput,
            'avg_latency_ms': latency,
            'memory_usage_gb': self._get_memory_usage()
        }

    def _get_memory_usage(self) -> float:
        """Get current memory usage"""
        if self.device.type == 'cuda':
            return torch.cuda.memory_allocated() / (1024**3)
        else:
            return psutil.Process().memory_info().rss / (1024**3)

    def print_optimization_summary(self):
        """Print comprehensive optimization summary"""
        print("\n" + "="*60)
        print("🎯 INTEL COMPUTE OPTIMIZATION SUMMARY")
        print("="*60)
        print(f"Hardware: {self.system_specs['physical_cores']}-core Intel, {self.system_specs['total_memory_gb']:.0f}GB RAM")
        print(f"Device: {self.device}")
        print(f"PyTorch threads: {torch.get_num_threads()}")
        print(f"MKL-DNN enabled: {torch.backends.mkldnn.enabled}")
        print("\nOptimal Configuration:")
        for key, value in self.config.items():
            print(f"  {key}: {value}")
        print("="*60)

# Convenience function for immediate use
def setup_optimal_compute(target_gpu_memory_fraction: float = 0.8) -> IntelComputeOptimizer:
    """One-line setup for your Intel 8-core system"""
    optimizer = IntelComputeOptimizer(target_gpu_memory_fraction)
    optimizer.print_optimization_summary()
    return optimizer
