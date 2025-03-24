import onnxruntime
import logging
import platform
import sys
import numpy as np

logger = logging.getLogger(__name__)

def get_onnx_runtime_info():
    """
    Get detailed information about the ONNX Runtime installation.
    
    Returns:
        dict: Dictionary containing ONNX Runtime information
    """
    info = {
        "version": onnxruntime.__version__,
        "available_providers": onnxruntime.get_available_providers(),
        "device": platform.platform(),
        "python_version": sys.version,
    }
    
    # Check if CUDA is available through torch if installed
    try:
        import torch
        info["torch_cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["torch_cuda_device_count"] = torch.cuda.device_count()
            info["torch_cuda_device_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        info["torch_cuda_available"] = "torch not installed"
    
    return info

def get_available_providers():
    """
    Get the available execution providers in the current onnxruntime installation.
    
    Returns:
        list: List of available provider names
    """
    return onnxruntime.get_available_providers()

def check_provider_performance(model_path):
    """
    Run a simple benchmark to check performance of different providers.
    This helps identify if GPU acceleration is working correctly.
    
    Args:
        model_path (str): Path to an ONNX model to test
    """
    available_providers = get_available_providers()
    logger.info(f"Testing performance for available providers: {available_providers}")
    
    # Create a simple test input
    try:
        # Create sessions for each provider
        for provider in available_providers:
            try:
                sess_options = onnxruntime.SessionOptions()
                sess = onnxruntime.InferenceSession(
                    model_path, 
                    sess_options=sess_options,
                    providers=[provider]
                )
                
                # Get input shape from model
                input_name = sess.get_inputs()[0].name
                input_shape = sess.get_inputs()[0].shape
                
                # Create random input data of appropriate shape
                # Replace any dynamic dimensions (None) with a reasonable value
                concrete_shape = [dim if dim is not None else 1 for dim in input_shape]
                dummy_input = np.random.rand(*concrete_shape).astype(np.float32)
                
                # Run inference and time it
                import time
                start_time = time.time()
                num_runs = 10
                
                for _ in range(num_runs):
                    sess.run(None, {input_name: dummy_input})
                
                elapsed_time = time.time() - start_time
                logger.info(f"Provider {provider}: {num_runs} runs took {elapsed_time:.4f}s ({elapsed_time/num_runs:.4f}s per run)")
                
            except Exception as e:
                logger.error(f"Error benchmarking provider {provider}: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to run performance check: {str(e)}")

def create_inference_session(model_path, use_gpu=True):
    """
    Creates an optimized ONNX Runtime inference session with GPU acceleration if available.
    
    Args:
        model_path (str): Path to the ONNX model file
        use_gpu (bool): Whether to attempt to use GPU acceleration (default: True)
        
    Returns:
        onnxruntime.InferenceSession: The created session
    """
    # Log detailed ONNX Runtime info
    runtime_info = get_onnx_runtime_info()
    logger.info(f"ONNX Runtime version: {runtime_info['version']}")
    logger.info(f"Python version: {runtime_info['python_version']}")
    logger.info(f"Device: {runtime_info['device']}")
    
    if 'torch_cuda_available' in runtime_info:
        if runtime_info['torch_cuda_available'] is True:
            logger.info(f"CUDA is available via PyTorch. Devices: {runtime_info.get('torch_cuda_device_count', 'unknown')}")
            logger.info(f"CUDA device name: {runtime_info.get('torch_cuda_device_name', 'unknown')}")
        else:
            logger.info("CUDA is not available via PyTorch")
    
    # Create session options
    sess_options = onnxruntime.SessionOptions()
    
    # Enable all graph optimizations
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # Enable profiling for performance analysis
    # sess_options.enable_profiling = True
    
    # Get available providers
    available_providers = get_available_providers()
    providers = []
    
    # Log available providers
    logger.info(f"Available ONNX Runtime providers: {available_providers}")
    
    # Set up priority list for providers to try
    if use_gpu:
        if 'CUDAExecutionProvider' in available_providers:
            providers.append('CUDAExecutionProvider')
            logger.info("Using CUDA GPU acceleration for ONNX model")
        elif 'TensorrtExecutionProvider' in available_providers:
            providers.append('TensorrtExecutionProvider')
            logger.info("Using TensorRT GPU acceleration for ONNX model")
        elif 'DmlExecutionProvider' in available_providers:
            providers.append('DmlExecutionProvider')
            logger.info("Using DirectML GPU acceleration for ONNX model")
        elif 'ROCMExecutionProvider' in available_providers:
            providers.append('ROCMExecutionProvider')
            logger.info("Using ROCm GPU acceleration for ONNX model")
        elif 'AzureExecutionProvider' in available_providers:
            # Azure's provider can use GPUs in the cloud, but typically won't on local machine
            providers.append('AzureExecutionProvider')
            logger.info("Using Azure Execution Provider for ONNX model - note this may not use local GPU")
    
    # Always include CPU provider as fallback
    providers.append('CPUExecutionProvider')
    
    # Create the inference session
    logger.info(f"Creating inference session with providers: {providers}")
    session = onnxruntime.InferenceSession(
        model_path, 
        sess_options=sess_options,
        providers=providers
    )
    
    # Log which provider was actually used
    logger.info(f"Effective execution provider: {session.get_providers()}")
    
    return session 