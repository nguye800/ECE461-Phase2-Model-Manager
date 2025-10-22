import requests
from urllib.parse import urlparse
from metric import BaseMetric

# Device specifications for memory and storage (in MB)
DEVICE_SPECS = {
    "desktop_pc": {
        "storage_mb": 512000,  # 512GB typical
        "memory_mb": 32768,  # 32GB typical
        "available_memory_ratio": 0.7,  # 70% of memory can be used
    },
    "raspberry_pi": {
        "storage_mb": 32768,  # 32GB SD card typical
        "memory_mb": 8192,  # 8GB model
        "available_memory_ratio": 0.5,  # 50% of memory can be used
    },
    "aws_server": {
        "storage_mb": 1048576,  # 1TB EBS volume typical
        "memory_mb": 32768,  # 32GB instance typical
        "available_memory_ratio": 0.8,  # 80% of memory can be used
    },
    "jetson_nano": {
        "storage_mb": 32768,  # 32 GB SD Card
        "memory_mb": 4096,  # 4GB officially listed
        "available_memory_ratio": 0.9,  # typical overhead for headless linux box
    },
}

# Tensor type sizes in bytes per parameter
TENSOR_TYPE_SIZES = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "int8": 1,
    "int4": 0.5,
    "int32": 4,
    "int64": 8,
    "bool": 0.125,  # 1 bit = 0.125 bytes
    "default": 4,  # assume float32 if unknown
}


def _get_cfg(cfg, *names, default=None):
    for n in names:
        if n in cfg:
            return cfg[n]
    return default


class SizeMetric(BaseMetric):
    """
    Metric that evaluates model size against target device capabilities.

    Calculates storage and memory requirements, assigns score based on:
    - Storage fit (0 if doesn't fit, continue if fits)
    - Memory usage and remaining free memory
    """

    metric_name: str = "size_score"

    def __init__(self):
        """Initialize the SizeMetric with default values."""
        super().__init__()
        self.model_info = None
        self.storage_size_mb = 0
        self.memory_size_mb = 0
        self.param_source = None

    def setup_resources(self):
        """
        Set up resources by fetching model information from HuggingFace API.

        Raises:
            IOError: If model information cannot be retrieved.
        """
        try:
            self.model_info = self._fetch_model_info()
            self.param_source = self.model_info.get("param_source", None)
            self.storage_size_mb = self._calculate_storage_size()
            self.memory_size_mb = self._calculate_memory_size()
        except Exception as e:
            raise IOError(f"Failed to setup resources for size metric: {e}")

    def calculate_device_score(self, device_spec: dict):
        # Check if model fits on storage
        if self.storage_size_mb > device_spec["storage_mb"]:
            return 0.0

        # Check if there wasn't enough information to calculate memory requirements
        if hasattr(self, "param_source") and self.param_source == "fallback":
            return 0.0

        # Calculate available memory for model loading
        available_memory = (
            device_spec["memory_mb"] * device_spec["available_memory_ratio"]
        )

        # If model doesn't fit in memory, return very low score
        memory_used_ratio = self.memory_size_mb / available_memory

        if memory_used_ratio > 1.2:
            # Significantly exceeds memory
            return 0.01
        elif memory_used_ratio > 0.8:
            return max(0.05, 0.8 - ((memory_used_ratio - 0.8) / 0.4) * (0.8 - 0.05))
        else:
            return 1.0 - (memory_used_ratio / 0.8) * 0.2

    def calculate_score(self) -> dict[str, float]:
        """
        Calculate the size metric score based on storage and memory constraints.

        Returns:
            dict[str, float]: Score for each device between 0.0 and 1.0 where:
                   - 0.0: Model doesn't fit on storage
                   - 0.1-1.0: Model fits, score based on remaining memory percentage
        """
        return {
            platform: self.calculate_device_score(specs)
            for platform, specs in DEVICE_SPECS.items()
        }

    def _fetch_model_info(self) -> dict:
        """
        Fetch model information from HuggingFace API.

        Returns:
            dict: Model information including config and file details.

        Raises:
            requests.RequestException: If API request fails.
        """
        # Extract repo_id from URL
        repo_id = self._extract_repo_id_from_url()

        # Fetch model config
        config_url = f"https://huggingface.co/api/models/{repo_id}"

        try:
            response = requests.get(config_url, timeout=30)
            response.raise_for_status()
            model_data = response.json()

            # Also fetch the config.json for parameter count and tensor info
            config_json_url = (
                f"https://huggingface.co/{repo_id}/resolve/main/config.json"
            )
            config_response = requests.get(config_json_url, timeout=30)

            if config_response.status_code == 200:
                model_data["config"] = config_response.json()

            return model_data

        except requests.RequestException as e:
            # Fallback: try to estimate from URL or use conservative estimates
            return self._get_fallback_model_info()

    def _extract_repo_id_from_url(self) -> str:
        """
        Extract repository ID from HuggingFace URL.

        Returns:
            str: Repository ID
        """
        # Handle different URL formats
        if not self.url:
            raise ValueError("No Model URL Provided")
        if "huggingface.co/" in self.url.model:
            path = urlparse(self.url.model).path
            # Remove leading slash and extract repo path
            repo_path = path.lstrip("/")
            # Remove /tree/main or similar suffixes
            if "/tree/" in repo_path:
                repo_path = repo_path.split("/tree/")[0]
            if "/blob/" in repo_path:
                repo_path = repo_path.split("/blob/")[0]
            return repo_path
        else:
            # Assume the URL is already a repo_id
            return self.url.model

    def _get_fallback_model_info(self) -> dict:
        """
        Provide fallback model information when API is unavailable.

        Returns:
            dict: Conservative estimates for model parameters.
        """
        # Conservative estimates based on common model sizes
        return {
            "config": {
                "model_type": "unknown",
                "hidden_size": 768,
                "num_attention_heads": 12,
                "num_hidden_layers": 12,
                "vocab_size": 50000,
            },
            "param_source": "fallback",
            "estimated_parameters": 110000000,  # 110M parameters estimate
            "tensor_type": "float32",
        }

    def _calculate_storage_size(self) -> float:
        """
        Calculate storage size requirements in MB.

        Returns:
            float: Storage size in megabytes.
        """
        # Try to get file sizes from model info
        if "siblings" in self.model_info:
            total_size = 0
            for file_info in self.model_info["siblings"]:
                if "size" in file_info:
                    total_size += file_info["size"]
            if total_size > 0:
                return total_size / (1024 * 1024)  # Convert to MB

        # Fallback: estimate from parameter count
        param_count = self._get_parameter_count()
        tensor_type = self._get_tensor_type()
        bytes_per_param = TENSOR_TYPE_SIZES.get(
            tensor_type, TENSOR_TYPE_SIZES["default"]
        )

        # Add overhead for model files (config, tokenizer, etc.)
        model_weights_bytes = param_count * bytes_per_param
        overhead_factor = 1.2
        total_storage_bytes = model_weights_bytes * overhead_factor

        return total_storage_bytes / (1024 * 1024)  # Convert to MB

    def _calculate_memory_size(self) -> float:
        """
        Calculate memory requirements in MB for loading the model.

        Returns:
            float: Memory size in megabytes.
        """
        param_count = self._get_parameter_count()
        tensor_type = self._get_tensor_type()
        bytes_per_param = TENSOR_TYPE_SIZES.get(
            tensor_type, TENSOR_TYPE_SIZES["default"]
        )

        # 1. Model weights memory (always required)
        weights_memory_bytes = param_count * bytes_per_param

        # 2. Activation memory (depends on architecture and sequence length)
        config = self.model_info.get("config", {})
        seq_length = config.get("max_position_embeddings", 512)
        hidden_size = config.get("hidden_size", 768)
        num_layers = config.get("num_hidden_layers", 12)

        # Activation memory estimation for transformers:
        # - Self-attention: (seq_len^2 * num_heads * head_dim) per layer
        # - FFN activations: (seq_len * intermediate_size) per layer
        # Simplified: seq_length × hidden_size × num_layers × bytes_per_activation
        bytes_per_activation = 4  # float32 activations typically
        activation_memory_bytes = (
            seq_length * hidden_size * num_layers * bytes_per_activation
        )

        # 3. Framework overhead (PyTorch/TensorFlow runtime, buffers, etc.)
        # Typically 20-30% of model weights
        framework_overhead_bytes = weights_memory_bytes * 0.25

        # 4. KV cache for autoregressive models (if applicable)
        # For generation, we need to cache keys and values
        model_type = config.get("model_type", "").lower()
        kv_cache_bytes = 0
        if any(arch in model_type for arch in ["gpt", "llama", "mistral", "phi"]):
            # KV cache: 2 (K+V) × num_layers × seq_length × hidden_size × bytes_per_param
            kv_cache_bytes = 2 * num_layers * seq_length * hidden_size * bytes_per_param

        # Total memory requirement
        total_memory_bytes = (
            weights_memory_bytes
            + activation_memory_bytes
            + framework_overhead_bytes
            + kv_cache_bytes
        )
        return total_memory_bytes / (1024 * 1024)  # Convert to MB

    def _get_parameter_count(self) -> int:
        """
        Extract or estimate the number of parameters in the model.

        Returns:
            int: Number of parameters.
        """
        # Try to get from model config
        if "config" in self.model_info:
            config = self.model_info["config"]

            # Some models have explicit parameter count but no standardized naming
            if "n_params" in config:
                return config["n_params"]
            if "num_parameters" in config:
                return config["num_parameters"]
            if "parameter_count" in config:
                return config["parameter_count"]
            if "total_params" in config:
                return config["total_params"]
            if "nparams" in config:
                return config["nparams"]

            # Estimate from architecture if no match found
            return self._estimate_parameters_from_config(config)

        # Use fallback estimate
        return self.model_info.get("estimated_parameters", 110000000)

    def _estimate_parameters_from_config(self, cfg: dict) -> int:
        """
        Estimate parameter count from model configuration.
        Estimate based on common model architchtures (Transformer-based).

        Args:
            config (dict): Model configuration.

        Returns:
            int: Estimated parameter count.
        """
        # Hidden size
        H = _get_cfg(cfg, "hidden_size", "dim", "n_embd", default=768)
        # Layers
        L = _get_cfg(cfg, "num_hidden_layers", "n_layers", "num_layers", default=12)
        # Intermediate (FFN) size
        I = _get_cfg(cfg, "intermediate_size", "hidden_dim", default=(4 * H))
        # Vocab / positions / token types
        V = _get_cfg(cfg, "vocab_size", "n_words", default=30000)
        P = _get_cfg(cfg, "max_position_embeddings", "n_positions", default=512)
        T = _get_cfg(cfg, "type_vocab_size", default=2)

        if None in (H, L, V):
            # Fallback if config incomplete
            raise ValueError("Insufficient config to estimate parameter count")

        # 1. Embedding parameters
        token_embeddings = V * H
        position_embeddings = P * H if P else 0
        token_type_embeddings = T * H if T else 0
        embedding_params = (
            token_embeddings + position_embeddings + token_type_embeddings
        )

        # 2. Per-layer parameters
        # Attention: Q, K, V, O projections (4 × H × H)
        attention_params = 4 * H * H

        # Feed-forward: up-projection (H → I) + down-projection (I → H)
        ffn_params = H * I + I * H

        # Layer normalization and biases
        # - 2 LayerNorm layers per transformer block (scale + bias = 2*H each)
        # - Attention biases (4 × H for Q,K,V,O)
        # - FFN biases (I + H)
        layer_norm_params = 2 * 2 * H  # 2 LayerNorm × 2 params each
        bias_params = (4 * H) + (I + H)  # Attention + FFN biases
        per_layer_overhead = layer_norm_params + bias_params

        per_layer_params = attention_params + ffn_params + per_layer_overhead
        total_layer_params = L * per_layer_params

        output_params = 0
        if cfg.get("add_pooling_layer", False) or cfg.get("pooler_fc_size"):
            output_params += H * H + H  # Pooler dense layer

        if not cfg.get("tie_word_embeddings", True):
            output_params += V * H  # Separate LM head

        total_params = embedding_params + total_layer_params + output_params

        return int(total_params)

    def _get_tensor_type(self) -> str:
        """
        Determine the tensor type used by the model.

        Returns:
            str: Tensor type (e.g., 'float32', 'float16').
        """
        # Try to get from model config
        if "config" in self.model_info:
            config = self.model_info["config"]

            # Check common tensor type fields
            tensor_type_fields = ["torch_dtype", "dtype", "model_dtype"]
            for field in tensor_type_fields:
                if field in config:
                    dtype = str(config[field])
                    # Clean up torch dtype format
                    if "torch." in dtype:
                        dtype = dtype.replace("torch.", "")
                    return dtype

        # Check if model info has tensor type
        if "tensor_type" in self.model_info:
            return self.model_info["tensor_type"]

        # Default to float32
        return "float32"

    def get_size_details(self, device_name) -> dict:
        """
        Get detailed size information for debugging and analysis.
        Args:
            device_name: name of the device to get details of
        Returns:
            dict: Detailed size breakdown.
        """
        device_specs = DEVICE_SPECS.get(device_name, None)
        if device_specs is None:
            raise ValueError("Unknown Device Type")
        return {
            "storage_size_mb": self.storage_size_mb,
            "memory_size_mb": self.memory_size_mb,
            "parameter_count": self._get_parameter_count(),
            "tensor_type": self._get_tensor_type(),
            "target_platform": self.target_platform,
            "device_storage_mb": device_specs["storage_mb"],
            "device_memory_mb": device_specs["memory_mb"],
            "available_memory_mb": device_specs["memory_mb"]
            * device_specs["available_memory_ratio"],
            "storage_fits": self.storage_size_mb <= device_specs["storage_mb"],
            "memory_fits": self.memory_size_mb
            <= (device_specs["memory_mb"] * device_specs["available_memory_ratio"]),
        }
