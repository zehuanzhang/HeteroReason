import os
# Ensure the PRM worker uses the vLLM V0 engine.
# This must happen before importing vllm.
os.environ.setdefault("VLLM_USE_V1", "0")

import torch
import numpy as np
from typing import List, Optional, Union, Dict, Any
from vllm import LLM

class LocalPRMModel:
    """
    A local PRM model wrapper that matches the OpenAI embeddings API format, using vLLM for inference.
    """
    def __init__(
        self,
        model_name_or_path,
        tokenizer=None,
        device="cuda",
    ):
        """
        Initialize the local PRM model using vLLM.
        Args:
            model_name_or_path: Path to the model or model name
            tokenizer: Optional tokenizer (will load from model_name_or_path if not provided)
            device: Device to run the model on
        """
        self.device = device
        self.tokenizer = tokenizer
        self.model_name = model_name_or_path.split("/")[-1]

        self.model = LLM(
            model=model_name_or_path,
            trust_remote_code=True,
            dtype="float16",
            tensor_parallel_size=1,
            #task="reward"
            #device=torch.device("cuda:0")
        )
        if self.tokenizer and self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def embeddings_create(self, 
                         input: Union[str, List[str], List[List[int]]],
                         model: Optional[str] = None,
                         **kwargs: Any):
        """
        Use vLLM's encode method to get PRM rewards/embeddings, matching the OpenAI embeddings API.
        Args:
            input: Input data (token IDs or strings)
            model: Model name (ignored)
            **kwargs: Additional parameters (ignored)
        Returns:
            Object with data attribute containing embeddings/rewards
        """


        # Accepts either token IDs or text prompts, robust to tuple input
        inp = input
        if isinstance(inp, tuple):
            # If tuple of a list, convert to list of lists
            if len(inp) == 1 and isinstance(inp[0], list) and all(isinstance(i, int) for i in inp[0]):
                inp = [inp[0]]
            else:
                inp = list(inp)

        if isinstance(inp, str):
            prompts = [inp]
            prompt_token_ids = None
        elif isinstance(inp, list) and len(inp) > 0 and isinstance(inp[0], str):
            prompts = inp
            prompt_token_ids = None
        elif isinstance(inp, list) and len(inp) > 0 and isinstance(inp[0], list):
            # Ensure all elements are lists of ints
            if not all(isinstance(x, list) and all(isinstance(i, int) for i in x) for x in inp):
                raise TypeError("If passing token IDs, input must be List[List[int]]")
            prompts = None
            prompt_token_ids = inp
        else:
            raise TypeError("Input to embeddings_create must be str, List[str], or List[List[int]]")

        # Call vLLM's encode method with chunking to avoid CUDA OOM
        chunk_size = 4 if "Qwen2.5-Math-PRM-7B" in self.model_name else 16
        outputs = []
        
        if prompt_token_ids is not None:
            # Process token IDs in chunks
            for i in range(0, len(prompt_token_ids), chunk_size):
                chunk = prompt_token_ids[i:i + chunk_size]
                chunk_outputs = self.model.encode(prompt_token_ids=chunk)
                outputs.extend(chunk_outputs)
        else:
            # Process prompts in chunks
            for i in range(0, len(prompts), chunk_size):
                chunk = prompts[i:i + chunk_size]
                chunk_outputs = self.model.encode(chunk)
                outputs.extend(chunk_outputs)

        if not outputs or len(outputs) == 0:
            raise ValueError("vLLM encode returned no outputs. Check input and model configuration.")
        from types import SimpleNamespace
        embeddings = []
        
        # Handle different output formats based on model
        if "Qwen2.5-Math-PRM-7B" in self.model_name:
            # Handle PoolingRequestOutput format for Qwen2.5-Math-PRM-7B
            for i, output in enumerate(outputs):
                if hasattr(output, 'outputs') and hasattr(output.outputs, 'data'):
                    # Extract tensor data and convert to list
                    tensor_data = output.outputs.data
                    if hasattr(tensor_data, 'tolist'):
                        emb = tensor_data.tolist()
                    elif hasattr(tensor_data, 'cpu'):
                        emb = tensor_data.cpu().tolist()
                    else:
                        emb = tensor_data
                    
                    emb = [score[1] for score in emb]
                else:
                    raise ValueError(f"Expected output.outputs.data for Qwen2.5-Math-PRM-7B, but got: {output}")
                
                if not emb or (isinstance(emb, (list, np.ndarray)) and len(emb) == 0):
                    raise ValueError(f"vLLM encode returned an empty embedding for input index {i}. Input: {input}, Output: {output}")
                embedding_obj = SimpleNamespace(
                    embedding=emb,
                    index=i,
                    object='embedding'
                )
                embeddings.append(embedding_obj)
        else:
            # Original format for other models
            for i, output in enumerate(outputs):
                emb = None

                if hasattr(output, "outputs"):
                    out_obj = output.outputs
                    # Some vLLM outputs store a single object, some store a list.
                    if isinstance(out_obj, list) and len(out_obj) == 1:
                        out_obj = out_obj[0]

                    # Pooling/reward-style outputs commonly expose `.data`.
                    if hasattr(out_obj, "data"):
                        tensor_data = out_obj.data
                        if hasattr(tensor_data, "detach"):
                            tensor_data = tensor_data.detach()
                        if hasattr(tensor_data, "cpu"):
                            tensor_data = tensor_data.cpu()
                        emb = tensor_data.tolist() if hasattr(tensor_data, "tolist") else tensor_data

                    # Embedding-style outputs expose `.embedding`.
                    elif hasattr(out_obj, "embedding"):
                        emb = out_obj.embedding

                if emb is None:
                    raise ValueError(
                        "vLLM encode output did not contain `.outputs.data` or `.outputs.embedding`. "
                        "This usually means the PRM output format is not being parsed correctly. "
                        f"Got output={type(output)} with outputs={getattr(output, 'outputs', None)}"
                    )

                # Flatten embedding if it's a list of lists (e.g., [[x], [y], ...])
                if isinstance(emb, list) and len(emb) > 0 and isinstance(emb[0], list) and len(emb[0]) == 1:
                    emb = [x[0] for x in emb]
                if not emb or (isinstance(emb, (list, np.ndarray)) and len(emb) == 0):
                    raise ValueError(f"vLLM encode returned an empty embedding for input index {i}. Input: {input}, Output: {output}")
                embedding_obj = SimpleNamespace(
                    embedding=emb,
                    index=i,
                    object='embedding'
                )
                embeddings.append(embedding_obj)

        # Create response object as a SimpleNamespace (for .data attribute and pickling)
        response = {
            'data': embeddings,
            # 'model': self.model_name,
            # 'object': 'list',
            # 'usage': {
            #     'prompt_tokens': 0,
            #     'total_tokens': 0
            # }
        }
        return SimpleNamespace(**response)

class LocalPRMEmbeddings:
    """Wrapper class to match the client.embeddings.create() API"""
    def __init__(self, prm_model):
        self.prm_model = prm_model
    
    def create(self, **kwargs):
        return self.prm_model.embeddings_create(**kwargs)