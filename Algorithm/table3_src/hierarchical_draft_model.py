import torch
import time
from typing import List, Optional, Union, Dict, Any
from vllm import LLM, SamplingParams


class HierarchicalDraftModel:
    """
    Hierarchical speculative decoding draft model.
    
    Uses a smaller token-level drafter (0.5B) to accelerate a larger step-level drafter (1.5B)
    via vLLM's built-in speculative decoding. The accelerated 1.5B model outputs are then
    used to guide the 7B generator with PRM scoring.
    
    Architecture:
    - Token-level drafter: 0.5B model (e.g., Qwen2.5-0.5B-Instruct)
    - Step-level drafter: 1.5B model (e.g., Qwen2.5-1.5B-Instruct) 
    - The 0.5B drafts tokens for the 1.5B model using speculative decoding
    """
    
    def __init__(
        self,
        step_drafter_path: str,
        token_drafter_path: str,
        tokenizer=None,
        device: str = "cuda",
        num_speculative_tokens: int = 5,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
    ):
        """
        Initialize the hierarchical draft model.
        
        Args:
            step_drafter_path: Path to the 1.5B step-level drafter model
            token_drafter_path: Path to the 0.5B token-level drafter model
            tokenizer: Optional tokenizer (will load from step_drafter_path if not provided)
            device: Device to run the model on
            num_speculative_tokens: Number of tokens the 0.5B model drafts at a time
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: Fraction of GPU memory to use
        """
        self.device = device
        self.tokenizer = tokenizer
        self.model_name = step_drafter_path.split("/")[-1]
        self.step_drafter_path = step_drafter_path
        self.token_drafter_path = token_drafter_path
        self.num_speculative_tokens = num_speculative_tokens
        
        print(f"[HierarchicalDraftModel] Initializing hierarchical speculative decoding:")
        print(f"  - Step-level drafter (target): {step_drafter_path}")
        print(f"  - Token-level drafter (speculative): {token_drafter_path}")
        print(f"  - Num speculative tokens: {num_speculative_tokens}")
        
        # Initialize vLLM with speculative decoding configuration
        # The 1.5B model is the "target" being accelerated by the 0.5B "draft" model
        self.model = LLM(
            model=step_drafter_path,
            trust_remote_code=True,
            dtype="float16",
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            speculative_config={
                "model": token_drafter_path,
                "num_speculative_tokens": num_speculative_tokens,
            },
        )
        
        # Set padding token if not set
        if self.tokenizer is not None and self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        print(f"[HierarchicalDraftModel] Initialization complete")
    
    def generate(self, prompt, temperature=0.0, top_p=1.0, max_tokens=100, stop=None, n=1):
        """
        Generate text from the prompt using hierarchical speculative decoding.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_tokens: Maximum number of tokens to generate
            stop: Token or list of tokens to stop generation (e.g., "\n\n")
            n: Number of completions to generate per prompt
            
        Returns:
            generated_texts: List of generated texts (length n)
        """
        # Handle stop tokens
        stop_tokens = stop if isinstance(stop, list) else [stop] if stop else None
        
        # Create sampling parameters for vLLM
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop_tokens,
            n=n
        )
        
        # If prompt is already tokenized, convert back to string
        if not isinstance(prompt, str):
            prompt = self.tokenizer.decode(prompt[0], skip_special_tokens=True)
        
        # Generate with vLLM (speculative decoding happens internally)
        outputs = self.model.generate(prompt, sampling_params)
        # Extract all generated texts for this prompt
        generated_texts = [seq.text for seq in outputs[0].outputs]
        return generated_texts
    
    def generate_batch(self, batch_prompts, temperature=0.0, top_p=1.0, max_tokens=100, stop=None, max_batch_size=32, n=1):
        """
        Generate text for multiple prompts in a batch using hierarchical speculative decoding.
        Process large batches in smaller chunks to avoid memory issues.
        
        Args:
            batch_prompts: List of input prompts
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_tokens: Maximum number of tokens to generate
            stop: Stop token or list of stop tokens
            max_batch_size: Maximum number of prompts to process in a single batch
            n: Number of completions to generate per prompt
            
        Returns:
            List of generated texts (if n==1: [str, ...], if n>1: [[str, ...], ...])
        """
        # Handle stop tokens
        stop_tokens = stop if isinstance(stop, list) else [stop] if stop else None
        
        # Create sampling parameters for vLLM
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop_tokens,
            n=n
        )
        
        # Convert any tokenized prompts to strings
        processed_prompts = []
        for prompt in batch_prompts:
            if not isinstance(prompt, str):
                prompt = self.tokenizer.decode(prompt[0], skip_special_tokens=True)
            processed_prompts.append(prompt)
        
        # Process in smaller batches to avoid memory issues
        all_responses = []
        total_prompts = len(processed_prompts)
        
        for i in range(0, total_prompts, max_batch_size):
            batch = processed_prompts[i:i+max_batch_size]
            
            # Generate with vLLM in smaller batch (speculative decoding happens internally)
            outputs = self.model.generate(batch, sampling_params)
            
            # Extract generated texts for each prompt in batch
            for output in outputs:
                completions = [seq.text for seq in output.outputs]
                all_responses.append(completions)
        
        return all_responses
    
    def completions_create(self, 
                         model: Optional[str] = None,
                         prompt: Union[str, List[str]] = "",
                         temperature: float = 0.0,
                         top_p: float = 1.0,
                         max_tokens: int = 100,
                         stop: Optional[Union[str, List[str]]] = None,
                         max_batch_size: int = 32,
                         n: int = 1,
                         **kwargs: Any):
        """
        Match the OpenAI API format used in main_online.py
        
        Args:
            model: Model name (ignored, using the initialized model)
            prompt: List of prompts or single prompt string
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_tokens: Maximum number of tokens to generate
            stop: List of stop sequences or single stop sequence
            max_batch_size: Maximum number of prompts to process in a single batch
            n: Number of completions to generate per prompt
            **kwargs: Additional parameters (ignored)
            
        Returns:
            Object with choices attribute containing the generated completions
        """
        # Unify: always treat prompt as a list of length 1 (for n>1) or list of prompts
        if isinstance(prompt, str):
            prompts = [prompt]
        else:
            prompts = prompt

        # Call vLLM directly to get full output objects with finish/stop reasons.
        stop_tokens = stop if isinstance(stop, list) else [stop] if stop else None
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop_tokens,
            n=n
        )
        outputs = self.model.generate(prompts, sampling_params)

        from types import SimpleNamespace
        choices = []
        for i, output in enumerate(outputs):
            for j, completion in enumerate(output.outputs):
                choice = SimpleNamespace(
                    text=completion.text,
                    index=i * n + j,
                    stop_reason=completion.stop_reason
                )
                choices.append(choice)

        response = SimpleNamespace(
            choices=choices,
        )
        return response

    # Add completions.create for OpenAI-style interface
    @property
    def completions(self):
        class _Completions:
            def __init__(self, outer):
                self._outer = outer
            def create(self, **kwargs):
                return self._outer.completions_create(**kwargs)
        return _Completions(self)
