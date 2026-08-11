import torch
import time
from typing import List, Optional, Union, Dict, Any
from vllm import LLM, SamplingParams

class LocalDraftModel:
    """
    A simple wrapper for a local draft model that doesn't use the tree structure.
    This implementation matches the behavior of vLLM's completions.create API.
    """
    def __init__(
        self,
        model_name_or_path,
        tokenizer=None,
        device="cuda",
    ):
        """
        Initialize the local draft model.
        
        Args:
            model_name_or_path: Path to the model or model name
            tokenizer: Optional tokenizer (will load from model_name_or_path if not provided)
            device: Device to run the model on
        """
        self.device = device
        self.tokenizer = tokenizer
        self.model_name = model_name_or_path.split("/")[-1]
            
        # Initialize vLLM model instead of HF model
        self.model = LLM(
            model=model_name_or_path,
            trust_remote_code=True,
            dtype="float16",
            tensor_parallel_size=1,  # Use single GPU
            #enforce_eager=True
            #device=torch.device(device)
        )
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate(self, prompt, temperature=0.0, top_p=1.0, max_tokens=100, stop=None, n=1):
        """
        Generate text from the prompt using vLLM.
        
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
        
        # Generate with vLLM
        outputs = self.model.generate(prompt, sampling_params)
        # Extract all generated texts for this prompt
        generated_texts = [seq.text for seq in outputs[0].outputs]
        return generated_texts
    
    def generate_batch(self, batch_prompts, temperature=0.0, top_p=1.0, max_tokens=100, stop=None, max_batch_size=32, n=1):
        """
        Generate text for multiple prompts in a batch using vLLM.
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
            
            # Generate with vLLM in smaller batch
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
            # model=self.model_name,
            # usage={
            #     'prompt_tokens': sum(len(self.tokenizer.encode(p)) for p in prompts),
            #     'completion_tokens': sum(len(self.tokenizer.encode(c.text)) for c in choices),
            #     'total_tokens': sum(len(self.tokenizer.encode(p)) for p in prompts) + sum(len(self.tokenizer.encode(c.text)) for c in choices)
            # }
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
