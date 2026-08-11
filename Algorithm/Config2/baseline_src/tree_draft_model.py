import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import copy
import math
import time
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple, Dict, Any

# EAGLE's predefined tree structures
tree_structure = [[0], [1], [2], [3], [0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [2, 0], [2, 1], [3, 0],
                 [0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0], [0, 1, 1], [0, 2, 0], [0, 2, 1], [1, 0, 0],
                 [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 0, 0, 0], [0, 0, 0, 0, 1]]

chain_structure = [[0], [0, 0], [0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0, 0]]

class BoundedWidthTree:
    """
    A class that generates tree paths with a bounded width at each level.
    """
    def __init__(self, max_width=4, max_cached_depth=10):
        """
        Initialize the tree with bounded width.
        
        Args:
            max_width: Maximum number of children per node
            max_cached_depth: Maximum depth to cache paths for
        """
        self.max_width = max_width
        self.max_cached_depth = max_cached_depth
        self.cached_paths = {}
        
        # Initialize cache with first level paths
        self._initialize_cache()
    
    def _initialize_cache(self):
        """Initialize the cache with paths up to max_cached_depth"""
        # First level (single tokens)
        level_1 = [[i] for i in range(self.max_width)]
        self.cached_paths[1] = level_1
        
        # Generate paths for deeper levels up to max_cached_depth
        for depth in range(2, self.max_cached_depth + 1):
            current_level_paths = []
            for parent_path in self.cached_paths[depth-1]:
                for i in range(self.max_width):
                    new_path = parent_path + [i]
                    current_level_paths.append(new_path)
            self.cached_paths[depth] = current_level_paths
    
    def get_paths_at_depth(self, depth):
        """Get all paths at the specified depth"""
        if depth in self.cached_paths:
            return self.cached_paths[depth]
        
        # For depths beyond cache, generate on-the-fly
        if depth <= 1:
            return [[i] for i in range(self.max_width)]
        
        # Get parent paths and generate children
        parent_paths = self.get_paths_at_depth(depth-1)
        current_level_paths = []
        
        for parent_path in parent_paths:
            for i in range(self.max_width):
                new_path = parent_path + [i]
                current_level_paths.append(new_path)
        
        # Cache if within limit
        if depth <= self.max_cached_depth:
            self.cached_paths[depth] = current_level_paths
            
        return current_level_paths
    
    def get_all_paths_up_to_depth(self, max_depth):
        """Get all paths up to the specified maximum depth"""
        all_paths = []
        for depth in range(1, max_depth + 1):
            all_paths.extend(self.get_paths_at_depth(depth))
        return all_paths

class FixedNodesPerLevelTree:
    """
    A class that generates tree paths with a fixed number of nodes at each level.
    """
    def __init__(self, nodes_per_level=4, max_cached_depth=10):
        """
        Initialize the tree with fixed nodes per level.
        
        Args:
            nodes_per_level: Maximum number of nodes at each level
            max_cached_depth: Maximum depth to cache paths for
        """
        self.nodes_per_level = nodes_per_level
        self.max_cached_depth = max_cached_depth
        self.cached_paths = {}
        
        # Initialize cache with first level paths
        self._initialize_cache()
    
    def _initialize_cache(self):
        """Initialize the cache with paths up to max_cached_depth"""
        # First level (single tokens) - exactly nodes_per_level nodes
        level_1 = [[i] for i in range(self.nodes_per_level)]
        self.cached_paths[1] = level_1
        
        # Generate paths for deeper levels up to max_cached_depth
        for depth in range(2, self.max_cached_depth + 1):
            # For each level, we select exactly nodes_per_level paths
            # We do this by taking the first node from each parent and then
            # the second node from each parent, etc., until we have nodes_per_level paths
            parent_paths = self.cached_paths[depth-1]
            current_level_paths = []
            
            # Calculate how many children per parent to maintain nodes_per_level
            # If we have fewer parents than nodes_per_level, each parent gets multiple children
            # If we have more parents than nodes_per_level, some parents get no children
            children_per_parent = max(1, self.nodes_per_level // len(parent_paths))
            
            # Generate children for each parent
            for i, parent_path in enumerate(parent_paths):
                # Skip parents beyond what we need
                if i * children_per_parent >= self.nodes_per_level:
                    break
                    
                # Generate children for this parent
                for j in range(children_per_parent):
                    # Skip if we've reached the maximum nodes for this level
                    if len(current_level_paths) >= self.nodes_per_level:
                        break
                    
                    # Create new path by extending parent path
                    new_path = parent_path + [j]
                    current_level_paths.append(new_path)
            
            # Ensure we have exactly nodes_per_level paths
            if len(current_level_paths) > self.nodes_per_level:
                current_level_paths = current_level_paths[:self.nodes_per_level]
            
            self.cached_paths[depth] = current_level_paths
    
    def get_paths_at_depth(self, depth):
        """Get all paths at the specified depth"""
        if depth in self.cached_paths:
            return self.cached_paths[depth]
        
        # For depths beyond cache, generate on-the-fly
        if depth <= 1:
            return [[i] for i in range(self.nodes_per_level)]
        
        # Get parent paths and generate exactly nodes_per_level children
        parent_paths = self.get_paths_at_depth(depth-1)
        current_level_paths = []
        
        # Calculate children per parent as before
        children_per_parent = max(1, self.nodes_per_level // len(parent_paths))
        
        # Generate children for each parent
        for i, parent_path in enumerate(parent_paths):
            if i * children_per_parent >= self.nodes_per_level:
                break
                
            for j in range(children_per_parent):
                if len(current_level_paths) >= self.nodes_per_level:
                    break
                
                new_path = parent_path + [j]
                current_level_paths.append(new_path)
        
        # Ensure we have exactly nodes_per_level paths
        if len(current_level_paths) > self.nodes_per_level:
            current_level_paths = current_level_paths[:self.nodes_per_level]
        
        # Cache if within limit
        if depth <= self.max_cached_depth:
            self.cached_paths[depth] = current_level_paths
            
        return current_level_paths
    
    def get_all_paths_up_to_depth(self, max_depth):
        """Get all paths up to the specified maximum depth"""
        all_paths = []
        for depth in range(1, max_depth + 1):
            all_paths.extend(self.get_paths_at_depth(depth))
        return all_paths

def generate_bounded_tree_structure(max_depth=5, max_width=4, fixed_nodes_per_level=False):
    """
    Generate a tree structure with bounded width at each level to avoid exponential explosion.
    For compatibility with existing code.
    
    Args:
        max_depth: Maximum depth of the tree
        max_width: Maximum width at each level or number of nodes per level if fixed_nodes_per_level=True
        fixed_nodes_per_level: If True, use exactly max_width nodes at each level
    
    Returns:
        tree_structure: List of paths in the tree
    """
    # Special case for width=1: create a chain structure of appropriate depth
    if max_width == 1:
        chain = []
        for i in range(max_depth):
            if i == 0:
                chain.append([0])
            else:
                chain.append([0] * (i + 1))
        return chain
    
    if fixed_nodes_per_level:
        # Create tree with fixed number of nodes per level
        tree = FixedNodesPerLevelTree(nodes_per_level=max_width)
    else:
        # Create bounded width tree (each node has max_width children)
        tree = BoundedWidthTree(max_width=max_width)
    
    # Get all paths up to max_depth
    return tree.get_all_paths_up_to_depth(max_depth)

class TreeDraftModel:
    """
    A draft model that generates a tree of candidates using EAGLE's approach.
    """
    def __init__(
        self,
        model_name_or_path,
        tokenizer=None,
        device="cuda",
        top_k=10,
        depth=3,
        total_tokens=25,
        threshold=1.0,
        use_chain=False,
        rope_scaling=None,
        parallel_tree_processing=False,
        max_threads=4,
        bounded_tree=False,
        width_per_level=None
    ):
        """
        Initialize the tree draft model.
        
        Args:
            model_name_or_path: Path to the model or model name
            tokenizer: Optional tokenizer (will load from model_name_or_path if not provided)
            device: Device to run the model on
            top_k: Number of top candidates to consider at each step
            depth: Depth of the tree
            total_tokens: Maximum number of tokens to generate
            threshold: Threshold for accepting candidates
            use_eagle_tree: Whether to use EAGLE's predefined tree structure
            use_chain: Whether to use chain structure instead of tree structure
            bounded_tree: Whether to use a bounded tree structure with controlled width
            width_per_level: List specifying the width at each level for bounded tree
        """
        self.device = device
        self.top_k = top_k
        self.depth = depth
        self.total_tokens = total_tokens
        self.threshold = math.log(threshold)
        
        # Set tree structure based on parameters
        # Always use EAGLE tree approach for deep trees
        if True:  # Previously checked use_eagle_tree
            if bounded_tree:
                # Get max width from width_per_level or use default
                max_width = 4  # Default max width
                if width_per_level is not None and len(width_per_level) > 0:
                    max_width = width_per_level[0]  # Use first element as max width
                
                # Adjust top_k to match max_width if needed
                self.top_k = min(self.top_k, max(1, max_width))
                
                # Use consistent approach for all depths
                self.dynamic_tree = False
                self.tree_structure = generate_bounded_tree_structure(
                    max_depth=min(depth, 10),  # Limit to reasonable depth for tree structure
                    max_width=max_width,
                    fixed_nodes_per_level=True  # Use fixed nodes per level
                )
                    
                # Log the tree structure for debugging
                print(f"Using bounded tree with max_width={max_width}, top_k={self.top_k}, depth={depth}")
                print(f"Tree structure has {len(self.tree_structure)} paths")
            else:
                self.dynamic_tree = False
                self.tree_structure = chain_structure if use_chain else tree_structure
            
        # RoPE scaling for position embeddings (important for tree generation)
        self.rope_scaling = rope_scaling
        
        # Parallel processing options
        self.parallel_tree_processing = parallel_tree_processing
        self.max_threads = max_threads
        
        # Speculative verification parameters
        self.verification_threshold = 0.9  # Threshold for accepting speculative tokens
        self.max_verification_tokens = 5   # Maximum tokens to verify at once
        
        # Load model and tokenizer
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        else:
            self.tokenizer = tokenizer
        
        # Configure model with RoPE scaling if provided
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16,
            "output_hidden_states": True
        }
        
        # Add RoPE scaling if provided
        if self.rope_scaling:
            model_kwargs["rope_scaling"] = self.rope_scaling
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, 
            **model_kwargs
        ).to(device)
        
        # Initialize cache for efficient generation
        self.past_key_values = None
        self.current_length = 0
        
        # Initialize tree mask for efficient tree generation
        self.tree_mask_init = None
        
        # Debug flag
        self.debug = True  # Enable debugging by default
    
    def reset_cache(self):
        """Reset the KV cache and tree mask"""
        self.past_key_values = None
        self.current_length = 0
        self.stable_kv = None
        
        # Initialize tree mask for the next generation
        self.tree_mask_init = torch.eye(self.top_k, device=self.device).unsqueeze(0).unsqueeze(0)  # [1, 1, top_k, top_k]
    
    def prepare_logits_processor(self, temperature=0.0, top_p=1.0, top_k=0):
        """Prepare logits processor for sampling"""
        from transformers.generation.logits_process import (
            LogitsProcessorList,
            TemperatureLogitsWarper,
            TopKLogitsWarper,
            TopPLogitsWarper,
        )
        
        processor_list = LogitsProcessorList()
        if temperature > 1e-5 and temperature != 1.0:
            processor_list.append(TemperatureLogitsWarper(temperature))
        if top_p < 1.0:
            processor_list.append(TopPLogitsWarper(top_p))
        if top_k > 0:
            processor_list.append(TopKLogitsWarper(top_k))
        
        return processor_list if processor_list else None
    
    def process_logits(self, logits, logits_processor):
        """Process logits with the given processor"""
        if logits_processor is not None:
            return logits_processor(None, logits)[0]
        return logits
    
    @torch.no_grad()
    def generate_tree_candidates(self, prompt, temperature=0.0, top_p=1.0, max_tokens=2048, stop_token=None):
        """
        Generate a tree of candidates using the EAGLE approach, optimized for deep trees.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_tokens: Maximum number of tokens to generate
            stop_token: Token to stop generation
            
        Returns:
            candidates: List of candidate sequences
            scores: Scores for each candidate
            tree_position_ids: Position IDs for tree verification
        """
        # Always use the EAGLE tree approach for deep trees
        return self.generate_eagle_tree_candidates(prompt, temperature, top_p, max_tokens, stop_token=stop_token, step_token=stop_token)
        
        # Process the tree to get draft tokens and retrieve indices
        scores_list = torch.cat(scores_list, dim=0).view(-1)
        token_list_flat = torch.cat(token_list, dim=0).view(-1)
        
        # Get top-k overall tokens
        top_scores = torch.topk(scores_list, self.total_tokens, dim=-1)
        top_scores_index = top_scores.indices
        top_scores_index = torch.sort(top_scores_index).values
        
        # Get draft tokens
        draft_tokens = token_list_flat[top_scores_index]
        draft_tokens = torch.cat((sample_token, draft_tokens), dim=0)
        
        # Calculate tree structure for verification
        draft_parents = torch.cat(parents_list, dim=0)[top_scores_index // self.top_k].long()
        mask_index = torch.searchsorted(top_scores_index, draft_parents - 1, right=False)
        mask_index[draft_parents == 0] = -1
        mask_index = mask_index + 1
        mask_index_list = mask_index.tolist()
        
        # Create tree mask
        tree_mask = torch.eye(self.total_tokens + 1, device=self.device).bool()
        tree_mask[:, 0] = True
        for i in range(self.total_tokens):
            tree_mask[i + 1].add_(tree_mask[mask_index_list[i]])
        
        # Calculate position IDs based on tree depth
        tree_position_ids = torch.sum(tree_mask, dim=1) - 1
        
        # Create retrieve indices for verification
        max_depth = torch.max(tree_position_ids) + 1
        noleaf_index = torch.unique(mask_index).tolist()
        noleaf_num = len(noleaf_index) - 1
        leaf_num = self.total_tokens - noleaf_num
        
        retrieve_indices = torch.zeros(leaf_num, max_depth.item(), dtype=torch.long, device=self.device) - 1
        rid = 0
        position_ids_list = tree_position_ids.tolist()
        
        for i in range(self.total_tokens + 1):
            if i not in noleaf_index:
                cid = i
                depth = position_ids_list[i]
                for j in reversed(range(depth + 1)):
                    retrieve_indices[rid, j] = cid
                    cid = mask_index_list[cid - 1] if cid > 0 else 0
                rid += 1
        
        # Sort retrieve indices if using logits processor
        if logits_processor is not None:
            maxitem = self.total_tokens + 5
            
            def custom_sort(idx):
                lst = retrieve_indices[idx].tolist()
                return [x if x >= 0 else maxitem for x in lst]
            
            sorted_indices = sorted(range(leaf_num), key=custom_sort)
            retrieve_indices = retrieve_indices[sorted_indices]
        
        # Create candidates from draft tokens and retrieve indices
        candidates = []
        for indices in retrieve_indices:
            valid_indices = indices[indices >= 0]
            candidate = draft_tokens[valid_indices]
            candidates.append(torch.cat([input_ids[0, :-1], candidate]))  # Exclude the last token of input_ids
        
        # Get scores for candidates
        candidate_scores = scores[:leaf_num]
        
        return candidates, candidate_scores, tree_position_ids
    
    def decode_candidates(self, candidates):
        """Decode candidates to text"""
        return [self.tokenizer.decode(c, skip_special_tokens=True) for c in candidates]
    
    @torch.no_grad()
    def generate_candidates_for_prm(self, prompt, temperature=0.0, top_p=1.0, step_token="\n\n"):
        """
        Generate candidates specifically formatted for PRM evaluation.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            step_token: Token to separate steps
            
        Returns:
            candidates: List of candidate sequences with their scores
            formatted_candidates: Candidates formatted for PRM evaluation
        """
        candidates, scores, _ = self.generate_tree_candidates(prompt, temperature, top_p)
        
        # Decode candidates
        decoded_candidates = self.decode_candidates(candidates)
        
        # Format candidates for PRM
        formatted_candidates = []
        for i, (candidate, score) in enumerate(zip(decoded_candidates, scores)):
            # Add step token if not already present
            if not candidate.endswith(step_token):
                candidate += step_token
            
            formatted_candidates.append({
                "text": candidate,
                "score": score.item(),
                "index": i
            })
        
        return candidates, formatted_candidates
        
    def generate(self, prompt, temperature=0.0, top_p=1.0, max_tokens=100, stop_token=None):
        """
        Generate text from the prompt using the tree-based approach.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_tokens: Maximum number of tokens to generate
            stop_token: Token to stop generation (e.g., "\n\n")
            
        Returns:
            generated_text: Generated text (without the stop token)
        """
        # Generate candidates
        candidates, scores, _ = self.generate_tree_candidates(prompt, temperature, top_p)
        
        # Select the best candidate
        best_idx = torch.argmax(scores).item()
        best_candidate = candidates[best_idx]
        
        # Decode the best candidate
        generated_text = self.tokenizer.decode(best_candidate[len(self.tokenizer.encode(prompt, return_tensors="pt")[0]):], skip_special_tokens=True)
        
        # Truncate at stop token if provided
        if stop_token and stop_token in generated_text:
            generated_text = generated_text[:generated_text.find(stop_token)]
        
        return generated_text
    
    def _handle_linear_generation(self, prompt, temperature=0.0, top_p=1.0, max_tokens=100, stop_token=None):
        """
        Special method to handle the case when max_width=1 and top_k=1, which should behave like
        standard autoregressive generation rather than tree-based generation.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_tokens: Maximum number of tokens to generate
            stop_token: Token to stop generation
            
        Returns:
            candidates: List of candidate sequences
            scores: Scores for each candidate
            tree_position_ids: Position IDs for verification
        """
        # Prepare logits processor
        logits_processor = self.prepare_logits_processor(temperature, top_p)
        
        # Tokenize input if it's a string
        if isinstance(prompt, str):
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        else:
            input_ids = prompt.to(self.device)
        
        # Reset cache
        self.reset_cache()
        
        # Initial forward pass
        with torch.no_grad():
            outputs = self.model(
                input_ids, 
                use_cache=True
            )
            self.past_key_values = outputs.past_key_values
        
        # Initialize generation
        generated_ids = input_ids.clone()
        cumulative_score = 0.0
        
        # Generate tokens autoregressively
        if self.debug:
            print(f"Linear generation: generating up to {min(self.total_tokens, max_tokens)} tokens")
            
        for i in range(min(self.total_tokens, max_tokens)):
            # Get the last token's logits
            logits = outputs.logits[:, -1, :]
            processed_logits = self.process_logits(logits, logits_processor)
            
            # Get the top token
            top_logits, top_indices = torch.topk(processed_logits, 1, dim=-1)
            next_token = top_indices[0, 0].unsqueeze(0).unsqueeze(0)
            
            if self.debug and i % 10 == 0:  # Print every 10 tokens
                token_text = self.tokenizer.decode([next_token[0, 0].item()])
                print(f"Token {i}: {token_text} (id: {next_token[0, 0].item()})")
                print(f"Logit: {top_logits[0, 0].item():.4f}")
            
            # Update cumulative score
            token_score = top_logits[0, 0].item()
            cumulative_score += token_score
            
            # Append to generated sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            # Check for stop token
            if stop_token is not None:
                # Check if the generated text contains the stop token
                generated_text = self.tokenizer.decode(generated_ids[0, input_ids.shape[1]:], skip_special_tokens=True)
                if stop_token in generated_text:
                    break
            
            # Forward pass with the new token
            outputs = self.model(
                input_ids=next_token,
                past_key_values=self.past_key_values,
                use_cache=True
            )
            self.past_key_values = outputs.past_key_values
        
        # Create a single candidate
        candidates = [generated_ids[0]]
        scores = torch.tensor([cumulative_score], device=self.device)
        
        # Create simple position IDs (just a sequence from 0 to length-1)
        tree_position_ids = torch.arange(len(generated_ids[0]), device=self.device)
        
        if self.debug:
            # Show the generated text
            generated_text = self.tokenizer.decode(generated_ids[0, input_ids.shape[1]:], skip_special_tokens=True)
            print(f"Linear generation produced: {generated_text[:100]}{'...' if len(generated_text) > 100 else ''}")
            print(f"Total tokens generated: {len(generated_ids[0]) - input_ids.shape[1]}")
            print(f"Cumulative score: {cumulative_score:.4f}")
        
        return candidates, scores, tree_position_ids
    
    def completions_create(self, prompt, temperature=0.0, top_p=1.0, max_tokens=100, stop=None, use_speculative=False):  # Default to False for backward compatibility
        """
        Match the OpenAI API format used in main_online.py
        
        Args:
            prompt: List of prompts or single prompt string
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_tokens: Maximum number of tokens to generate
            stop: List of stop sequences or single stop sequence
            
        Returns:
            Object with choices attribute containing the generated completions
        """
        # Handle both single prompt and list of prompts
        if isinstance(prompt, str):
            prompts = [prompt]
        else:
            prompts = prompt
            
        # Handle stop tokens
        stop_token = stop[0] if isinstance(stop, list) and len(stop) > 0 else stop
        
        # Process prompts in batches for efficiency
        batch_size = 1  # Can be increased for models that support batching
        choices = []
        
        for batch_start in range(0, len(prompts), batch_size):
            batch_end = min(batch_start + batch_size, len(prompts))
            batch_prompts = prompts[batch_start:batch_end]
            
            # Process each prompt in the batch
            batch_choices = []
            for i, p in enumerate(batch_prompts):
                # Reset cache for each prompt
                self.reset_cache()
                
                # Generate candidates
                if use_speculative and hasattr(self, 'speculative_sampling'):
                    # Use speculative sampling for efficient generation
                    input_ids = self.tokenizer.encode(p, return_tensors="pt")[0].to(self.device)
                    output_ids = self.speculative_sampling(
                        input_ids,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        stop_token=stop_token
                    )
                    # Create a single candidate with the generated tokens
                    candidates = [output_ids]
                    scores = torch.tensor([1.0], device=self.device)  # Dummy score
                elif max_tokens > self.total_tokens and hasattr(self, 'generate_long_candidates'):
                    # Use long candidate generation for longer sequences
                    candidates, scores = self.generate_long_candidates(
                        p, temperature, top_p, max_tokens, stop_token
                    )
                else:
                    # Use regular tree generation
                    candidates, scores, _ = self.generate_tree_candidates(
                        p, temperature, top_p, stop_token=stop_token
                    )
                
                # Select the best candidate
                if len(scores) > 0:
                    best_idx = torch.argmax(scores).item()
                    best_candidate = candidates[best_idx]
                else:
                    # Fallback if no candidates were generated
                    best_candidate = self.tokenizer.encode(p, return_tensors="pt")[0].to(self.device)
                
                # Decode the best candidate
                input_ids_len = len(self.tokenizer.encode(p, return_tensors="pt")[0])
                text = self.tokenizer.decode(best_candidate[input_ids_len:], skip_special_tokens=True)
                
                # Truncate at stop token if provided
                if stop_token and stop_token in text:
                    text = text[:text.find(stop_token)]
                
                # Create a response object similar to OpenAI API
                choice = type('obj', (object,), {
                    'text': text,
                    'index': batch_start + i,
                    'stop_reason': 'stop' if stop_token and stop_token in (text + stop_token) else None
                })
                batch_choices.append(choice)
            
            choices.extend(batch_choices)
            
        # Create a response object with choices attribute
        response = type('obj', (object,), {'choices': choices})
        return response
        
    def _prepare_decoder_attention_mask(self, attention_mask, tree_mask, input_shape, inputs_embeds, past_key_values_length):
        """
        Prepare decoder attention mask with tree structure, similar to EAGLE's approach.
        """
        try:
            # Create causal mask
            combined_attention_mask = None
            if input_shape[-1] > 1:
                combined_attention_mask = torch.ones(
                    (input_shape[0], input_shape[1], input_shape[1]), 
                    device=inputs_embeds.device
                )
                # Make it causal
                combined_attention_mask = torch.triu(combined_attention_mask, diagonal=1) * -1e10
                combined_attention_mask = combined_attention_mask.unsqueeze(1)  # [bs, 1, seq_len, seq_len]

            if attention_mask is not None:
                # Get the sequence length from attention_mask
                seq_len = attention_mask.size(1)
                
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                expanded_attn_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                expanded_attn_mask = (1.0 - expanded_attn_mask) * -1e10
                
                # Make sure dimensions match
                if combined_attention_mask is not None:
                    # Ensure expanded_attn_mask has the right shape
                    target_seq_len = combined_attention_mask.size(2)
                    expanded_attn_mask = expanded_attn_mask.expand(-1, -1, target_seq_len, seq_len)
                    
                    # Check if dimensions match before adding
                    if expanded_attn_mask.size(3) != combined_attention_mask.size(3):
                        # Pad the smaller one to match the larger one
                        if expanded_attn_mask.size(3) < combined_attention_mask.size(3):
                            pad_size = combined_attention_mask.size(3) - expanded_attn_mask.size(3)
                            expanded_attn_mask = torch.nn.functional.pad(
                                expanded_attn_mask, (0, pad_size, 0, 0, 0, 0, 0, 0), value=-1e10
                            )
                        else:
                            pad_size = expanded_attn_mask.size(3) - combined_attention_mask.size(3)
                            combined_attention_mask = torch.nn.functional.pad(
                                combined_attention_mask, (0, pad_size, 0, 0, 0, 0, 0, 0), value=-1e10
                            )
                else:
                    # If combined_attention_mask is None, just expand to input_shape
                    expanded_attn_mask = expanded_attn_mask.expand(-1, -1, input_shape[1], seq_len)
                    
                combined_attention_mask = (
                    expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
                )

            # Apply tree mask if provided
            if tree_mask is not None and combined_attention_mask is not None:
                tree_len = tree_mask.size(-1)
                bs = combined_attention_mask.size(0)
                
                # Make sure tree_len doesn't exceed the mask size
                mask_size = combined_attention_mask.size(3)
                if tree_len > mask_size:
                    # Truncate tree_mask to fit
                    tree_mask = tree_mask[:, :, :mask_size, :mask_size]
                    tree_len = mask_size
                
                # Apply tree mask to the last tree_len positions
                if tree_len <= combined_attention_mask.size(2) and tree_len <= combined_attention_mask.size(3):
                    # Get the region of the attention mask that corresponds to the tree structure
                    mask_region = combined_attention_mask[:, :, -tree_len:, -tree_len:]
                    
                    # Expand tree mask to match batch size
                    tree_mask_expanded = tree_mask.repeat(bs, 1, 1, 1)
                    
                    # Create a mask where tree_mask=1 means allow attention, tree_mask=0 means block attention
                    # We need to convert tree_mask from boolean to the same format as combined_attention_mask
                    # where -inf means no attention and 0 means full attention
                    attention_block = torch.zeros_like(mask_region)
                    attention_block[tree_mask_expanded == 0] = -1e10
                    
                    # Apply the tree structure to the attention mask
                    # This ensures tokens can only attend to their ancestors in the tree
                    mask_region.copy_(attention_block)
                
            return combined_attention_mask
            
        except Exception as e:
            # Fallback to a simple causal mask in case of errors
            print(f"Error in _prepare_decoder_attention_mask: {e}. Using fallback mask.")
            seq_len = input_shape[1] if input_shape else attention_mask.size(1) if attention_mask is not None else 1
            bs = input_shape[0] if input_shape else attention_mask.size(0) if attention_mask is not None else 1
            device = inputs_embeds.device
            
            # Create a simple causal mask
            mask = torch.ones((bs, 1, seq_len, seq_len), device=device)
            mask = torch.triu(mask, diagonal=1) * -1e10
            return mask
            
    def _get_rotary_dimensions(self):
        """
        Get rotary dimensions from model config.
        
        Returns:
            rotary_dim: Rotary dimension size
            head_dim: Head dimension size
        """
        # Get base model
        if hasattr(self.model, 'model'):
            base_model = self.model.model
        else:
            base_model = self.model
            
        # Try to get rotary dimension from config
        rotary_dim = 0
        head_dim = 0
        
        if hasattr(base_model.config, 'rotary_dim'):
            rotary_dim = base_model.config.rotary_dim
        elif hasattr(base_model.config, 'rope_dim'):
            rotary_dim = base_model.config.rope_dim
            
        # Try to get head dimension
        if hasattr(base_model.config, 'head_dim'):
            head_dim = base_model.config.head_dim
        elif hasattr(base_model.config, 'hidden_size') and hasattr(base_model.config, 'num_attention_heads'):
            # Calculate head_dim from hidden_size and num_attention_heads
            head_dim = base_model.config.hidden_size // base_model.config.num_attention_heads
            
        # If rotary_dim is still 0, use head_dim as default
        if rotary_dim == 0 and head_dim > 0:
            rotary_dim = head_dim
            
        return rotary_dim, head_dim
    
    def _ensure_position_ids_match_rotary_dims(self, position_ids, rotary_dim):
        """
        Ensure position IDs match the expected dimensions for rotary embeddings.
        This is crucial for avoiding dimension mismatches in apply_rotary_pos_emb.
        
        Args:
            position_ids: Position IDs tensor
            rotary_dim: Expected rotary dimension
            
        Returns:
            Adjusted position IDs tensor
        """
        if position_ids is None:
            return None
            
        # Check if we need to reshape or pad
        if len(position_ids.shape) < 2:
            # Add batch dimension if missing
            position_ids = position_ids.unsqueeze(0)
        
        # For Qwen2.5 models, position_ids should be a simple sequence
        # The rotary embeddings are handled internally by the model
        # So we'll just ensure the position_ids are properly shaped
        
        # Check if this is a Qwen model by looking at the model name
        if hasattr(self.model, 'config') and hasattr(self.model.config, 'model_type') and 'qwen' in self.model.config.model_type.lower():
            # For Qwen models, just return the position IDs as is
            # The model will handle the rotary embeddings internally
            return position_ids
        
        # For other models, ensure the position IDs have the right shape for rotary embeddings
        if position_ids.shape[-1] != rotary_dim and rotary_dim > 0:
            # Pad or truncate to match rotary dimension
            if position_ids.shape[-1] < rotary_dim:
                # Pad with zeros to match rotary dimension
                padding = torch.zeros(
                    *position_ids.shape[:-1], rotary_dim - position_ids.shape[-1],
                    device=position_ids.device, dtype=position_ids.dtype
                )
                position_ids = torch.cat([position_ids, padding], dim=-1)
            else:
                # Truncate to match rotary dimension
                position_ids = position_ids[..., :rotary_dim]
                
        return position_ids
    
    @torch.no_grad()
    def generate_eagle_tree_candidates(self, prompt, temperature=0.0, top_p=1.0, max_tokens=2048, stop_token=None, batch_size=1, step_token=None):
        """
        Generate a tree of candidates using EAGLE's predefined tree structure.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_tokens: Maximum number of tokens to generate
            stop_token: Optional token to stop generation
            
        Returns:
            candidates: List of candidate sequences
            scores: Scores for each candidate
            tree_position_ids: Position IDs for tree verification
        """
        # Special case for top_k=1 with chain structure
        # if self.top_k == 1 and hasattr(self, 'tree_structure'):
        #     # Check if we're using a chain-like structure
        #     is_chain = True
        #     for path in self.tree_structure:
        #         if len(path) > 0 and any(branch != 0 for branch in path):
        #             is_chain = False
        #             break
            
        #     if is_chain:
        #         print("Using linear generation in generate_eagle_tree_candidates")
        #         return self._handle_linear_generation(prompt, temperature, top_p, max_tokens, stop_token)
        # Prepare logits processor
        logits_processor = self.prepare_logits_processor(temperature, top_p)
        
        # Tokenize input if it's a string
        if isinstance(prompt, str):
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        else:
            input_ids = prompt.to(self.device)
        
        # Get EOS token ID for stopping condition
        eos_token_id = self.model.config.eos_token_id
        
        # Reset cache
        self.reset_cache()
        
        # Create attention mask
        attention_mask = torch.ones(input_ids.shape, device=self.device)
        
        # Initial forward pass
        with torch.no_grad():
            outputs = self.model(
                input_ids, 
                attention_mask=attention_mask,
                use_cache=True, 
                output_hidden_states=True
            )
            self.past_key_values = outputs.past_key_values
            self.current_length = input_ids.shape[1]
        
        # Get the last token's logits
        last_token_logits = outputs.logits[:, -1, :]
        processed_logits = self.process_logits(last_token_logits, logits_processor)
        
        # Get top-k tokens and their probabilities
        top_k_logits, top_k_indices = torch.topk(processed_logits, self.top_k, dim=-1)
        top_k_probs = F.softmax(top_k_logits, dim=-1)
        
        # Initialize tree tracking
        tree_nodes = []
        tree_scores = []
        tree_parents = []
        finish_flags = []
        
        # Add root node (last token from input)
        root_token = input_ids[0, -1].unsqueeze(0)
        tree_nodes.append(root_token)
        tree_scores.append(torch.zeros(1, device=self.device))
        tree_parents.append(-1)  # Root has no parent
        finish_flags.append(False)  # Root is not finished
        
        # For bounded tree, we don't need complex tree buffers
        # Just generate level by level with fixed width
        
        # Process each level of the tree according to EAGLE's structure
        bs = 1  # Batch size is 1 for now
        len_posi = input_ids.shape[1]
        position_ids = torch.arange(len_posi, device=self.device).unsqueeze(0)
        
        # Track hidden states for each node
        hidden_states = [outputs.hidden_states[-1][:, -1:]]  # Root hidden state
        
        # Generate tokens up to max_tokens using bounded tree approach
        tokens_generated = 0
        current_sequences = [input_ids[0].clone()]  # Start with input sequence
        
        while tokens_generated < max_tokens and tokens_generated < self.depth and len(current_sequences) > 0:
            new_sequences = []
            
            # For each current sequence, generate next tokens
            for seq_idx, current_seq in enumerate(current_sequences):
                # Forward pass to get next token predictions
                with torch.no_grad():
                    temp_input = current_seq.unsqueeze(0)
                    temp_outputs = self.model(
                        input_ids=temp_input,
                        use_cache=False,
                        output_hidden_states=True
                    )
                
                # Get logits for the last position
                last_logits = temp_outputs.logits[0, -1, :]
                processed_logits = self.process_logits(last_logits, logits_processor)
                
                # Get top-k tokens for this sequence (bounded width per sequence)
                width_per_seq = min(self.top_k, processed_logits.shape[-1])
                top_k_logits_seq, top_k_indices_seq = torch.topk(processed_logits, width_per_seq, dim=-1)
                
                # Add tokens to tree and create new sequences
                for j in range(width_per_seq):
                    token = top_k_indices_seq[j]
                    score = top_k_logits_seq[j]
                    
                    # Create new sequence by appending token
                    new_seq = torch.cat([current_seq, token.unsqueeze(0)])
                    
                    # Check for EOS or stop token - if found, don't expand this branch further
                    is_finished = False
                    if token.item() == eos_token_id:
                        is_finished = True
                    elif stop_token is not None:
                        # Check if the token or sequence contains stop token
                        token_text = self.tokenizer.decode([token.item()])
                        if stop_token in token_text:
                            is_finished = True
                        else:
                            # Check if the full sequence contains stop token
                            seq_text = self.tokenizer.decode(new_seq, skip_special_tokens=True)
                            if stop_token in seq_text:
                                is_finished = True
                    
                    # Always add the sequence (even if finished)
                    new_sequences.append(new_seq)
                    
                    # Add to tree tracking
                    tree_nodes.append(token.unsqueeze(0))
                    tree_scores.append(score.unsqueeze(0))
                    tree_parents.append(seq_idx if tokens_generated > 0 else 0)
                    finish_flags.append(is_finished)
            
            # Limit total sequences to maintain bounded width (max_width constraint)
            # Get max_width from the bounded tree configuration
            if hasattr(self, 'tree_structure') and hasattr(self, 'top_k'):
                max_width_limit = self.top_k * 5  # Allow some expansion but keep it bounded
            else:
                max_width_limit = 50  # Fallback
            max_width = min(max_width_limit, len(new_sequences))
            if len(new_sequences) > max_width:
                # Keep only the best sequences based on their scores
                seq_scores = []
                for seq in new_sequences:
                    # Simple scoring: use sequence length as a proxy
                    seq_scores.append(len(seq))
                
                # Sort and keep top sequences
                sorted_indices = sorted(range(len(new_sequences)), key=lambda i: seq_scores[i], reverse=True)
                new_sequences = [new_sequences[i] for i in sorted_indices[:max_width]]
            
            # Filter out finished sequences for next iteration
            next_sequences = []
            recent_flags = finish_flags[-len(new_sequences):] if finish_flags else []
            for i, seq in enumerate(new_sequences):
                # Only continue expanding sequences that aren't finished
                if i < len(recent_flags) and not recent_flags[i]:
                    next_sequences.append(seq)
                elif i >= len(recent_flags):  # No flag means not finished
                    next_sequences.append(seq)
            
            current_sequences = next_sequences
            tokens_generated += 1
            
            # Stop if no more sequences to expand or we have enough completed sequences
            if len(current_sequences) == 0 or (len(new_sequences) > 0 and tokens_generated >= self.depth):
                break
        
        # Create tree mask for attention
        tree_size = len(tree_nodes)
        tree_mask = torch.zeros(tree_size, tree_size, device=self.device).bool()
        
        # Fill tree mask based on parent relationships
        for i in range(tree_size):
            tree_mask[i, 0] = True  # All nodes can see root
            parent = tree_parents[i]
            while parent >= 0:
                tree_mask[i, parent] = True
                parent = tree_parents[parent] if parent > 0 else -1
        
        # Calculate position IDs
        tree_position_ids = torch.zeros(tree_size, dtype=torch.long, device=self.device)
        for i in range(tree_size):
            depth = 0
            parent = tree_parents[i]
            while parent >= 0:
                depth += 1
                parent = tree_parents[parent] if parent > 0 else -1
            tree_position_ids[i] = depth
        
        # Use all sequences we generated (including finished ones) as candidates
        all_sequences = []
        if 'new_sequences' in locals() and new_sequences:
            all_sequences = new_sequences
        else:
            all_sequences = [input_ids[0]]
        
        candidates = all_sequences
        
        # Create simple scores based on sequence length (longer = better)
        candidate_scores = []
        for seq in candidates:
            # Simple scoring: longer sequences get higher scores
            score = torch.tensor([len(seq) - len(input_ids[0])], device=self.device, dtype=torch.float)
            candidate_scores.append(score)
        
        # Convert scores to tensor
        if candidate_scores:
            candidate_scores = torch.cat(candidate_scores)
        else:
            # Fallback if no candidates were generated
            candidate_scores = torch.zeros(1, device=self.device)
            candidates = [input_ids[0]]  # Use input as fallback
        
        return candidates, candidate_scores, tree_position_ids
        
    def _generate_tree_buffers(self, tree_choices, device="cuda"):
        """
        Generate tree buffers for EAGLE-style tree generation.
        This is a simplified version of EAGLE's generate_tree_buffers_for_eagle function.
        """
        TOPK = self.top_k
        
        # Handle empty tree_choices
        if not tree_choices:
            # Return default simple tree buffer
            return {
                "attn_mask": [torch.eye(1, device=device).unsqueeze(0).unsqueeze(0)],
                "tree_indices": [torch.zeros(1, dtype=torch.long, device=device)],
                "position_ids": [torch.zeros(1, dtype=torch.long, device=device)],
                "repeat_nums": [[1]]
            }
        
        try:
            tree = self._build_tree(tree_choices)
            tree_len = tree.num_node_wchild()
            
            # Handle case where tree has no nodes with children
            if tree_len == 0:
                return {
                    "attn_mask": [torch.eye(1, device=device).unsqueeze(0).unsqueeze(0)],
                    "tree_indices": [torch.zeros(1, dtype=torch.long, device=device)],
                    "position_ids": [torch.zeros(1, dtype=torch.long, device=device)],
                    "repeat_nums": [[1]]
                }
            
            max_depth = tree.max_depth()
            nodes_wc = tree.get_node_wchild()
            
            # Handle case where max_depth is too small
            if max_depth <= 1:
                return {
                    "attn_mask": [torch.eye(1, device=device).unsqueeze(0).unsqueeze(0)],
                    "tree_indices": [torch.zeros(1, dtype=torch.long, device=device)],
                    "position_ids": [torch.zeros(1, dtype=torch.long, device=device)],
                    "repeat_nums": [[1]]
                }
            
            depth_counts = [0 for _ in range(max_depth - 1)]
            for x in nodes_wc:
                if x.depth - 1 < len(depth_counts):
                    depth_counts[x.depth - 1] += 1
            
            # Handle case where all depth counts are 0
            if all(count == 0 for count in depth_counts):
                return {
                    "attn_mask": [torch.eye(1, device=device).unsqueeze(0).unsqueeze(0)],
                    "tree_indices": [torch.zeros(1, dtype=torch.long, device=device)],
                    "position_ids": [torch.zeros(1, dtype=torch.long, device=device)],
                    "repeat_nums": [[1]]
                }
            
            depth_counts_sum = [sum(depth_counts[:i + 1]) for i in range(len(depth_counts))]
            
            tree_attn_mask = torch.eye(tree_len, tree_len, device=device)
            
            for id, x in enumerate(nodes_wc):
                try:
                    indices = x.all_index()
                    if indices and all(idx < tree_len for idx in indices):
                        tree_attn_mask[id, indices] = 1
                except (IndexError, RuntimeError):
                    continue
            
            tree_attn_mask_list0 = []
            for ml in depth_counts_sum:
                if ml > 0 and ml <= tree_attn_mask.shape[0]:
                    tree_attn_mask_list0.append(tree_attn_mask[:ml, :ml])
            
            # Handle case where tree_attn_mask_list0 is empty
            if not tree_attn_mask_list0:
                return {
                    "attn_mask": [torch.eye(1, device=device).unsqueeze(0).unsqueeze(0)],
                    "tree_indices": [torch.zeros(1, dtype=torch.long, device=device)],
                    "position_ids": [torch.zeros(1, dtype=torch.long, device=device)],
                    "repeat_nums": [[1]]
                }
            
            tree_attn_mask_list = []
            for id, x in enumerate(tree_attn_mask_list0):
                if id < len(depth_counts) and depth_counts[id] > 0 and depth_counts[id] <= x.shape[0]:
                    tree_attn_mask_list.append(x[-depth_counts[id]:])
            
            # Handle case where tree_attn_mask_list is empty
            if not tree_attn_mask_list:
                return {
                    "attn_mask": [torch.eye(1, device=device).unsqueeze(0).unsqueeze(0)],
                    "tree_indices": [torch.zeros(1, dtype=torch.long, device=device)],
                    "position_ids": [torch.zeros(1, dtype=torch.long, device=device)],
                    "repeat_nums": [[1]]
                }
            
            tree_indices_list = [torch.zeros(ml, dtype=torch.long, device=device) for ml in depth_counts if ml > 0]
            repeat_nums = [[] for _ in depth_counts if _ > 0]
            
            # Handle case where tree_indices_list is empty
            if not tree_indices_list:
                return {
                    "attn_mask": [torch.eye(1, device=device).unsqueeze(0).unsqueeze(0)],
                    "tree_indices": [torch.zeros(1, dtype=torch.long, device=device)],
                    "position_ids": [torch.zeros(1, dtype=torch.long, device=device)],
                    "repeat_nums": [[1]]
                }
            
            start = 0
            bias = 0
            for i in range(len(tree_indices_list)):
                bias = 0
                repeat_j = 0
                parent = None
                
                for j in range(depth_counts[i]):
                    if start + j < len(nodes_wc):
                        cur_node = nodes_wc[start + j]
                        cur_parent = cur_node.parent
                        
                        if j != 0:
                            if cur_parent != parent:
                                bias += 1
                                parent = cur_parent
                                repeat_nums[i].append(j - repeat_j)
                                repeat_j = j
                        else:
                            parent = cur_parent
                        
                        # Ensure value is within bounds
                        value = min(cur_node.value, TOPK - 1) if hasattr(cur_node, 'value') else 0
                        tree_indices_list[i][j] = value + TOPK * bias
                
                # Ensure j is defined
                if depth_counts[i] > 0:
                    j = depth_counts[i] - 1
                    repeat_nums[i].append(j - repeat_j + 1)
                else:
                    repeat_nums[i].append(1)  # Default value
                
                start += depth_counts[i]
            
            position_ids = [torch.zeros(ml, dtype=torch.long, device=device) for ml in depth_counts if ml > 0]
            
            # Handle case where position_ids is empty
            if not position_ids:
                position_ids = [torch.zeros(1, dtype=torch.long, device=device)]
            
            tree_buffers = {
                "attn_mask": [i.unsqueeze(0).unsqueeze(0) for i in tree_attn_mask_list],
                "tree_indices": tree_indices_list,
                "position_ids": position_ids,
                "repeat_nums": repeat_nums
            }
            
            return tree_buffers
            
        except Exception as e:
            # Fallback to simple tree buffer in case of any error
            print(f"Error in tree buffer generation: {e}. Using fallback.")
            return {
                "attn_mask": [torch.eye(1, device=device).unsqueeze(0).unsqueeze(0)],
                "tree_indices": [torch.zeros(1, dtype=torch.long, device=device)],
                "position_ids": [torch.zeros(1, dtype=torch.long, device=device)],
                "repeat_nums": [[1]]
            }
    
    def _build_tree(self, tree_list):
        """
        Build a tree structure from a list of paths.
        This is a simplified version of EAGLE's Tree class.
        """
        class Node:
            def __init__(self, parent=None, value=None, dict_key=None):
                self.parent = parent
                self.value = value
                if parent:
                    self.depth = parent.depth + 1
                    parent.children.append(self)
                else:
                    self.depth = 0
                self.children = []
                self.dict_key = dict_key
                self.index = None
                # Add probability tracking for tree pruning
                self.probability = 1.0
                self.score = 0.0

            def is_leaf(self):
                return len(self.children) == 0

            def all_index(self):
                if not self.parent or not self.parent.parent:
                    return [self.index]
                else:
                    return self.parent.all_index() + [self.index]
                    
            def prune_low_probability_children(self, threshold=0.01):
                """Prune children with probability below threshold"""
                if not self.children:
                    return
                    
                # Get total probability of all children
                total_prob = sum(child.probability for child in self.children)
                
                # Normalize probabilities
                for child in self.children:
                    child.probability /= (total_prob + 1e-10)
                
                # Prune low probability children
                self.children = [child for child in self.children if child.probability >= threshold]
                
                # Recursively prune children's children
                for child in self.children:
                    child.prune_low_probability_children(threshold)
        
        class Tree:
            def __init__(self, tree_list):
                sorted_tree_list = sorted(tree_list, key=lambda x: (len(x), x))
                self.root = Node()
                self.node_dic = {}
                for tree_node in sorted_tree_list:
                    cur_value = tree_node[-1]
                    if len(tree_node) == 1:
                        cur_node = Node(parent=self.root, value=cur_value, dict_key=tuple(tree_node))
                    else:
                        cur_parent = self.node_dic[tuple(tree_node[:-1])]
                        cur_node = Node(parent=cur_parent, value=cur_value, dict_key=tuple(tree_node))
                    self.node_dic[tuple(tree_node)] = cur_node
                self.indexnode()

            def max_depth(self):
                return max([item.depth for item in self.node_dic.values()])

            def num_node_wchild(self):
                num_c = 0
                for item in self.node_dic.values():
                    if not item.is_leaf():
                        num_c += 1
                return num_c

            def get_node_wchild(self):
                ns = []
                for item in self.node_dic.values():
                    if not item.is_leaf():
                        ns.append(item)
                return ns

            def indexnode(self):
                cur_index = 0
                for key in self.node_dic:
                    cur_node = self.node_dic[key]
                    if not cur_node.is_leaf():
                        cur_node.index = cur_index
                        cur_index += 1
        
        return Tree(tree_list)
        
    def adjust_tree_dynamically(self, logits, threshold=0.7):
        """
        Dynamically adjust the tree structure based on confidence scores (EAGLE-2 style).
        
        Args:
            logits: Logits from the model
            threshold: Confidence threshold for adjusting the tree
            
        Returns:
            adjusted_tree: Adjusted tree structure
        """
        # Calculate confidence scores from logits
        probs = F.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, self.top_k, dim=-1)
        
        # Calculate entropy and confidence metrics
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        confidence = top_probs[:, 0] / (top_probs.sum(dim=-1) + 1e-10)
        
        # Calculate token distribution statistics
        mean_prob = probs.mean().item()
        std_prob = probs.std().item()
        max_prob = probs.max().item()
        
        # EAGLE-2 style dynamic tree adjustment with more sophisticated metrics
        if confidence.mean() > threshold:
            # High confidence - use chain structure for efficiency
            self.tree_structure = chain_structure
            # Reduce tree depth for high confidence
            self.depth = min(self.depth, 2)
            # Reduce branching factor
            self.top_k = max(5, self.top_k - 2)
        else:
            # Low confidence - use full tree structure for exploration
            self.tree_structure = tree_structure
            
            # Adjust tree depth based on entropy
            if entropy.mean() > 2.0:  # High entropy threshold
                # Very uncertain - use deeper tree
                self.depth = min(self.depth, 4)
                # Increase top_k for more exploration
                self.top_k = min(self.top_k + 2, 15)
            elif entropy.mean() < 1.0:  # Low entropy threshold
                # More certain - use shallower tree
                self.depth = max(1, self.depth - 1)
                # Decrease top_k for more focus
                self.top_k = max(5, self.top_k - 2)
            else:
                # Medium entropy - use moderate tree depth
                self.depth = 3
                # Use moderate top_k
                self.top_k = 10
                
            # Further adjust based on token distribution
            if max_prob > 0.9:  # Very confident in one token
                # Use chain structure with limited branching
                self.tree_structure = chain_structure
                self.depth = 2
                self.top_k = 5
            elif std_prob < 0.01:  # Very uniform distribution
                # Use full tree with extensive branching
                self.tree_structure = tree_structure
                self.depth = 4
                self.top_k = 15
            
        # Regenerate tree buffers with new structure
        return self.tree_structure
        
    def sample_next_token(self, logits, temperature=0.0, top_p=1.0, top_k=None):
        """Sample next token from logits using temperature, top-p, and top-k."""
        if top_k is None:
            top_k = self.top_k
            
        # Apply temperature
        if temperature > 0:
            logits = logits / temperature
            
        # Get probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Apply top-k if specified
        if top_k > 0:
            top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
            # Create a new distribution with only top-k
            new_probs = torch.zeros_like(probs)
            new_probs.scatter_(-1, top_k_indices, top_k_probs)
            probs = new_probs
            
        # Apply top-p (nucleus) sampling
        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep the first token above threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # Create a mask for indices to remove
            indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
            probs = probs.masked_fill(indices_to_remove, 0.0)
            
        # Renormalize probabilities
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-10)
        
        # Sample from the distribution
        if temperature == 0:
            # Greedy sampling
            next_token = torch.argmax(probs, dim=-1)
        else:
            # Sample from the distribution
            next_token = torch.multinomial(probs, 1).squeeze(-1)
            
        return next_token, probs
        
    def verify_speculative_tokens(self, input_ids, speculative_ids):
        """
        Verify speculative tokens against the model's predictions.
        This is a key part of EAGLE's approach to ensure consistency.
        
        Args:
            input_ids: Input token IDs
            speculative_ids: Speculative token IDs to verify
            
        Returns:
            accepted_length: Number of accepted tokens
            next_token: Next token prediction after accepted tokens
        """
        # Concatenate input and speculative IDs
        full_ids = torch.cat([input_ids, speculative_ids], dim=-1)
        
        # Forward pass through the model
        with torch.no_grad():
            outputs = self.model(full_ids.unsqueeze(0))
            logits = outputs.logits
        
        # Get the model's predictions for each position
        input_len = input_ids.shape[0]
        spec_len = speculative_ids.shape[0]
        
        # Check each speculative token against model prediction
        accepted_length = 0
        for i in range(spec_len):
            pos = input_len + i - 1  # Position to check
            pred_logits = logits[0, pos]  # Model's prediction at this position
            pred_token = torch.argmax(pred_logits).item()  # Most likely token
            spec_token = speculative_ids[i].item()  # Speculative token
            
            # Check if prediction matches speculation
            if pred_token == spec_token:
                accepted_length += 1
            else:
                # Stop at first mismatch
                break
        
        # Get next token prediction after accepted tokens
        if accepted_length < spec_len:
            next_pos = input_len + accepted_length - 1
            next_logits = logits[0, next_pos]
            next_token = torch.argmax(next_logits).item()
        else:
            # All tokens accepted, predict next token
            next_logits = logits[0, -1]
            next_token = torch.argmax(next_logits).item()
        
        return accepted_length, torch.tensor([next_token], device=input_ids.device)
    
    def process_tree_node(self, node_idx, tree_nodes, tree_parents, tree_scores, level_tokens, outputs, i):
        """
        Process a single tree node for parallel tree processing.
        """
        token = level_tokens[0, node_idx]
        tree_nodes.append(token.unsqueeze(0))
        
        # Calculate score based on logits
        if i == 0:
            parent_idx = 0
            score = outputs.logits[0, node_idx, token.item()].unsqueeze(0)
        else:
            # Find parent index
            parent_path = self.tree_structure[node_idx][:-1] if node_idx < len(self.tree_structure) else [0]
            parent_idx = 0
            for k, path in enumerate(self.tree_structure):
                if path == parent_path:
                    parent_idx = k + 1  # +1 because root is at index 0
                    break
            
            # Calculate score
            parent_score = tree_scores[parent_idx]
            token_logit = outputs.logits[0, node_idx, token.item()]
            score = parent_score + token_logit.unsqueeze(0)
        
        tree_scores.append(score)
        tree_parents.append(parent_idx)
        
        return node_idx
        
    def speculative_sampling(self, input_ids, max_new_tokens=100, temperature=0.0, top_p=1.0, stop_token=None):
        """
        Perform speculative sampling using the tree-based draft model.
        This is a key part of EAGLE's approach for efficient generation.
        
        Args:
            input_ids: Input token IDs
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            stop_token: Token to stop generation
            
        Returns:
            output_ids: Generated token IDs
        """
        # Initialize output with input
        output_ids = input_ids.clone()
        
        # Get EOS token ID
        eos_token_id = self.model.config.eos_token_id
        
        # Generate tokens
        tokens_generated = 0
        while tokens_generated < max_new_tokens:
            # Generate tree candidates
            candidates, scores, _ = self.generate_eagle_tree_candidates(
                output_ids, 
                temperature=temperature, 
                top_p=top_p,
                stop_token=stop_token
            )
            
            if len(candidates) == 0:
                break
                
            # Select best candidate
            best_idx = torch.argmax(scores).item()
            best_candidate = candidates[best_idx]
            
            # Get speculative tokens (excluding input)
            speculative_ids = best_candidate[len(output_ids):]
            
            # Limit number of tokens to verify at once
            if len(speculative_ids) > self.max_verification_tokens:
                speculative_ids = speculative_ids[:self.max_verification_tokens]
            
            # Verify speculative tokens
            accepted_length, next_token = self.verify_speculative_tokens(output_ids, speculative_ids)
            
            # Add accepted tokens to output
            if accepted_length > 0:
                output_ids = torch.cat([output_ids, speculative_ids[:accepted_length]], dim=0)
            
            # Add next token
            output_ids = torch.cat([output_ids, next_token], dim=0)
            
            # Update tokens generated
            tokens_generated += accepted_length + 1
            
            # Check for EOS or stop token
            if output_ids[-1].item() == eos_token_id:
                break
            elif stop_token is not None:
                decoded = self.tokenizer.decode(output_ids[-10:], skip_special_tokens=True)
                if stop_token in decoded:
                    break
            
            # Break if we've generated enough tokens
            if tokens_generated >= max_new_tokens:
                break
        
        return output_ids
        
    def generate_long_candidates(self, prompt, temperature=0.0, top_p=1.0, max_tokens=100, stop_token=None):
        """
        Generate longer candidate sequences using a simpler, more robust approach.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_tokens: Maximum number of tokens to generate
            stop_token: Token to stop generation
            
        Returns:
            candidates: List of candidate sequences
            scores: Scores for each candidate
        """
        # Tokenize input if it's a string
        if isinstance(prompt, str):
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        else:
            input_ids = prompt.to(self.device)
        
        # Get EOS token ID
        eos_token_id = self.model.config.eos_token_id
        
        # Initialize tracking variables
        current_input = input_ids
        total_tokens_generated = 0
        cumulative_score = 0.0
        
        # Reset cache
        self.reset_cache()
        
        # Create a simple chain structure for more reliable generation
        original_tree_structure = self.tree_structure
        original_depth = self.depth
        
        # Use a simple chain structure with limited width
        self.tree_structure = [[0], [0, 0], [0, 0, 0]]
        self.depth = 3
        
        # Generate tokens in chunks
        generated_ids = input_ids[0].clone()
        
        # Use standard autoregressive generation with the model
        with torch.no_grad():
            # Initial forward pass
            outputs = self.model(
                input_ids, 
                use_cache=True
            )
            past_key_values = outputs.past_key_values
            
            # Generate tokens autoregressively
            for i in range(max_tokens):
                # Get the last token's logits
                logits = outputs.logits[:, -1, :]
                
                # Process logits with temperature and top_p
                if temperature > 0:
                    logits = logits / temperature
                
                # Apply top_p sampling
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep the first token above threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter_(
                        -1, sorted_indices, sorted_indices_to_remove
                    )
                    logits = logits.masked_fill(indices_to_remove, -float("Inf"))
                
                # Get the next token
                next_token_id = torch.argmax(logits, dim=-1).unsqueeze(0).unsqueeze(0)
                
                # Add to generated sequence - ensure dimensions match
                next_token = next_token_id[0, 0].unsqueeze(0)  # Shape: [1]
                generated_ids = torch.cat([generated_ids, next_token])
                
                # Update score
                token_score = logits[0, next_token_id[0, 0]].item()
                cumulative_score += token_score
                
                # Check for EOS or stop token
                if next_token_id[0, 0].item() == eos_token_id:
                    break
                elif stop_token is not None:
                    # Check if the generated text contains the stop token
                    recent_text = self.tokenizer.decode(generated_ids[-10:], skip_special_tokens=True)
                    if stop_token in recent_text:
                        break
                
                # Forward pass with the new token
                outputs = self.model(
                    input_ids=next_token_id,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                past_key_values = outputs.past_key_values
                
                # Update token count
                total_tokens_generated += 1
                
                # Break if we've generated enough tokens
                if total_tokens_generated >= max_tokens:
                    break
        
        # Restore original tree structure
        self.tree_structure = original_tree_structure
        self.depth = original_depth
        
        # Create candidates list with the generated sequence
        candidates = [generated_ids]
        scores = torch.tensor([cumulative_score], device=self.device)
        
        return candidates, scores
        
    def repeat_hidden(self, hidden_state, repeat_num):
        """
        Repeat hidden states according to EAGLE's implementation.
        """
        new_hidden = []
        for id, i in enumerate(repeat_num):
            new_hidden.append(hidden_state[:, id:id + 1].repeat(1, i, 1))
        return torch.cat(new_hidden, dim=1)
        
    def update_kv_cache(self, past_key_values, batch_indices, select_indices, prev_input_len, new_tokens_shape):
        """
        Update KV cache for tree generation, similar to EAGLE's approach.
        """
        new_kv = ()
        
        for past_key_values_data in past_key_values:
            layer_kv = ()
            for korv in past_key_values_data:
                # Select the relevant KV cache entries
                tgt = korv[batch_indices, :, select_indices, :]
                tgt = tgt.permute(0, 2, 1, 3)
                
                # Update the KV cache
                dst = korv[:, :, prev_input_len: prev_input_len + tgt.shape[-2], :]
                dst.copy_(tgt, non_blocking=True)
                
                # Add to layer KV cache
                layer_kv += (korv[:, :, : prev_input_len + tgt.shape[-2], :],)
            
            # Add to new KV cache
            new_kv += (layer_kv,)
        
        return new_kv
        
    def optimize_kv_cache(self, past_key_values, accepted_length):
        """
        Optimize KV cache by pruning unused entries.
        This is an important optimization in EAGLE for long sequences.
        """
        if accepted_length == 0:
            return past_key_values
            
        # Create optimized KV cache
        optimized_kv = []
        
        for layer_idx, layer_kv in enumerate(past_key_values):
            optimized_layer = []
            for kv_idx, kv in enumerate(layer_kv):
                # Keep only the used entries (up to accepted_length)
                optimized_kv_tensor = kv[:, :, :accepted_length, :].clone()
                optimized_layer.append(optimized_kv_tensor)
            optimized_kv.append(tuple(optimized_layer))
        
        return tuple(optimized_kv)
    def visualize_tree_structure(self, max_depth=10):
        """
        Visualize the current tree structure.
        
        Args:
            max_depth: Maximum depth to visualize for dynamic trees
            
        Returns:
            str: A string representation of the tree structure
        """
        if hasattr(self, 'dynamic_tree') and self.dynamic_tree:
            # Check if we're using fixed nodes per level
            if isinstance(self.tree_generator, FixedNodesPerLevelTree):
                result = "Fixed Nodes Per Level Tree Structure Visualization:\n"
                nodes_per_level = self.tree_generator.nodes_per_level
                result += f"Nodes per level: {nodes_per_level}\n"
                
                # Show nodes at each level
                for depth in range(1, min(max_depth, self.depth) + 1):
                    result += f"  Level {depth}: {nodes_per_level} nodes\n"
            else:
                result = "Bounded Width Tree Structure Visualization:\n"
                max_width = self.tree_generator.max_width
                result += f"Maximum width at each level: {max_width}\n"
                
                # Calculate total nodes at each level up to max_depth
                total_nodes = 1  # Root node
                for depth in range(1, min(max_depth, self.depth) + 1):
                    nodes_at_level = max_width ** depth
                    result += f"  Level {depth}: {nodes_at_level} nodes (width = {max_width})\n"
            
            # Show some example paths
            result += "\nExample paths (first few levels):\n"
            sample_paths = self.tree_generator.get_all_paths_up_to_depth(min(3, max_depth))
            for i, path in enumerate(sample_paths[:10]):
                result += f"  {path}\n"
            
            if len(sample_paths) > 10:
                result += f"  ... and more paths\n"
            
            return result
        elif not hasattr(self, 'tree_structure') or not self.tree_structure:
            return "No tree structure defined"
        else:
            # Static tree structure visualization
            # Count nodes at each depth
            depth_counts = {}
            for path in self.tree_structure:
                depth = len(path)
                if depth not in depth_counts:
                    depth_counts[depth] = 0
                depth_counts[depth] += 1
            
            # Create visualization
            result = "Tree Structure Visualization:\n"
            result += f"Total paths: {len(self.tree_structure)}\n"
            result += "Width at each depth level:\n"
            
            for depth, count in sorted(depth_counts.items()):
                result += f"  Depth {depth}: {count} nodes\n"
            
            # Show some example paths
            result += "\nExample paths:\n"
            for i, path in enumerate(self.tree_structure[:10]):  # Show first 10 paths
                result += f"  {path}\n"
            
            if len(self.tree_structure) > 10:
                result += f"  ... and {len(self.tree_structure) - 10} more paths\n"
                
            return result
    def get_paths_at_depth(self, depth):
        """
        Get tree paths at a specific depth.
        
        Args:
            depth: The depth to get paths for (1-indexed)
            
        Returns:
            List of paths at the specified depth
        """
        if hasattr(self, 'dynamic_tree') and self.dynamic_tree:
            return self.tree_generator.get_paths_at_depth(depth)
        else:
            # For static tree structure, filter paths with the specified length
            return [path for path in self.tree_structure if len(path) == depth]
class FixedNodesPerLevelTree:
    """
    A class that generates tree paths with a fixed number of nodes at each level.
    """
    def __init__(self, nodes_per_level=4, max_cached_depth=10):
        """
        Initialize the tree with fixed nodes per level.
        
        Args:
            nodes_per_level: Maximum number of nodes at each level
            max_cached_depth: Maximum depth to cache paths for
        """
        self.nodes_per_level = nodes_per_level
        self.max_cached_depth = max_cached_depth
        self.cached_paths = {}
        
        # Initialize cache with first level paths
        self._initialize_cache()
    
    def _initialize_cache(self):
        """Initialize the cache with paths up to max_cached_depth"""
        # First level (single tokens) - exactly nodes_per_level nodes
        level_1 = [[i] for i in range(self.nodes_per_level)]
        self.cached_paths[1] = level_1
        
        # Generate paths for deeper levels up to max_cached_depth
        for depth in range(2, self.max_cached_depth + 1):
            # For each level, we select exactly nodes_per_level paths
            parent_paths = self.cached_paths[depth-1]
            current_level_paths = []
            
            # Calculate how many children per parent to maintain nodes_per_level
            children_per_parent = max(1, self.nodes_per_level // len(parent_paths))
            
            # Generate children for each parent
            for i, parent_path in enumerate(parent_paths):
                # Skip parents beyond what we need
                if i * children_per_parent >= self.nodes_per_level:
                    break
                    
                # Generate children for this parent
                for j in range(children_per_parent):
                    # Skip if we've reached the maximum nodes for this level
                    if len(current_level_paths) >= self.nodes_per_level:
                        break
                    
                    # Create new path by extending parent path
                    new_path = parent_path + [j]
                    current_level_paths.append(new_path)
            
            # Ensure we have exactly nodes_per_level paths
            if len(current_level_paths) > self.nodes_per_level:
                current_level_paths = current_level_paths[:self.nodes_per_level]
            
            self.cached_paths[depth] = current_level_paths
    
    def get_paths_at_depth(self, depth):
        """Get all paths at the specified depth"""
        if depth in self.cached_paths:
            return self.cached_paths[depth]
        
        # For depths beyond cache, generate on-the-fly
        if depth <= 1:
            return [[i] for i in range(self.nodes_per_level)]
        
        # Get parent paths and generate exactly nodes_per_level children
        parent_paths = self.get_paths_at_depth(depth-1)
        current_level_paths = []
        
        # Calculate children per parent as before
        children_per_parent = max(1, self.nodes_per_level // len(parent_paths))
        
        # Generate children for each parent
        for i, parent_path in enumerate(parent_paths):
            if i * children_per_parent >= self.nodes_per_level:
                break
                
            for j in range(children_per_parent):
                if len(current_level_paths) >= self.nodes_per_level:
                    break
                
                new_path = parent_path + [j]
                current_level_paths.append(new_path)
        
        # Ensure we have exactly nodes_per_level paths
        if len(current_level_paths) > self.nodes_per_level:
            current_level_paths = current_level_paths[:self.nodes_per_level]
        
        # Cache if within limit
        if depth <= self.max_cached_depth:
            self.cached_paths[depth] = current_level_paths
            
        return current_level_paths
    
    def get_all_paths_up_to_depth(self, max_depth):
        """Get all paths up to the specified maximum depth"""
        all_paths = []
        for depth in range(1, max_depth + 1):
            all_paths.extend(self.get_paths_at_depth(depth))
        return all_paths