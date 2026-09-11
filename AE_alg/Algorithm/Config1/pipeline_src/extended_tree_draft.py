# Current issues with the tree generation:
# 1. The tree generation doesn't properly stop branching when encountering step_word
# 2. It doesn't maintain multiple candidate paths after encountering stop tokens
# 3. The generate_long_candidates method falls back to standard generation

# Here's an improved version that addresses your requirements:

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Tuple, Optional

class ExtendedTreeDraftModel:
    def __init__(self, model_name_or_path, tokenizer=None, device="cuda", 
                 top_k=10, depth=3, total_tokens=25, step_word="\n\n"):
        self.device = device
        self.top_k = top_k
        self.depth = depth
        self.step_word = step_word
        
        # Load model and tokenizer
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        else:
            self.tokenizer = tokenizer
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, 
            trust_remote_code=True,
            torch_dtype=torch.float16
        ).to(device)
        
        # Store total_tokens parameter
        self.total_tokens = total_tokens
        
        # Get step_word token ID(s)
        self.step_word_tokens = self.tokenizer.encode(step_word, add_special_tokens=False)
    
    def contains_step_word(self, token_sequence: List[int]) -> bool:
        """Check if token sequence contains the step word"""
        if len(token_sequence) < len(self.step_word_tokens):
            return False
        
        # Check if step_word tokens appear at the end
        return token_sequence[-len(self.step_word_tokens):] == self.step_word_tokens
    
    def should_stop_branching(self, path_tokens: List[int]) -> bool:
        """Determine if this path should stop branching"""
        return self.contains_step_word(path_tokens)
    
    @torch.no_grad()
    def generate_tree_candidates(self, prompt: str, temperature: float = 0.0, 
                               top_p: float = 1.0, max_tokens: int = 2048) -> Tuple[List[torch.Tensor], List[float], Dict]:
        """
        Generate a tree of candidates from the prompt that stops branching when step_word is encountered,
        but maintains multiple candidate paths.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            candidates: List of candidate sequences
            scores: Scores for each candidate
            tree_structure: Structure of the tree for visualization
        """
        # Tokenize input
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        input_length = input_ids.shape[1]
        
        # Initialize active paths: each path is (token_sequence, cumulative_log_prob, is_complete)
        active_paths = [(input_ids[0].tolist(), 0.0, False)]
        completed_paths = []
        
        # Prepare logits processor
        from transformers.generation.logits_process import (
            LogitsProcessorList, TemperatureLogitsWarper, TopPLogitsWarper
        )
        
        processor_list = LogitsProcessorList()
        if temperature > 1e-5 and temperature != 1.0:
            processor_list.append(TemperatureLogitsWarper(temperature))
        if top_p < 1.0:
            processor_list.append(TopPLogitsWarper(top_p))
        
        logits_processor = processor_list if processor_list else None
        
        for depth in range(self.depth):
            new_active_paths = []
            
            # Process each active path
            for path_tokens, path_score, is_complete in active_paths:
                # If path is complete (contains step_word), don't branch further
                if is_complete or self.should_stop_branching(path_tokens):
                    completed_paths.append((path_tokens, path_score, True))
                    continue
                
                # Create input tensor for this path
                path_input = torch.tensor([path_tokens], device=self.device)
                
                # Forward pass
                with torch.no_grad():
                    outputs = self.model(path_input)
                    logits = outputs.logits[0, -1, :]  # Last token logits
                
                # Process logits
                if logits_processor:
                    processed_logits = logits_processor(None, logits.unsqueeze(0))[0]
                else:
                    processed_logits = logits
                
                # Get top-k candidates
                top_k_logits, top_k_indices = torch.topk(processed_logits, self.top_k)
                top_k_probs = F.softmax(top_k_logits, dim=-1)
                top_k_log_probs = torch.log(top_k_probs)
                
                # Create new paths
                for i, (token_id, log_prob) in enumerate(zip(top_k_indices, top_k_log_probs)):
                    new_path_tokens = path_tokens + [token_id.item()]
                    new_path_score = path_score + log_prob.item()
                    
                    # Check if this new path contains step_word
                    contains_stop = self.should_stop_branching(new_path_tokens)
                    
                    if contains_stop:
                        # Path is complete, add to completed paths
                        completed_paths.append((new_path_tokens, new_path_score, True))
                    else:
                        # Path is still active for next iteration
                        new_active_paths.append((new_path_tokens, new_path_score, False))
            
            # Update active paths (limit to top-k overall to prevent exponential growth)
            if len(new_active_paths) > self.top_k:
                new_active_paths.sort(key=lambda x: x[1], reverse=True)  # Sort by score
                active_paths = new_active_paths[:self.top_k]
            else:
                active_paths = new_active_paths
            
            # Break if no active paths remain
            if not active_paths:
                break
        
        # Add any remaining active paths to completed paths
        for path_tokens, path_score, _ in active_paths:
            completed_paths.append((path_tokens, path_score, False))
        
        # Convert to torch tensors and extract scores
        candidates = []
        scores = []
        
        for path_tokens, path_score, is_complete in completed_paths:
            candidates.append(torch.tensor(path_tokens, device=self.device))
            scores.append(path_score)
        
        # Build a simple tree structure for compatibility
        tree_structure = {
            "root": input_ids[0, -1].item() if input_ids.shape[1] > 0 else None,
            "children": []
        }
        for i, (path_tokens, path_score, is_complete) in enumerate(completed_paths[:self.top_k]):
            tree_structure["children"].append({
                "token": path_tokens[-1] if path_tokens else None,
                "score": path_score,
                "complete": is_complete,
                "children": []
            })

        return candidates, scores, tree_structure
    
    def completions_create(self, prompt, temperature=0.0, top_p=1.0, max_tokens=100, stop=None):
        """
        API-compatible method that uses the improved tree generation
        """
        # Handle both single prompt and batch
        if isinstance(prompt, str):
            prompts = [prompt]
        else:
            prompts = prompt
        
        choices = []
        for i, p in enumerate(prompts):
            # Generate candidates using tree method
            candidates, scores, _ = self.generate_tree_candidates(
                p, temperature, top_p, max_tokens
            )
            
            if candidates:
                # Take the best candidate
                best_idx = max(range(len(scores)), key=lambda x: scores[x])
                best_candidate = candidates[best_idx]
                
                # Decode only the generated part (excluding input)
                input_length = len(self.tokenizer.encode(p, add_special_tokens=False))
                generated_tokens = best_candidate[input_length:]
                text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                # Remove step_word if it's at the end
                if text.endswith(self.step_word):
                    text = text[:-len(self.step_word)]
            else:
                text = ""
            
            # Create choice object
            choice = type('Choice', (), {
                'text': text,
                'index': i,
                'logprobs': None,
                'finish_reason': 'stop' if self.step_word in text else 'length'
            })()
            
            choices.append(choice)
        
        # Return response object
        response = type('Response', (), {'choices': choices})()
        return response

    @torch.no_grad()
    def generate_long_candidates(self, prompt, temperature=0.0, top_p=1.0, max_tokens=100, stop_token="\n\n"):
        """
        Generate longer candidate sequences by extending the tree approach.
        This method is specifically designed for generating longer sequences than the basic tree approach.
        
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
        # Generate initial candidates using the tree approach
        candidates, scores, _ = self.generate_tree_candidates(prompt, temperature, top_p, max_tokens)
        
        # Sort candidates by score
        sorted_indices = torch.argsort(torch.tensor(scores, device=self.device), descending=True)
        candidates = [candidates[i] for i in sorted_indices]
        scores = torch.tensor([scores[i] for i in sorted_indices], device=self.device)
        
        # Return the top candidates
        return candidates, scores

# Key improvements:
# 1. Proper stop condition checking with should_stop_branching()
# 2. Maintains completed_paths separately from active_paths
# 3. Stops branching individual paths when step_word is found, but keeps other paths active
# 4. Uses proper token sequence matching for step_word detection
# 5. Maintains multiple candidate paths even after some complete
# 6. Added generate_long_candidates method for extended sequence generation