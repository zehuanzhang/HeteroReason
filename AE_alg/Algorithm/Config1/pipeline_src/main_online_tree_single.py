import random
import os
import argparse
import time
import json
import math
import multiprocessing as mp
import atexit
import signal
import sys
from datetime import datetime
from tqdm import tqdm
import torch.cuda.nvtx as nvtx

from transformers import AutoTokenizer
from openai import OpenAI

from external.qwen25_math_evaluation.evaluate import evaluate
from external.qwen25_math_evaluation.utils import set_seed, load_jsonl, save_jsonl, construct_prompt
from external.qwen25_math_evaluation.parser import *
from external.qwen25_math_evaluation.trajectory import *
from external.qwen25_math_evaluation.data_loader import load_data
from external.qwen25_math_evaluation.python_executor import PythonExecutor
from external.skywork_o1_prm_inference.model_utils.io_utils import prepare_input, derive_step_rewards_vllm
from target_trace import attach_target_trace, build_target_trace, get_target_trace

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_n_paths", type=int, default=3, help="Number of top unique paths to print in reranking logs")
    parser.add_argument("--beam_expansion_factor", type=int, default=2, help="Number of candidates to generate per beam node")
    parser.add_argument("--data_names", default="math500,gsm8k,gaokao2023en,olympiadbench", type=str) #"math500,gsm8k,gaokao2023en,olympiadbench"
    parser.add_argument("--data_dir", default="./external/qwen25_math_evaluation/data", type=str)
    parser.add_argument("--draft_model_name_or_path", default="Qwen/Qwen2.5-Math-1.5B-Instruct", type=str)
    parser.add_argument("--draft_model_ip_address", default="http://localhost:12340/v1", type=str)
    parser.add_argument("--target_model_name_or_path", default="Qwen/Qwen2.5-Math-7B-Instruct", type=str)
    parser.add_argument("--target_model_ip_address", default="http://localhost:12341/v1", type=str)
    parser.add_argument("--prm_name_or_path", default="Skywork/", type=str)
    parser.add_argument("--prm_ip_address", default="http://localhost:12342/v1", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--prompt_type", default="qwen25-math-cot", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int)  # -1 for full data
    parser.add_argument("--fixed_subset_size", default=0, type=int, help="Use a fixed subset of examples (0 to disable)")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--n_sampling", default=1, type=int)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--max_tokens_per_call", default=2048, type=int)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--save_outputs", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--use_safetensors", action="store_true")
    parser.add_argument("--num_shots", type=int, default=0)
    parser.add_argument("--step_word", type=str, default="\n\n") # <extra_0> for qwen-math-prm-7b, \n\n for skywork
    parser.add_argument("--prm_threshold", type=float, default=0.7) 
    parser.add_argument("--backtrack_threshold", type=float, default=None,
                       help="Threshold for triggering backtracking (defaults to prm_threshold)")
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--apply_chat_template", action="store_true", help="Apply chat template to prompt.")
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--adapt_few_shot", action="store_true", help="Few shot for multiple-choice questions, zero shot for others.")
    
    # Search strategy options
    parser.add_argument("--beam_search", action="store_true",
                       help="Enable beam search (maintains global top-k candidates)")
    
    # Beam search parameters
    parser.add_argument("--beam_width", type=int, default=3,
                       help="Number of candidates to keep in beam search")
    parser.add_argument("--beam_use_log_scores", action="store_true",
                       help="Use cumulative log scores for beam search")
    
    # Shared parameters
    parser.add_argument("--search_temperature", type=float, default=0.7,
                       help="Temperature for search expansion (higher = more diverse)")
    
    # Backtracking parameters
    parser.add_argument("--enable_backtracking", action="store_true",
                       help="Enable backtracking when all candidates fall below PRM threshold")
    parser.add_argument("--max_backtrack_steps", type=int, default=10,#3,
                       help="Maximum number of steps to backtrack")
    parser.add_argument("--max_backtrack_attempts", type=int, default=5,#2,
                       help="Maximum number of backtracking attempts per node")
    parser.add_argument("--backtrack_expansion_factor", type=int, default=None,
                       help="Number of candidates to generate during backtracking target intervention (defaults to beam_expansion_factor)")

    # Alternative to classic backtracking: if target intervention is still low-PRM,
    # append a corrective prompt and continue generation instead of jumping to previous steps.
    parser.add_argument("--enable_rethink_prompt", action="store_true",
                        help="When target intervention falls below PRM threshold, append a 'rethink the step above' prompt instead of backtracking")
    parser.add_argument(
        "--rethink_prompt_text",
        type=str,
        default=(
            "\n\nThe previous step is incorrect. Ignore it completely. "
            "Rewrite that step correctly as a replacement. "
            "Output exactly ONE corrected step, ending with a period '.', then the step delimiter.\n\n"
        ),
        help="Prompt appended after a low-PRM intervened target step",
    )
    parser.add_argument(
        "--max_rethink_attempts",
        type=int,
        default=1,
        help="Maximum number of rethink-prompt insertions allowed per path (to avoid infinite loops)",
    )
    
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--use_local_models", action="store_true", help="Use all local models instead of server-based models")
    parser.add_argument("--enable_target_prefetch", action="store_true",
                        help="Enable asynchronous target prefetch when supported (local target models only)")
    parser.add_argument("--enable_prefetch_cache", action="store_true",
                        help="Cache partial target outputs when stopping prefetched jobs for reuse")
    # [DraftPipelineChange] Optional experiment: run draft for the next step while PRM scores the current step.
    parser.add_argument("--enable_draft_pipeline_prefetch", action="store_true",
                        help="Prefetch the next draft step during PRM scoring (beam_width=1, expansion=1 only)")
    
    # Memory management options
    parser.add_argument("--max_batch_size", type=int, default=32, 
                       help="Maximum batch size for model calls to control memory usage")
    parser.add_argument("--memory_cleanup_freq", type=int, default=1,
                       help="Frequency of memory cleanup (every N problems, 0 to disable)")
    
    # Hierarchical speculative decoding options
    parser.add_argument("--enable_hierarchical_spec", action="store_true",
                       help="Enable hierarchical speculative decoding: 0.5B token drafter -> 1.5B step drafter -> 7B target")
    parser.add_argument("--token_drafter_path", type=str, 
                       default="/mnt/ccnas2/bdp/zr523/RSD/Qwen2.5-0.5B-Instruct",
                       help="Path to the 0.5B token-level drafter for hierarchical spec decoding")
    parser.add_argument("--num_speculative_tokens", type=int, default=5,
                       help="Number of tokens the 0.5B model drafts at a time for speculative decoding")

    args = parser.parse_args()
    args.top_p = 1 if args.temperature == 0 else args.top_p  # top_p must be 1 when using greedy sampling (vllm)
    
    if args.backtrack_threshold is None:
        args.backtrack_threshold = args.prm_threshold

    # Set default for backtrack_expansion_factor if not specified
    if args.backtrack_expansion_factor is None:
        args.backtrack_expansion_factor = args.beam_expansion_factor
    
    # Validate search options
    if not args.beam_search:
        print("[INFO] No search method specified, using default sequential generation")
    
    # Validate backtracking parameters
    if args.enable_backtracking and not args.beam_search:
        print("[WARNING] Backtracking enabled but beam search not specified. Backtracking requires beam search and will be ignored.")
        args.enable_backtracking = False

    if args.enable_prefetch_cache and not args.enable_target_prefetch:
        print("[WARNING] Prefetch cache requested but target prefetch is disabled; disabling cache option.")
        args.enable_prefetch_cache = False

    # [DraftPipelineChange] Keep this experiment scoped to the single-path bbeam1 setting.
    if args.enable_draft_pipeline_prefetch:
        supported = (
            args.beam_search
            and args.beam_width == 1
            and args.beam_expansion_factor == 1
        )
        if not supported:
            print("[WARNING] Draft pipeline prefetch currently supports only beam_width=1 and beam_expansion_factor=1; disabling.")
            args.enable_draft_pipeline_prefetch = False
    
    return args

def prepare_data(data_name, args):
    # Same as original implementation
    examples = load_data(data_name, args.split, args.data_dir)
    
    # Set a fixed seed for reproducibility when using fixed subset
    if args.fixed_subset_size > 0:
        random.seed(args.seed)
        if len(examples) > args.fixed_subset_size:
            # Randomly select a fixed subset of examples
            examples = random.sample(examples, args.fixed_subset_size)
            print(f"Using a fixed subset of {args.fixed_subset_size} examples from {data_name}")
    # Otherwise use num_test_sample if specified
    elif args.num_test_sample > 0:
        examples = examples[: args.num_test_sample]

    # shuffle
    if args.shuffle:
        random.seed(datetime.now().timestamp())
        random.shuffle(examples)

    # select start and end
    examples = examples[args.start : len(examples) if args.end == -1 else args.end]

    # get out_file name
    out_file_prefix = f"{args.split}_{args.prompt_type}_{args.num_test_sample}_seed{args.seed}_t{args.temperature}"
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        output_dir = f"outputs/{output_dir}"
    
    bt_suffix = f"_bt{args.backtrack_threshold}" if args.backtrack_threshold != args.prm_threshold else ""
    out_file = f"{output_dir}/{data_name}/{out_file_prefix}_s{args.start}_e{args.end}_delta{args.prm_threshold}{bt_suffix}_maxsteps{args.max_steps}.jsonl"
    os.makedirs(f"{output_dir}/{data_name}", exist_ok=True)

    # load all processed samples
    processed_samples = []
    if not args.overwrite:
        processed_files = [
            f
            for f in os.listdir(f"{output_dir}/{data_name}/")
            if f.endswith(".jsonl") and f.startswith(out_file_prefix)
        ]
        for f in processed_files:
            processed_samples.extend(
                list(load_jsonl(f"{output_dir}/{data_name}/{f}"))
            )

    # dedepulicate
    processed_samples = {sample["idx"]: sample for sample in processed_samples}
    processed_idxs = list(processed_samples.keys())
    processed_samples = list(processed_samples.values())
    examples = [example for example in examples if example["idx"] not in processed_idxs]
    return examples, processed_samples, out_file

def setup(args):
    # Load model
    openai_api_key = "EMPTY"
    
    # Initialize draft model
    draft_model = None
    draft_client = None
    target_client = None
    prm_client = None
    
    # Assign each model to a specific GPU: draft: cuda:0, target: cuda:3, prm: cuda:2
    
    # Initialize tokenizers
    draft_tokenizer = AutoTokenizer.from_pretrained(args.draft_model_name_or_path, trust_remote_code=True)
    target_tokenizer = AutoTokenizer.from_pretrained(args.target_model_name_or_path, trust_remote_code=True)
    prm_tokenizer = AutoTokenizer.from_pretrained(args.prm_name_or_path, trust_remote_code=True)
    

    
    # Initialize models based on mode
    if args.use_local_models:
        print("Using all local models (PRM and Target in separate processes)")
        import multiprocessing as mp
        import os

        # --- Subprocess worker functions ---
        def prm_worker(conn, model_name_or_path, tokenizer, device):
            import os
            import torch
            import gc
            
            # os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Force PRM to GPU 2
            # 动态获取外部设置的可用显卡，分给 PRM 第 2 张卡
            parent_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2").split(",")
            os.environ["CUDA_VISIBLE_DEVICES"] = parent_gpus[1] if len(parent_gpus) > 1 else parent_gpus[0]
            
            from local_prm_model import LocalPRMModel, LocalPRMEmbeddings
            model = None
            embeddings = None
            try:
                model = LocalPRMModel(model_name_or_path=model_name_or_path, tokenizer=tokenizer, device="cuda:0")  # device arg is local to process
                embeddings = LocalPRMEmbeddings(model)
                while True:
                    try:
                        msg = conn.recv()
                        if msg[0] == "embeddings.create":
                            kwargs = msg[1]
                            result = embeddings.create(**kwargs)
                            conn.send(result)
                        elif msg[0] == "close":
                            break
                    except EOFError:
                        break
                    except Exception as e:
                        print(f"PRM worker error: {e}")
                        break
            finally:
                # Clean up CUDA resources
                try:
                    if embeddings:
                        del embeddings
                    if model and hasattr(model, 'model'):
                        del model.model
                    if model:
                        del model
                    gc.collect()
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(f"PRM cleanup error: {e}")

        def target_worker(conn, model_name_or_path, tokenizer, device):
            import os
            import torch
            import gc
            
            # os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # Force Target to GPU 3
            # 动态获取外部设置的可用显卡，分给 Target 第 3 张卡
            parent_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2").split(",")
            os.environ["CUDA_VISIBLE_DEVICES"] = parent_gpus[2] if len(parent_gpus) > 2 else parent_gpus[0]
            
            from local_target_model import LocalTargetModel
            model = None
            try:
                model = LocalTargetModel(model_name_or_path=model_name_or_path, tokenizer=tokenizer, device="cuda:0")  # device arg is local to process
                while True:
                    try:
                        msg = conn.recv()
                        cmd = msg[0]
                        kwargs = msg[1]
                        if cmd == "completions.create":
                            result = model.completions.create(**kwargs)
                            conn.send(result)
                        elif cmd == "start_async_generation":
                            job_ids = model.start_async_generation(**kwargs)
                            conn.send(job_ids)
                        elif cmd == "collect_async_generation":
                            # returns (outputs, metadata)
                            res = model.collect_async_generation(**kwargs)
                            conn.send(res)
                        elif cmd == "request_async_stop":
                            # returns (outputs, metadata) or intermediate status
                            res = model.request_async_stop(**kwargs)
                            conn.send(res)
                        elif cmd == "stop_async_generation":
                            # blocking stop, keep for completeness
                            res = model.stop_async_generation(**kwargs)
                            conn.send(res)
                        elif cmd == "close":
                            break
                    except EOFError:
                        break
                    except Exception as e:
                        print(f"Target worker error: {e}")
                        break
            finally:
                # Clean up CUDA resources
                try:
                    if model and hasattr(model, 'model'):
                        del model.model
                    if model:
                        del model
                    gc.collect()
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(f"Target cleanup error: {e}")

        # --- Proxy classes for main process ---
        class PRMProxy:
            def __init__(self, conn):
                self.conn = conn
                self.closed = False
                class Embeddings:
                    def __init__(self, conn):
                        self.conn = conn
                    def create(self, **kwargs):
                        self.conn.send(("embeddings.create", kwargs))
                        return self.conn.recv()
                self.embeddings = Embeddings(conn)
            def close(self):
                if not self.closed:
                    try:
                        self.conn.send(("close", None))
                        self.closed = True
                    except:
                        pass

        class TargetProxy:
            def __init__(self, conn):
                self.conn = conn
                self.closed = False
                class Completions:
                    def __init__(self, conn):
                        self.conn = conn
                    def create(self, **kwargs):
                        self.conn.send(("completions.create", kwargs))
                        return self.conn.recv()
                self.completions = Completions(conn)
            # Async prefetch control methods forwarded to the worker process
            def start_async_generation(self, **kwargs):
                self.conn.send(("start_async_generation", kwargs))
                return self.conn.recv()

            def collect_async_generation(self, **kwargs):
                self.conn.send(("collect_async_generation", kwargs))
                return self.conn.recv()

            def request_async_stop(self, **kwargs):
                self.conn.send(("request_async_stop", kwargs))
                return self.conn.recv()

            def stop_async_generation(self, **kwargs):
                self.conn.send(("stop_async_generation", kwargs))
                return self.conn.recv()
            def close(self):
                if not self.closed:
                    try:
                        self.conn.send(("close", None))
                        self.closed = True
                    except:
                        pass

        # --- Start subprocesses ---
        draft_device = "cuda:0"
        target_device = "cuda:3"
        prm_device = "cuda:2"

        # Pipes for communication
        prm_parent_conn, prm_child_conn = mp.Pipe()
        target_parent_conn, target_child_conn = mp.Pipe()

        # Start PRM process
        prm_proc = mp.Process(target=prm_worker, args=(prm_child_conn, args.prm_name_or_path, prm_tokenizer, prm_device))
        prm_proc.start()
        prm_client = PRMProxy(prm_parent_conn)

        # Start Target process
        target_proc = mp.Process(target=target_worker, args=(target_child_conn, args.target_model_name_or_path, target_tokenizer, target_device))
        target_proc.start()
        target_client = TargetProxy(target_parent_conn)

        # Initialize draft model - either hierarchical or standard
        if args.enable_hierarchical_spec:
            from hierarchical_draft_model import HierarchicalDraftModel
            print(f"[Setup] Using hierarchical speculative decoding:")
            print(f"  Token drafter (0.5B): {args.token_drafter_path}")
            print(f"  Step drafter (1.5B): {args.draft_model_name_or_path}")
            print(f"  Num speculative tokens: {args.num_speculative_tokens}")
            draft_model = HierarchicalDraftModel(
                step_drafter_path=args.draft_model_name_or_path,
                token_drafter_path=args.token_drafter_path,
                tokenizer=draft_tokenizer,
                device=draft_device,
                num_speculative_tokens=args.num_speculative_tokens,
            )
        else:
            from local_draft_model import LocalDraftModel
            draft_model = LocalDraftModel(
                model_name_or_path=args.draft_model_name_or_path,
                tokenizer=draft_tokenizer,
                device=draft_device
            )
        # Register cleanup for subprocesses
        def cleanup():
            print("Cleanup: Shutting down processes...")
            try:
                # Send close messages with timeout
                prm_client.close()
                target_client.close()
                
                # Give processes a moment to handle close message
                time.sleep(1)
                
                # Close the parent connections
                try:
                    prm_parent_conn.close()
                    target_parent_conn.close()
                except:
                    pass
                
                # Join with timeout to avoid hanging
                if prm_proc.is_alive():
                    prm_proc.join(timeout=5)
                if target_proc.is_alive():
                    target_proc.join(timeout=5)
                
                # Force terminate any remaining processes
                if prm_proc.is_alive():
                    print("Force terminating PRM process...")
                    prm_proc.terminate()
                    prm_proc.join(timeout=3)
                    if prm_proc.is_alive():
                        prm_proc.kill()  # Nuclear option
                
                if target_proc.is_alive():
                    print("Force terminating target process...")
                    target_proc.terminate()
                    target_proc.join(timeout=3)
                    if target_proc.is_alive():
                        target_proc.kill()  # Nuclear option
                        
                print("Cleanup: Processes shut down successfully.")
            except Exception as e:
                print(f"Cleanup: Error during shutdown: {e}")
        
        # Register signal handler for robust cleanup on Ctrl+C or SIGTERM
        def cleanup_and_exit(signum, frame):
            cleanup()
            sys.exit(0)
        signal.signal(signal.SIGINT, cleanup_and_exit)
        signal.signal(signal.SIGTERM, cleanup_and_exit)
        
        # Also register for normal program exit as backup
        atexit.register(cleanup)
    else:
        print("Using server-based models")
        # Initialize API clients for target model and PRM
        target_client = OpenAI(
            api_key=openai_api_key,
            base_url=args.target_model_ip_address,
        )
        # Remote server-based target client does not support the async prefetch API.
        if args.enable_target_prefetch:
            print("[WARNING] Target prefetch is only supported with --use_local_models; disabling prefetch for server client.")

        prm_client = OpenAI(
            api_key=openai_api_key,
            base_url=args.prm_ip_address,
        )
        
        draft_client = OpenAI(
            api_key=openai_api_key,
            base_url=args.draft_model_ip_address,
        )

    # infer & eval
    data_list = args.data_names.split(",")
    results = []
    overall_stats = {"draft_tokens": 0, "draft_time": 0.0, "target_tokens": 0, "target_time": 0.0, "prm_tokens": 0, "prm_time": 0.0}
    for data_name in data_list:
        # Pass draft_model or draft_client as appropriate
        if args.use_local_models:
            result = main(target_client, prm_client, None, draft_tokenizer, target_tokenizer, prm_tokenizer, draft_model, data_name, args)
        else:
            result = main(target_client, prm_client, draft_client, draft_tokenizer, target_tokenizer, prm_tokenizer, draft_client, data_name, args)
        results.append(result)
        overall_stats["draft_tokens"] += result.get("draft_tokens", 0)
        overall_stats["draft_time"] += result.get("draft_time", 0.0)
        overall_stats["target_tokens"] += result.get("target_tokens", 0)
        overall_stats["target_time"] += result.get("target_time", 0.0)
        overall_stats["prm_tokens"] += result.get("prm_tokens", 0)
        overall_stats["prm_time"] += result.get("prm_time", 0.0)
    
    # add "avg" result to data_list and results
    data_list.append("avg")
    results.append({"acc": sum([result["acc"] for result in results]) / len(results),})

    # print all results
    pad = max([len(data_name) for data_name in data_list])
    print("\t".join(data_name.ljust(pad, " ") for data_name in data_list))
    print("\t".join([f"{result['acc']:.1f}".ljust(pad, " ") for result in results]))

    # Print overall tokens/sec summary
    print("\n=== Overall Model Throughput ===")
    print(f"Draft model: {overall_stats['draft_tokens']} tokens in {overall_stats['draft_time']:.2f}s, {overall_stats['draft_tokens']/overall_stats['draft_time'] if overall_stats['draft_time']>0 else 0:.2f} tokens/sec")
    print(f"Target model: {overall_stats['target_tokens']} tokens in {overall_stats['target_time']:.2f}s, {overall_stats['target_tokens']/overall_stats['target_time'] if overall_stats['target_time']>0 else 0:.2f} tokens/sec")
    print(f"PRM model: {overall_stats['prm_tokens']} tokens in {overall_stats['prm_time']:.2f}s, {overall_stats['prm_tokens']/overall_stats['prm_time'] if overall_stats['prm_time']>0 else 0:.2f} tokens/sec")

    # Clean shutdown for local models
    if args.use_local_models:
        print("Shutting down local model processes...")
        try:
            # Send close messages
            prm_client.close()
            target_client.close()
            
            # Give processes a moment to handle close message
            time.sleep(2)
            
            # Close the parent connections
            try:
                prm_parent_conn.close()
                target_parent_conn.close()
            except:
                pass
            
            # Join with timeout to avoid hanging
            if prm_proc.is_alive():
                prm_proc.join(timeout=10)
            if target_proc.is_alive():
                target_proc.join(timeout=10)
            
            # Force terminate any remaining processes
            if prm_proc.is_alive():
                print("Force terminating PRM process...")
                prm_proc.terminate()
                prm_proc.join(timeout=5)
                if prm_proc.is_alive():
                    print("Killing PRM process...")
                    prm_proc.kill()
            
            if target_proc.is_alive():
                print("Force terminating target process...")
                target_proc.terminate()
                target_proc.join(timeout=5)
                if target_proc.is_alive():
                    print("Killing target process...")
                    target_proc.kill()
                
            print("Local model processes shut down successfully.")
        except Exception as e:
            print(f"Error during shutdown: {e}")
            # Force kill any remaining processes
            try:
                if 'prm_proc' in locals() and prm_proc.is_alive():
                    prm_proc.kill()
                if 'target_proc' in locals() and target_proc.is_alive():
                    target_proc.kill()
            except:
                pass

def is_multi_choice(answer):
    for c in answer:
        if c not in ["A", "B", "C", "D", "E"]:
            return False
    return True

import copy
from typing import List, Tuple, Dict, Any, Optional, Union
import numpy as np

def _init_stats():
    return {
        "draft_time": 0.0, "target_time": 0.0, "prm_time": 0.0,
        "draft_tokens": 0, "target_tokens": 0, "prm_tokens": 0,
        "draft_input_tokens": 0, "draft_output_tokens": 0,
        "target_input_tokens": 0, "target_output_tokens": 0,
    "prm_input_tokens": 0, "prm_output_tokens": 0,
    "draft_batches": [], "target_batches": [], "prm_batches": [],
    # backtracking instrumentation
    "bt_forward_good": 0,                 # times forward draft produced >= threshold
    "target_intervention_attempts": 0,    # times we called target due to no good draft
    "target_intervention_successes": 0,   # times target produced >= threshold
    "bt_anchor_attempts": 0,              # times we attempted backtracking with anchors
    "bt_anchor_count_total": 0,           # total anchors considered across attempts
    "bt_anchor_successes": 0,             # times backtracking produced >= threshold
    "total_backtrack_attempts": 0,         # global counter for total backtracking attempts
    # Prefetch instrumentation
    "prefetch_consume_time": 0.0,
    "prefetch_consume_calls": 0,
    "prefetch_collect_wait_time": 0.0,
    "prefetch_collect_calls": 0,
    "prefetch_stop_time": 0.0,
    "prefetch_stop_calls": 0,
    "prefetch_stop_request_time": 0.0,
    "prefetch_stop_requests": 0,
    "prefetch_start_time": 0.0,
    "prefetch_start_calls": 0,
    # [DraftPipelineChange] Draft pipeline prefetch instrumentation.
    "draft_pipeline_start_calls": 0,
    "draft_pipeline_consume_calls": 0,
    "draft_pipeline_reused_tokens": 0,
    "draft_pipeline_reused_latency": 0.0,
    "draft_pipeline_wasted_tokens": 0,
    "draft_pipeline_wasted_latency": 0.0,
    "draft_pipeline_wasted_actual_latency": 0.0,
    "draft_pipeline_wasted_latency_overhead": 0.0,
    "draft_pipeline_unused_completion_tokens": 0,
    "draft_pipeline_unused_completion_latency": 0.0,

    
    # ... 其他原有字段 ...
    "step_details": [],  
    "step_attempts": [],
    "backtrack_events": [],  # <--- [新增] 专门记录回退轨迹
    "target_trace_events": [],
    # [DraftPipelineChange] Per-event records for reused/wasted/unused next-step draft prefetch.
    "draft_pipeline_events": [],
    }

def _finalize_stats(stats):
    return stats

# [DraftPipeCancelChange] Async job handle for one cancellable next-step draft prefetch.
class DraftPipelineJob:
    """Cancellable draft generation for the single-path draft pipeline experiment."""

    def __init__(self, parent_node, job_id):
        self.parent_node = parent_node
        self.job_id = job_id
        self.created_at = time.time()
        self.finished_at = None
        self.recorded = False

    @property
    def finished(self):
        return self.finished_at is not None

class SearchNode:
    # Memory-optimized tree node - stores only single response per node
    def __init__(self, single_response: str = "", response_type: int = 0, 
                 score: float = 0.0, step: int = 0, prob_idx: int = 0, depth: int = 0, stop_reason=None):
        # For root node, single_response is the prompt; for others, it's the response text
        self.single_response = single_response  # Only this node's text content
        self.response_type = response_type      # 0=prompt, 1=draft, 2=target
        self.stop_reason = stop_reason          # Stop reason for this node's generation
        self.score = score
        self.step = step              # number of responses in this path
        self.prob_idx = prob_idx
        self.depth = depth
        self.log_score = 0.0
        self.parent = None
        self.can_intervene = True
        self.target_job_id = None
        self.target_prefetch_expansion = None
        self.target_partial_cache = []
        self.target_prefetch_metadata = None
        attach_target_trace(self, build_target_trace("none"))
        # [DraftPipelineChange] Optional background next-step draft job attached to this path node.
        self.draft_pipeline_job = None
        # Count of rethink-prompt insertions along this path
        self.rethink_attempts = 0

    def create_child(self, new_response: Tuple[str, int], new_score: float = 0.0,
                     target_intervened: bool = False):
        if len(new_response) == 2:
            response_text, response_type = new_response
            stop_reason = None
        else:
            response_text, response_type, stop_reason = new_response

        child = SearchNode(
            single_response=response_text,
            response_type=response_type,
            stop_reason=stop_reason,
            score=new_score,
            step=self.step + 1,
            prob_idx=self.prob_idx,
            depth=self.depth + 1
        )
        child.parent = self
        child.log_score = self.log_score
        child.rethink_attempts = getattr(self, "rethink_attempts", 0)
        if target_intervened:
            self.can_intervene = False
        child.target_job_id = None
        child.target_prefetch_expansion = None
        child.target_partial_cache = []
        child.target_prefetch_metadata = None
        attach_target_trace(child, build_target_trace("none"))
        # [DraftPipelineChange] Children start with no inherited draft prefetch job unless explicitly attached later.
        child.draft_pipeline_job = None
        return child

    def create_prompt_child(self, prompt_text: str, increment_rethink: bool = False):
        """Create a child node that represents a prompt/instruction injection.

        This does not advance the logical "step" counter, since it's not a model-generated step.
        """
        child = SearchNode(
            single_response=prompt_text,
            response_type=0,
            stop_reason=None,
            score=self.score,
            step=self.step,
            prob_idx=self.prob_idx,
            depth=self.depth + 1,
        )
        child.parent = self
        child.log_score = self.log_score
        child.can_intervene = self.can_intervene
        child.target_job_id = None
        child.target_prefetch_expansion = None
        child.target_partial_cache = []
        child.target_prefetch_metadata = None
        attach_target_trace(child, build_target_trace("none"))
        # [DraftPipelineChange] Prompt-injection nodes should not inherit an in-flight draft prefetch job.
        child.draft_pipeline_job = None
        base_attempts = getattr(self, "rethink_attempts", 0)
        child.rethink_attempts = base_attempts + (1 if increment_rethink else 0)
        return child

    def can_intervene_here(self):
        return self.can_intervene

    def get_full_text(self):
        # Reconstruct full text by traversing up to root
        parts = []
        current = self
        while current is not None:
            parts.append(current.single_response)
            current = current.parent
        parts.reverse()
        return ''.join(parts)
    
    def get_responses_list(self):
        # Get responses in old format for compatibility with existing code
        responses = []
        current = self
        path_nodes = []
        while current is not None and current.parent is not None:
            path_nodes.append(current)
            current = current.parent
        path_nodes.reverse()
        
        for node in path_nodes:
            # Skip prompt/instruction injections (response_type=0) so they don't
            # contaminate PRM scoring or final answer text.
            if node.response_type == 0:
                continue
            responses.append((node.single_response, node.response_type))
        return responses

    def __lt__(self, other):
        # keep "max-heap" semantics if you rely on heapq with inverted order
        return self.score > other.score


def _update_target_stats_from_metadata(stats, meta):
    if not meta:
        return
    if meta.get("status") == "missing":
        return
    generation_time = meta.get("generation_time", 0.0) or 0.0
    prompt_tokens = meta.get("prompt_tokens", 0) or 0
    output_tokens = meta.get("output_tokens", 0) or 0
    stats["target_time"] += generation_time
    stats["target_input_tokens"] += prompt_tokens
    stats["target_output_tokens"] += output_tokens
    stats["target_tokens"] += prompt_tokens + output_tokens
    stats["target_batches"].append(
        (
            meta.get("batch_size", 1),
            output_tokens,
            generation_time,
            meta.get("max_output_tokens", output_tokens),
            prompt_tokens,
        )
    )


def _count_target_tokens(target_tokenizer, text):
    return len(target_tokenizer.encode(text or ""))


def _record_target_trace_event(
    stats,
    target_children,
    target_tokenizer,
    *,
    phase,
    logical_step,
):
    if target_children:
        child = target_children[0]
        fallback_tokens = _count_target_tokens(
            target_tokenizer, child.single_response
        )
        trace = get_target_trace(child, fallback_tokens=fallback_tokens)
    else:
        trace = build_target_trace("none")

    event = {
        "event_idx": len(stats["target_trace_events"]),
        "logical_step": int(logical_step),
        "phase": phase,
        **trace,
    }
    stats["target_trace_events"].append(event)
    print(
        "[TargetTrace] "
        f"step={event['logical_step']} phase={phase} "
        f"mode={trace['target_mode']} total={trace['target_tokens']} "
        f"cached={trace['target_cached_tokens']} "
        f"continuation={trace['target_continuation_tokens']} "
        f"critical={trace['target_critical_gen_tokens']}"
    )
    return trace


def _start_target_prefetch(nodes, target_client, args, expansion_factor, stats=None):
    if not args.enable_target_prefetch:
        return
    call_start = time.time()
    started = False
    schedulable_nodes = []
    prompts = []
    for node in nodes:
        if hasattr(node, "can_intervene_here") and not node.can_intervene_here():
            continue
        if getattr(node, "target_job_id", None):
            continue
        try:
            prompt_text = node.get_full_text()
        except Exception as exc:
            print(f"[Prefetch] Failed to build prompt for node: {exc}")
            continue
        schedulable_nodes.append(node)
        prompts.append(prompt_text)

    if not schedulable_nodes:
        return

    start_fn = getattr(target_client, "start_async_generation", None)
    if not callable(start_fn):
        return

    try:
        job_ids: Optional[Union[List[str], Tuple[str, ...]]] = start_fn(
            prompts=prompts,
            temperature=args.search_temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens_per_call,
            n=expansion_factor,
            stop=[args.step_word] if args.step_word else None,
        )
    except Exception as exc:
        print(f"[Prefetch] Failed to start target generation: {exc}")
        job_ids = None

    if job_ids is None:
        return

    if not isinstance(job_ids, (list, tuple)):
        print(f"[Prefetch] Unexpected job id payload ({type(job_ids)}); skipping prefetch")
        return

    job_ids = list(job_ids)
    if len(job_ids) != len(schedulable_nodes):
        if len(job_ids) < len(schedulable_nodes):
            job_ids.extend([None] * (len(schedulable_nodes) - len(job_ids)))
        else:
            job_ids = job_ids[: len(schedulable_nodes)]

    for node, job_id in zip(schedulable_nodes, job_ids):
        if not job_id:
            continue
        node.target_job_id = job_id
        node.target_prefetch_expansion = expansion_factor
        started = True
    if stats is not None and started:
        stats["prefetch_start_calls"] += 1
        stats["prefetch_start_time"] += time.time() - call_start


def _stop_target_prefetch(nodes, target_client, args, stats, save_partial=True):
    if not args.enable_target_prefetch:
        return
    if not callable(getattr(target_client, "request_async_stop", None)):
        return
    save_partial = bool(save_partial and args.enable_prefetch_cache)
    fn_start = time.time()
    stats["prefetch_stop_calls"] += 1
    for node in nodes:
        job_id = getattr(node, "target_job_id", None)
        if not job_id:
            continue
        stats["prefetch_stop_requests"] += 1
        call_start = time.time()
        try:
            outputs, meta = target_client.request_async_stop(
                job_id=job_id,
                save_partial=save_partial,
            )
        except Exception as exc:
            stats["prefetch_stop_request_time"] += time.time() - call_start
            print(f"[Prefetch] Failed to stop target generation: {exc}")
            node.target_job_id = None
            node.target_prefetch_expansion = None
            continue

        stats["prefetch_stop_request_time"] += time.time() - call_start
        status = meta.get("status")
        if status in {"finished", "cancelled", "error"}:
            node.target_job_id = None
            node.target_prefetch_expansion = None
            node.target_prefetch_metadata = meta
            _update_target_stats_from_metadata(stats, meta)
            if save_partial and outputs:
                node.target_partial_cache.extend(outputs)
            else:
                node.target_partial_cache = []
        elif status == "missing":
            node.target_job_id = None
            node.target_prefetch_expansion = None
            node.target_prefetch_metadata = None
            node.target_partial_cache = []
        else:
            # Pending cancellation; keep job id so we can collect later.
            node.target_prefetch_metadata = meta
    stats["prefetch_stop_time"] += time.time() - fn_start


def _consume_target_prefetch(nodes, target_client, target_tokenizer, args, stats):
    if not args.enable_target_prefetch:
        return [], list(nodes)
    if not callable(getattr(target_client, "collect_async_generation", None)):
        return [], list(nodes)

    fn_start = time.time()
    stats["prefetch_consume_calls"] += 1
    prefetched_children = []
    pending_nodes = []

    collect_fn = getattr(target_client, "collect_async_generation", None)
    stop_suffix = args.step_word or ""

    for node in nodes:
        # Use cached partials first (from previous cancellations)
        if args.enable_prefetch_cache and getattr(node, "target_partial_cache", None):
            resumed_nodes: List[SearchNode] = []
            for cached in node.target_partial_cache:
                text = cached.text or ""
                finished = bool(
                    getattr(cached, "finish_reason", None)
                    or getattr(cached, "stop_reason", None)
                )
                if finished:
                    if stop_suffix and not text.endswith(stop_suffix):
                        text = text + stop_suffix
                    child = node.create_child(
                        (text, 2, getattr(cached, "stop_reason", None)),
                        new_score=args.backtrack_threshold,
                        target_intervened=True,
                    )
                    attach_target_trace(
                        child,
                        build_target_trace(
                            "prefetched_complete",
                            cached_tokens=_count_target_tokens(
                                target_tokenizer, text
                            ),
                        ),
                    )
                    prefetched_children.append(child)
                else:
                    resumed_child = node.create_child(
                        (text, 2, getattr(cached, "stop_reason", None)),
                        new_score=args.backtrack_threshold,
                        target_intervened=True,
                    )
                    attach_target_trace(
                        resumed_child,
                        build_target_trace(
                            "partial_cache",
                            cached_tokens=_count_target_tokens(
                                target_tokenizer, text
                            ),
                        ),
                    )
                    setattr(resumed_child, "resumed_from_prefetch_cache", True)
                    resumed_child.target_job_id = None
                    resumed_child.target_prefetch_expansion = None
                    resumed_child.target_partial_cache = []
                    resumed_nodes.append(resumed_child)
            node.target_partial_cache = []
            if resumed_nodes:
                pending_nodes.extend(resumed_nodes)
        elif not args.enable_prefetch_cache and getattr(node, "target_partial_cache", None):
            # Ensure we do not reuse stale partials when caching is disabled
            node.target_partial_cache = []

        job_id = getattr(node, "target_job_id", None)
        if callable(collect_fn) and job_id:
            try:
                collect_start = time.time()
                stats["prefetch_collect_calls"] += 1
                outputs, meta = collect_fn(
                    job_id=job_id,
                    wait=True,
                    allow_partial=False,
                    remove=True,
                )
            except Exception as exc:
                stats["prefetch_collect_wait_time"] += time.time() - collect_start
                print(f"[Prefetch] Failed to collect target generation: {exc}")
                node.target_job_id = None
                node.target_prefetch_expansion = None
                pending_nodes.append(node)
                continue

            stats["prefetch_collect_wait_time"] += time.time() - collect_start
            status = meta.get("status")
            if status in {"finished", "cancelled", "error"}:
                node.target_job_id = None
                node.target_prefetch_expansion = None
                node.target_prefetch_metadata = meta
                _update_target_stats_from_metadata(stats, meta)

                if outputs:
                    for completion in outputs:
                        text = completion.text or ""
                        finished = bool(
                            getattr(completion, "finish_reason", None)
                            or getattr(completion, "stop_reason", None)
                        )

                        if finished:
                            if stop_suffix and not text.endswith(stop_suffix):
                                text = text + stop_suffix
                            child = node.create_child(
                                (text, 2, getattr(completion, "stop_reason", None)),
                                new_score=args.backtrack_threshold,
                                target_intervened=True,
                            )
                            attach_target_trace(
                                child,
                                build_target_trace(
                                    "prefetched_complete",
                                    cached_tokens=_count_target_tokens(
                                        target_tokenizer, text
                                    ),
                                ),
                            )
                            prefetched_children.append(child)
                        elif args.enable_prefetch_cache:
                            resumed_child = node.create_child(
                                (text, 2, getattr(completion, "stop_reason", None)),
                                new_score=args.backtrack_threshold,
                                target_intervened=True,
                            )
                            attach_target_trace(
                                resumed_child,
                                build_target_trace(
                                    "partial_cache",
                                    cached_tokens=_count_target_tokens(
                                        target_tokenizer, text
                                    ),
                                ),
                            )
                            setattr(resumed_child, "resumed_from_prefetch_cache", True)
                            resumed_child.target_job_id = None
                            resumed_child.target_prefetch_expansion = None
                            resumed_child.target_partial_cache = []
                            pending_nodes.append(resumed_child)

                    node.target_partial_cache = [] if args.enable_prefetch_cache else []
                    continue
            elif status == "missing":
                node.target_job_id = None
                node.target_prefetch_expansion = None
                node.target_prefetch_metadata = None
                node.target_partial_cache = []
                continue

        pending_nodes.append(node)

    stats["prefetch_consume_time"] += time.time() - fn_start
    return prefetched_children, pending_nodes

def _beam_search_with_backtracking(args, target_client, prm_client, draft_tokenizer, target_tokenizer, prm_tokenizer, draft_model, prompt, problem):
    """
    Enhanced beam search with backtracking capability.
    When all candidates fall below PRM threshold, backtrack to previous steps and try target intervention.
    """
    assert isinstance(prompt, str) and isinstance(problem, str), "_beam_search_with_backtracking only supports one problem at a time as a string."
    
    # Check if backtracking is enabled
    enable_backtracking = getattr(args, 'enable_backtracking', False)
    
    best_completion = None
    num_completed = 0
    all_rewards = [[]]
    stats = _init_stats()
    
    # Create root node
    root_beam_node = SearchNode(single_response=prompt, response_type=0, score=0.0, step=0, prob_idx=0)
    root_beam_node.log_score = 0.0
    beam = [root_beam_node]
    
    num_step = 0
    consecutive_backtrack_failures = 0
    
    while beam and num_step < args.max_steps:
        print(f"=== Beam Search Step {num_step} ===")
        print(f"Beam size: {len(beam)}")
        
        # Generate candidates using existing unified search
        beam, candidates = _unified_search_step_with_backtracking(
            beam, args, draft_model, target_client, prm_client,
            draft_tokenizer, target_tokenizer, prm_tokenizer, problem, stats, all_rewards,
            collect_all_candidates=False
        )
        
        # try:
        #     import gc
        #     import torch
        #     # Clear Python garbage
        #     gc.collect()
        #     # Clear CUDA cache
        #     if torch.cuda.is_available():
        #         torch.cuda.empty_cache()
        #         # Force synchronization to ensure cleanup
        #         torch.cuda.synchronize()
        #         if args.debug:
        #             print(f"Memory cleanup after problem {problem_counter}")
        # except Exception as e:
        #     print(f"Memory cleanup warning: {e}")
        
        print(f"Generated {len(candidates)} candidates, kept {len(beam)} in beam")
        
        if args.beam_use_log_scores:
            print(f"PRM log_scores: {[round(c.log_score, 3) for c in candidates[:5]]}")
            print(f"Kept top {len(beam)} candidates (log_scores: {[f'{c.log_score:.3f}' for c in beam[:3]]})")
        else:
            print(f"PRM scores: {[round(c.score, 3) for c in candidates[:5]]}")
            print(f"Kept top {len(beam)} candidates (scores: {[f'{c.score:.3f}' for c in beam[:3]]})")
        
        # Check for completion
        beam, newly_completed = _check_completion_beam(
            beam, args, draft_model, draft_tokenizer, target_tokenizer, stats
        )
        
        # Process completed nodes
        for completed_node in newly_completed:
            responses = completed_node.get_responses_list()
            response_text = ''.join(r[0] for r in responses)
            final_output = response_text[:-len(args.step_word)] if response_text.endswith(args.step_word) else response_text
            draft_count = sum(len(draft_tokenizer.encode(r[0])) for r in responses if r[1] == 1)
            target_count = sum(len(target_tokenizer.encode(r[0])) for r in responses if r[1] == 2)
            token_count = (draft_count, target_count, 0)
            step_info_node = [(i, r[1]) for i, r in enumerate(responses)]
            
            node_score = completed_node.log_score if args.beam_use_log_scores else completed_node.score
            score_type = "log_score" if args.beam_use_log_scores else "score"
            
            current_completion = (final_output, token_count, step_info_node, node_score)
            if best_completion is None or node_score > best_completion[3]:
                best_completion = current_completion
                print(f"New best completion found with {score_type} {node_score:.3f}")
            
            stats["draft_tokens"] += draft_count
            stats["target_tokens"] += target_count
            num_completed += 1
        
        print(f"Completed {len(newly_completed)} nodes this step. Total completed: {num_completed}")
        num_step += 1
        
        # Early termination
        if num_completed >= args.beam_width:
            print(f"Sufficient completions ({num_completed}) reached, terminating search early")
            break
    
    # Return results
    if best_completion:
        best_output, best_token_count, best_step_info, best_score = best_completion
        score_type = "log_score" if args.beam_use_log_scores else "score"
        print(f"Selected best completion with {score_type} {best_score:.3f} from {num_completed} candidates")
        # Backtracking summary
        print("[BT][Summary] forward_good=", stats.get("bt_forward_good", 0),
          ", target_attempts=", stats.get("target_intervention_attempts", 0),
          ", target_successes=", stats.get("target_intervention_successes", 0),
          ", total_backtrack_attempts=", stats.get("total_backtrack_attempts", 0),
          ", anchor_attempts=", stats.get("bt_anchor_attempts", 0),
          ", anchors_considered=", stats.get("bt_anchor_count_total", 0),
          ", anchor_successes=", stats.get("bt_anchor_successes", 0))
        
        outputs = [best_output]
        token_counts = [best_token_count]
        step_info = [best_step_info]
    else:
        outputs = [""]
        token_counts = [(0, 0, 0)]
        step_info = [[]]
    
    return outputs, token_counts, step_info, all_rewards, _finalize_stats(stats)

def _find_backtrack_anchor_nodes(current_beam, max_backtrack_steps=3, max_backtrack_attempts=2):
    """
    For each node in the current beam (at step t), we look backward:
    - first anchor is the node’s GRANDPARENT (step t-2), so we can "fix the parent" (step t-1).
    - If that anchor can't be intervened (already tried s+1), go further up, up to max_backtrack_steps.
    - We never anchor at root’s parent; stop at step >= 0.
    """
    anchors = []
    seen = set()

    for leaf in current_beam:
        steps_back = 1  # we want to fix the parent, so anchor starts at t-2 (one more than just parent)
        cur = leaf
        # move once to parent (t-1), then again to grandparent (t-2) before testing
        if cur.parent: cur = cur.parent
        #if cur.parent: cur = cur.parent
        else:
            # No grandparent (t-2) exists → per policy, skip anchoring for this leaf
            # We do not anchor at parent/root in this mode.
            continue

        while cur and steps_back <= max_backtrack_steps:
            # can we still intervene at next step (cur.step + 1)?
            if cur.can_intervene_here():  # Removed per-node backtrack_count check
                key = (cur.get_full_text(), cur.step)
                if key not in seen:
                    seen.add(key)
                    anchors.append(cur)
                    break  # one anchor per path for this round
            # go one ancestor up
            cur = cur.parent
            steps_back += 1

    return anchors

def _unified_search_step_with_backtracking(
    beam, args, draft_model, target_client, prm_client,
    draft_tokenizer, target_tokenizer, prm_tokenizer, problem, stats, all_rewards,
    collect_all_candidates=False
):
    # Print depth information for the current beam
    beam_depths = [node.depth for node in beam]
    print(f"[BT][Depth] Current beam depths: {beam_depths} (min: {min(beam_depths)}, max: {max(beam_depths)}, avg: {sum(beam_depths)/len(beam_depths):.1f})")

    # Determine expansion strategy: always use expansion factor for beam expansion
    use_expansion_factor = True
    expansion_factor = args.beam_expansion_factor if use_expansion_factor else 1

    # Start parallel target generation for current beam if supported
    _start_target_prefetch(beam, target_client, args, expansion_factor, stats)

    # [DraftPipelineChange] Try to reuse a next-step draft prefetched during the previous PRM window.
    # 1) Expand with DRAFT from the current beam (normal forward step).
    # In the draft-pipeline experiment, the previous accepted node may already
    # have generated this step while PRM was scoring the prior step.
    draft_children = []
    if _draft_pipeline_enabled(args) and len(beam) == 1:
        prefetched_children, prefetch_event = _consume_draft_pipeline_prefetch(
            beam[0], draft_model, draft_tokenizer, args, stats, reason="reuse_next_step"
        )
        if prefetch_event:
            if prefetched_children:
                draft_children = prefetched_children
                _record_draft_pipeline_event(stats, prefetch_event, "reused")
                print(
                    f"[DraftPipeline] Reused prefetched draft step from parent step "
                    f"{prefetch_event.get('parent_step')} "
                    f"({prefetch_event.get('prefetch_tokens', 0)} tokens, "
                    f"{prefetch_event.get('draft_prefetch_time', 0.0):.3f}s)"
                )
            else:
                _record_draft_pipeline_event(stats, prefetch_event, "reused_empty")
                print("[DraftPipeline] Prefetch produced no reusable children; falling back to synchronous draft")

    if not draft_children:
        draft_children = _expand_beam_parallel_with_backtracking(beam, args, draft_model, draft_tokenizer, stats, use_expansion_factor)

    # [新增] 计算这一轮 Draft 产生的 Token 数（即便它待会儿被 Reject）
    # 假设 beam_width=1, 我们取第一个候选
    draft_token_count = 0
    if draft_children:
        # 调用 tokenizer 计算这个被拒候选的长度
        draft_token_count = len(draft_tokenizer.encode(draft_children[0].single_response))

    # [DraftPipelineChange] Start next-step draft generation while PRM validates the current draft.
    # If PRM accepts, the next loop can reuse it; if PRM rejects, it is counted
    # as the second-class "wasted" prefetch work requested for this experiment.
    if _draft_pipeline_enabled(args) and draft_children:
        _start_draft_pipeline_prefetch(draft_children[0], args, draft_model, draft_tokenizer, stats)

    # 2) Score with PRM
    use_log = collect_all_candidates or (not collect_all_candidates and args.beam_use_log_scores)
    if use_log:
        draft_children = _evaluate_with_prm_log_scores_backtracking(
            draft_children, [problem], prm_client, prm_tokenizer, args, stats, all_rewards
        )
    else:
        draft_children = _evaluate_with_prm_backtracking(
            draft_children, [problem], prm_client, prm_tokenizer, args, stats, all_rewards
        )
    
    # Split by threshold
    good = [c for c in draft_children if c.score >= args.backtrack_threshold]
    if good:
        # --- 情况：Accept ---
        stats["step_attempts"].append({
            "step": len(stats["step_attempts"]),
            "draft_tokens": draft_token_count,
            **build_target_trace("none"),
            "outcome": "accept",
        })

        # 情况 A: Draft 被接受
        stats["bt_forward_good"] += 1
        # 记录状态
        stats["step_details"].append({
            "step": len(stats["step_details"]),
            "status": "accept",
            "model": "draft",
            "score": float(good[0].score)
        })


        good_depths = [c.depth for c in good]
        print(f"[BT] Forward draft produced {len(good)} >= threshold candidates "
              f"(top score: {max((g.log_score if args.beam_use_log_scores else g.score) for g in good):.3f}, depths: {good_depths})")

        # --- ORIGINAL BEAM SELECTION (unchanged) ---
        if use_log:
            good.sort(key=lambda x: x.log_score, reverse=True)
        else:
            good.sort(key=lambda x: x.score, reverse=True)
        next_beam = good[: args.beam_width]
        _stop_target_prefetch(beam, target_client, args, stats, save_partial=args.enable_prefetch_cache)
        return next_beam, good

    # [DraftPipelineChange] Current draft was rejected; the prefetched next draft step is wasted.
    # 3) No good candidates → the prefetched next draft step is now wasted,
    # because the current draft step was rejected by PRM.
    rejected_prm_time = getattr(draft_children[0], "last_prm_time", None) if draft_children else None
    rejected_prm_score = draft_children[0].score if draft_children else None
    _stop_draft_pipeline_prefetch_for_nodes(
        draft_children[:1],
        draft_model,
        draft_tokenizer,
        args,
        stats,
        "wasted_reject",
        prm_time=rejected_prm_time,
        prm_score=rejected_prm_score,
    )

    # TARGET INTERVENTION on the ORIGINAL BEAM (not on "bad children")
    stats["target_intervention_attempts"] += 1

    stats["step_details"].append({
        "step": len(stats["step_details"]),
        "status": "reject",
        "model": "draft",
        "score": float(draft_children[0].score) if draft_children else 0.0
    })

    print(f"[BT] No good draft candidates; attempting TARGET intervention on {len(beam)} nodes")
    target_children = _target_intervene_from_nodes(
        beam, target_client, target_tokenizer, args, stats
    )
    step_index = len(stats["step_attempts"])
    target_trace = _record_target_trace_event(
        stats,
        target_children,
        target_tokenizer,
        phase="target_intervention",
        logical_step=step_index,
    )
    
    stats["step_attempts"].append({
        "step": step_index,
        "draft_tokens": draft_token_count,   # 这里就是你想看的：被 Reject 的 Draft Token 数
        **target_trace,
        "outcome": "target_intervention"
    })

    skip_prm_for_target = (
        args.beam_expansion_factor == 1
        and not getattr(args, "enable_backtracking", False)
        and not collect_all_candidates
    )

    # Score target children
    if not skip_prm_for_target:
        if use_log:
            target_children = _evaluate_with_prm_log_scores_backtracking(
                target_children, [problem], prm_client, prm_tokenizer, args, stats, all_rewards
            )
        else:
            target_children = _evaluate_with_prm_backtracking(
                target_children, [problem], prm_client, prm_tokenizer, args, stats, all_rewards
            )

    good = [c for c in target_children if c.score >= args.backtrack_threshold]
    if good:
        stats["target_intervention_successes"] += 1
        good_depths = [c.depth for c in good]
        print(f"[BT] Target intervention produced {len(good)} >= threshold candidates (top score: {max((g.log_score if args.beam_use_log_scores else g.score) for g in good):.3f}, depths: {good_depths})")
        if use_log:
            good.sort(key=lambda x: x.log_score, reverse=True)
        else:
            good.sort(key=lambda x: x.score, reverse=True)
        next_beam = good[: args.beam_width]
        _stop_target_prefetch(beam, target_client, args, stats, save_partial=args.enable_prefetch_cache)
        return next_beam, good

    # 4) Still none ≥ threshold → BACKTRACK (if enabled)
    enable_backtracking = getattr(args, 'enable_backtracking', False)
    max_backtrack_attempts = getattr(args, 'max_backtrack_attempts', 2)

    should_backtrack = True
    if not enable_backtracking:
        print("[BT] Backtracking disabled")
        should_backtrack = False
    elif stats["total_backtrack_attempts"] >= max_backtrack_attempts:
        print(f"[BT] Maximum backtracking attempts ({max_backtrack_attempts}) reached; no more backtracking allowed")
        should_backtrack = False
    else:
        # any_leaf_has_grandparent = any((n.parent is not None and n.parent.parent is not None) for n in beam)
        # if not any_leaf_has_grandparent:
        #     print("[BT] No grandparent available for any leaf")
        #     should_backtrack = False
        any_leaf_has_parent = any((n.parent is not None) for n in beam)
        if not any_leaf_has_parent:
            print("[BT] No parent available for any leaf")
            should_backtrack = False
        else:
            back_nodes = _find_backtrack_anchor_nodes(beam, max_backtrack_steps=getattr(args, 'max_backtrack_steps', 3),
            max_backtrack_attempts=max_backtrack_attempts  # Pass for compatibility but won't use for per-node filtering
            )
            if not back_nodes:
                print("[BT] No backtrack nodes available")
                should_backtrack = False
            else:
                # --- [核心修改点] 记录回退动作 ---
                anchor = back_nodes[0]  # 系统选择的回归点
                current_step = beam[0].step # 当前失败的步骤
                
                # 记录到 step_attempts 保证顺序流
                stats["step_attempts"].append({
                    "step": current_step,
                    "outcome": "backtrack_trigger",
                    "from_step": current_step,
                    "to_step": anchor.step,
                    "draft_tokens": 0,
                    **build_target_trace("none"),
                })
                
                # 记录到专项列表方便后续快速计算浪费量
                stats["backtrack_events"].append({
                    "from_step": current_step,
                    "to_step": anchor.step,
                    "reason": "all_candidates_below_threshold"
                })
                # -------------------------------

    # Alternative to classic backtracking: keep the low-PRM intervened target step in-context,
    # append a corrective instruction, then ask the TARGET model to redo the step.
    # If rethink fails to produce good candidates, fall through to classic backtracking.
    if should_backtrack and getattr(args, "enable_rethink_prompt", False) and target_children:
        max_rethink = int(getattr(args, "max_rethink_attempts", 1) or 0)
        if use_log:
            target_children.sort(key=lambda x: x.log_score, reverse=True)
        else:
            target_children.sort(key=lambda x: x.score, reverse=True)

        base = target_children[: args.beam_width]
        prompt_nodes = []
        applied = 0
        for child in base:
            attempts = int(getattr(child, "rethink_attempts", 0) or 0)
            if max_rethink > 0 and attempts < max_rethink and hasattr(child, "create_prompt_child"):
                prompt_nodes.append(child.create_prompt_child(getattr(args, "rethink_prompt_text", ""), increment_rethink=True))
                applied += 1

        if applied > 0 and prompt_nodes:
            print(f"[BT][Rethink] Applied rethink prompt to {applied}/{len(base)} target-intervened candidates; asking target to redo step")
            rethink_children = _target_intervene_from_nodes(
                prompt_nodes,
                target_client,
                target_tokenizer,
                args,
                stats,
                override_expansion_factor=getattr(args, "backtrack_expansion_factor", None) or args.beam_expansion_factor,
            )
            _record_target_trace_event(
                stats,
                rethink_children,
                target_tokenizer,
                phase="rethink_target_intervention",
                logical_step=prompt_nodes[0].step,
            )

            if use_log:
                rethink_children = _evaluate_with_prm_log_scores_backtracking(
                    rethink_children, [problem], prm_client, prm_tokenizer, args, stats, all_rewards
                )
            else:
                rethink_children = _evaluate_with_prm_backtracking(
                    rethink_children, [problem], prm_client, prm_tokenizer, args, stats, all_rewards
                )

            # Prefer rethink outputs if they are good; otherwise keep them as the "best available" target_children
            # but continue into classic backtracking.
            if rethink_children:
                target_children = rethink_children

            good = [c for c in rethink_children if c.score >= args.backtrack_threshold]
            if good:
                good_depths = [c.depth for c in good]
                print(
                    f"[BT][Rethink] Target rethink produced {len(good)} >= threshold candidates "
                    f"(top score: {max((g.log_score if args.beam_use_log_scores else g.score) for g in good):.3f}, depths: {good_depths})"
                )
                if use_log:
                    good.sort(key=lambda x: x.log_score, reverse=True)
                else:
                    good.sort(key=lambda x: x.score, reverse=True)
                next_beam = good[: args.beam_width]
                _stop_target_prefetch(beam, target_client, args, stats, save_partial=args.enable_prefetch_cache)
                return next_beam, good

    if not should_backtrack:
        print("[BT] Continuing with best target_children as beam")
        if use_log:
            target_children.sort(key=lambda x: x.log_score, reverse=True)
            next_beam = target_children[: args.beam_width]
            top_metric = next_beam[0].log_score if next_beam else float('-inf')
        else:
            target_children.sort(key=lambda x: x.score, reverse=True)
            next_beam = target_children[: args.beam_width]
            top_metric = next_beam[0].score if next_beam else float('-inf')
        print(f"[BT] Using {len(next_beam)} target child(ren) as beam (top={'log_score' if use_log else 'score'} {top_metric:.3f})")
        _stop_target_prefetch(beam, target_client, args, stats, save_partial=args.enable_prefetch_cache)
        return next_beam, target_children

    stats["bt_anchor_attempts"] += 1
    stats["bt_anchor_count_total"] += len(back_nodes)
    stats["total_backtrack_attempts"] += 1  # Increment global backtracking attempts counter
    print(f"[BT] Backtracking attempt {stats['total_backtrack_attempts']}/{max_backtrack_attempts} with {len(back_nodes)} anchor node(s)")

    # Intervene at the anchor nodes (grandparents etc.)
    # Debug: summarize anchors
    try:
        anchor_summ = [(n.step, n.depth, n.can_intervene_here()) for n in back_nodes[:5]]
        print(f"[BT][Anchors] details (step, depth, can_intervene) sample: {anchor_summ}")
    except Exception:
        pass
    back_target_children = _target_intervene_from_nodes(
        back_nodes, target_client, target_tokenizer, args, stats
    )
    _record_target_trace_event(
        stats,
        back_target_children,
        target_tokenizer,
        phase="backtrack_target_intervention",
        logical_step=back_nodes[0].step,
    )
    print(f"[BT] Target generated {len(back_target_children)} backtracking child(ren) from {len(back_nodes)} anchor(s)")
    if not back_target_children:
        print("[BT][Warn] No back_target_children produced; possible causes: target error, all anchors rejected by can_intervene_here, or n=0")

    if use_log:
        back_target_children = _evaluate_with_prm_log_scores_backtracking(
            back_target_children, [problem], prm_client, prm_tokenizer, args, stats, all_rewards
        )
    else:
        back_target_children = _evaluate_with_prm_backtracking(
            back_target_children, [problem], prm_client, prm_tokenizer, args, stats, all_rewards
        )
    # Debug PRM results
    if back_target_children:
        scores = [ (c.log_score if args.beam_use_log_scores else c.score) for c in back_target_children ]
        try:
            top_k = sorted(scores, reverse=True)[:5]
            print(f"[BT][PRM] evaluated {len(scores)} backtracking children; top scores: {[round(s,3) for s in top_k]} (threshold={args.backtrack_threshold})")
        except Exception:
            pass

    good = [c for c in back_target_children if c.score >= args.backtrack_threshold]
    if good:
        stats["bt_anchor_successes"] += 1
        good_depths = [c.depth for c in good]
        print(f"[BT] Backtracking intervention produced {len(good)} >= threshold candidates (top score: {max((g.log_score if args.beam_use_log_scores else g.score) for g in good):.3f}, depths: {good_depths})")
        if use_log:
            good.sort(key=lambda x: x.log_score, reverse=True)
        else:
            good.sort(key=lambda x: x.score, reverse=True)
        next_beam = good[: args.beam_width]
        _stop_target_prefetch(beam, target_client, args, stats, save_partial=args.enable_prefetch_cache)
        return next_beam, good

    print("[BT] No luck even after backtracking: continue with target")
    if use_log:
        target_children.sort(key=lambda x: x.log_score, reverse=True)
        next_beam = target_children[: args.beam_width]
        top_metric = next_beam[0].log_score if next_beam else float('-inf')
    else:
        target_children.sort(key=lambda x: x.score, reverse=True)
        next_beam = target_children[: args.beam_width]
        top_metric = next_beam[0].score if next_beam else float('-inf')
    _stop_target_prefetch(beam, target_client, args, stats, save_partial=args.enable_prefetch_cache)
    return next_beam, target_children

#def _expand_beam_parallel_with_backtracking(beam, args, draft_model, draft_tokenizer, use_expansion_factor=True):
def _expand_beam_parallel_with_backtracking(beam, args, draft_model, draft_tokenizer, stats, use_expansion_factor=True):
    """Generate draft candidates with proper parent-child linkage for backtracking.
    
    Args:
        beam: List of nodes to expand
        args: Arguments containing beam_expansion_factor and other settings
        draft_model: Model to use for generation
        draft_tokenizer: Tokenizer for the draft model
        use_expansion_factor: If True, generate beam_expansion_factor candidates per node.
                            If False, generate only 1 candidate per node.
    """
    if not beam:
        return []
    
    node_prompts = [node.get_full_text() for node in beam]
    
    # Determine how many candidates to generate per node
    expansion_factor = args.beam_expansion_factor if use_expansion_factor else 1
    
    try:
        # [修改点 1] 记录 Draft 生成的开始时间
        start_time = time.time()
        
        response = draft_model.completions.create(
            model=args.draft_model_name_or_path.split("/")[-1],
            prompt=node_prompts,
            temperature=args.search_temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens_per_call,
            stop=[args.step_word],
            n=expansion_factor,
        )
        
        # [修改点 2] 计算耗时
        generation_time = time.time() - start_time
        draft_responses = response.choices
        
        # [修改点 3] 统计 Input 和 Output Tokens，用于计算真实的吞吐量 (tokens/sec)
        batch_input_tokens = sum(len(draft_tokenizer.encode(p)) for p in node_prompts)
        outputs_tokens = sum(len(draft_tokenizer.encode((resp.text or "") + args.step_word)) for resp in draft_responses)
        max_tokens_in_batch = max((len(draft_tokenizer.encode((resp.text or "") + args.step_word)) for resp in draft_responses), default=0)
        
        # [修改点 4] 把数据累加到全局的 stats 字典中
        stats["draft_time"] += generation_time
        stats["draft_input_tokens"] += batch_input_tokens
        stats["draft_output_tokens"] += outputs_tokens
        stats["draft_batches"].append(
            (len(node_prompts), outputs_tokens, generation_time, max_tokens_in_batch, batch_input_tokens)
        )
        
        draft_candidates = []
        for node_idx, parent_node in enumerate(beam):
            node_responses = draft_responses[node_idx * expansion_factor : (node_idx + 1) * expansion_factor]
            
            for draft_resp in node_responses:
                response_text = draft_resp.text + args.step_word
                
                # Use create_child if available (backtracking mode), otherwise create manually (compatibility mode)
                if hasattr(parent_node, 'create_child'):
                    new_response = (response_text, 1, draft_resp.stop_reason)
                    child_node = parent_node.create_child(new_response, target_intervened=False)
                else:
                    # Backward compatibility: create node manually
                    new_response = (response_text, 1)  # Original 2-tuple format
                    child_node = SearchNode(
                        single_response=response_text, response_type=1, stop_reason=draft_resp.stop_reason, score=0.0,
                        step=parent_node.step + 1, prob_idx=parent_node.prob_idx, depth=parent_node.depth + 1
                    )
                    child_node.parent = parent_node
                    child_node.log_score = parent_node.log_score
                
                draft_candidates.append(child_node)
        
        return draft_candidates
        
    except Exception as e:
        print(f"Error in parallel beam expansion with backtracking: {e}")
        return []

# [DraftPipelineChange] Helper group for the single-path draft pipeline prefetch experiment.
def _draft_pipeline_enabled(args):
    return (
        getattr(args, "enable_draft_pipeline_prefetch", False)
        and getattr(args, "beam_width", None) == 1
        and getattr(args, "beam_expansion_factor", None) == 1
    )

def _update_draft_stats_from_pipeline_metadata(stats, meta, job=None):
    """Count async draft-pipeline work once in the draft model totals."""
    if not meta or meta.get("status") == "missing":
        return
    if job is not None and getattr(job, "recorded", False):
        return

    generation_time = meta.get("generation_time", 0.0) or 0.0
    prompt_tokens = meta.get("prompt_tokens", 0) or 0
    output_tokens = meta.get("output_tokens", 0) or 0
    max_output_tokens = meta.get("max_output_tokens", output_tokens) or 0
    batch_size = meta.get("batch_size", 1) or 1

    stats["draft_time"] += generation_time
    stats["draft_input_tokens"] += prompt_tokens
    stats["draft_output_tokens"] += output_tokens
    stats["draft_batches"].append(
        (batch_size, output_tokens, generation_time, max_output_tokens, prompt_tokens)
    )
    if job is not None:
        job.recorded = True

def _draft_pipeline_tokens_from_outputs(outputs, meta, draft_tokenizer, args):
    if meta and meta.get("output_tokens") is not None:
        return meta.get("output_tokens", 0) or 0
    total = 0
    for output in outputs or []:
        token_ids = getattr(output, "token_ids", None)
        if token_ids is not None:
            total += len(token_ids)
            continue
        text = getattr(output, "text", "") or ""
        if text and args.step_word and not text.endswith(args.step_word):
            text += args.step_word
        total += len(draft_tokenizer.encode(text)) if text else 0
    return total

def _create_draft_pipeline_children(parent_node, outputs, args):
    children = []
    stop_suffix = args.step_word or ""
    for output in sorted(outputs or [], key=lambda item: getattr(item, "index", 0)):
        text = getattr(output, "text", "") or ""
        if not text:
            continue
        if stop_suffix and not text.endswith(stop_suffix):
            text += stop_suffix
        child = parent_node.create_child(
            (text, 1, getattr(output, "stop_reason", None)),
            target_intervened=False,
        )
        children.append(child)
    return children

def _build_draft_pipeline_event(node, job, reason, outputs, meta, draft_tokenizer, args):
    meta = meta or {}
    error = meta.get("error")
    return {
        "parent_step": getattr(node, "step", None),
        "reason": reason,
        "job_id": getattr(job, "job_id", None),
        "status": meta.get("status"),
        "prefetch_tokens": _draft_pipeline_tokens_from_outputs(outputs, meta, draft_tokenizer, args),
        "draft_prefetch_time": meta.get("generation_time", 0.0) or 0.0,
        "prompt_tokens": meta.get("prompt_tokens", 0) or 0,
        "max_output_tokens": meta.get("max_output_tokens", 0) or 0,
        "num_outputs": meta.get("num_outputs", len(outputs or [])) or 0,
        "error": repr(error) if error else None,
    }

def _start_draft_pipeline_prefetch(parent_node, args, draft_model, draft_tokenizer, stats):
    """Start one cancellable next-step draft generation from an accepted node."""
    if not _draft_pipeline_enabled(args) or parent_node is None:
        return None
    if getattr(parent_node, "draft_pipeline_job", None) is not None:
        return parent_node.draft_pipeline_job

    start_fn = getattr(draft_model, "start_async_generation", None)
    if not callable(start_fn):
        print("[DraftPipeline] Draft model does not support cancellable async generation; skipping prefetch")
        return None

    try:
        job_ids = start_fn(
            prompts=parent_node.get_full_text(),
            temperature=args.search_temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens_per_call,
            n=1,
            stop=[args.step_word] if args.step_word else None,
        )
    except Exception as exc:
        print(f"[DraftPipeline] Failed to start cancellable draft prefetch: {exc}")
        return None

    if not isinstance(job_ids, (list, tuple)) or not job_ids:
        print(f"[DraftPipeline] Unexpected draft prefetch job id payload: {type(job_ids)}")
        return None

    job = DraftPipelineJob(parent_node, job_ids[0])
    parent_node.draft_pipeline_job = job
    stats["draft_pipeline_start_calls"] += 1
    return job

def _finish_draft_pipeline_prefetch(node, draft_model, draft_tokenizer, args, stats, reason, cancel):
    """Collect or cancel a draft-pipeline prefetch job and detach it from the node."""
    job = getattr(node, "draft_pipeline_job", None) if node is not None else None
    if job is None:
        return [], None

    stats["draft_pipeline_consume_calls"] += 1
    call_start = time.time()
    try:
        if cancel:
            # [DraftPipeCancelChange] Reject path: abort immediately and keep
            # only the partial tokens generated before PRM made the decision.
            outputs, meta = draft_model.stop_async_generation(
                job_id=job.job_id,
                save_partial=True,
            )
        else:
            outputs, meta = draft_model.collect_async_generation(
                job_id=job.job_id,
                wait=True,
                allow_partial=False,
                remove=True,
            )
    except Exception as exc:
        outputs, meta = [], {"status": "error", "error": exc}
    call_time = time.time() - call_start
    job.finished_at = time.time()
    if node is not None:
        node.draft_pipeline_job = None

    _update_draft_stats_from_pipeline_metadata(stats, meta, job=job)
    children = [] if cancel else _create_draft_pipeline_children(node, outputs, args)
    event = _build_draft_pipeline_event(
        node, job, reason, outputs, meta, draft_tokenizer, args
    )
    event["control_call_time"] = call_time
    event["cancel_requested"] = bool(cancel)
    return children, event

def _consume_draft_pipeline_prefetch(node, draft_model, draft_tokenizer, args, stats, reason):
    """Wait for and detach a reusable draft pipeline prefetch job from a node."""
    return _finish_draft_pipeline_prefetch(
        node, draft_model, draft_tokenizer, args, stats, reason, cancel=False
    )

def _record_draft_pipeline_event(stats, event, outcome, prm_time=None, prm_score=None):
    if not event:
        return
    event = dict(event)
    event["outcome"] = outcome
    event["prm_time"] = prm_time
    event["prm_score"] = float(prm_score) if prm_score is not None else None
    tokens = event.get("prefetch_tokens", 0) or 0
    draft_time = event.get("draft_prefetch_time", 0.0) or 0.0
    if outcome == "reused":
        event["wasted_tokens"] = 0
        event["wasted_latency"] = 0.0
        stats["draft_pipeline_reused_tokens"] += tokens
        stats["draft_pipeline_reused_latency"] += draft_time
    elif outcome == "wasted_reject":
        # [DraftPipeCancelChange] Reject path records only partial tokens that
        # actually existed when the next-step draft was aborted.
        event["wasted_tokens"] = tokens
        event["wasted_latency"] = prm_time if prm_time is not None else draft_time
        event["wasted_latency_overhead"] = max(draft_time - event["wasted_latency"], 0.0)
        stats["draft_pipeline_wasted_tokens"] += tokens
        stats["draft_pipeline_wasted_latency"] += event["wasted_latency"]
        stats["draft_pipeline_wasted_actual_latency"] += draft_time
        stats["draft_pipeline_wasted_latency_overhead"] += event["wasted_latency_overhead"]
    elif outcome == "unused_completion":
        event["wasted_tokens"] = 0
        event["wasted_latency"] = 0.0
        stats["draft_pipeline_unused_completion_tokens"] += tokens
        stats["draft_pipeline_unused_completion_latency"] += draft_time
    stats["draft_pipeline_events"].append(event)

def _stop_draft_pipeline_prefetch_for_nodes(
    nodes, draft_model, draft_tokenizer, args, stats, outcome, prm_time=None, prm_score=None
):
    for node in nodes or []:
        _, event = _finish_draft_pipeline_prefetch(
            node, draft_model, draft_tokenizer, args, stats, reason=outcome, cancel=True
        )
        _record_draft_pipeline_event(stats, event, outcome, prm_time=prm_time, prm_score=prm_score)

def _target_intervene_from_nodes(nodes, target_client, target_tokenizer, args, stats, override_expansion_factor=None):
    """
    For each anchor node (step s), generate target continuations at step s+1.
    Mark intervention history at (s+1). Do NOT shift to parent prompts.
    
    Args:
        nodes: List of nodes to intervene from
        override_expansion_factor: If provided, use this expansion factor instead of the default logic
    """
    if not nodes:
        return []

    prefetched_children, pending_nodes = _consume_target_prefetch(
        nodes, target_client, target_tokenizer, args, stats
    )

    # Debug: show what consume returned so we can see why we may return early
    try:
        if getattr(args, 'debug', False):
            print(
                f"[Prefetch Debug] prefetched_children={len(prefetched_children)}, "
                f"pending_nodes={len(pending_nodes)}, prefetch_enabled={args.enable_target_prefetch}, "
                f"cache_enabled={args.enable_prefetch_cache}"
            )
    except Exception:
        pass

    # group by exact anchor prompt to avoid duplicate target calls for remaining nodes
    prompt_to_nodes = {}
    for node in pending_nodes:
        if not node.can_intervene_here():
            continue
        key = node.get_full_text()
        prompt_to_nodes.setdefault(key, []).append(node)

    if not prompt_to_nodes:
        try:
            if getattr(args, 'debug', False):
                print(
                    f"[Prefetch Debug] prompt_to_nodes empty (no intervenable pending nodes); "
                    f"returning {len(prefetched_children)} prefetched child(ren)"
                )
        except Exception:
            pass
        return prefetched_children

    prompts = list(prompt_to_nodes.keys())

    # Determine expansion factor
    if override_expansion_factor is not None:
        expansion_factor = override_expansion_factor
    else:
        expansion_factor = args.backtrack_expansion_factor
    if expansion_factor is None:
        expansion_factor = args.beam_expansion_factor
    if expansion_factor < 1:
        expansion_factor = 1

    stop_tokens = [args.step_word] if args.step_word else None

    try:
        if getattr(args, 'debug', False):
            print(f"[Target Debug] calling synchronous completions.create for {len(prompts)} prompt(s) with expansion={expansion_factor}")
    except Exception:
        pass

    start = time.time()
    resp = target_client.completions.create(
        model=args.target_model_name_or_path.split("/")[-1],
        prompt=prompts,
        temperature=args.search_temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens_per_call,
        n=expansion_factor,
        stop=stop_tokens,
    )
    generation_time = time.time() - start
    # Quick console log so per-intervention generation time is visible in run output
    try:
        print(f"[Target] intervention: prompts={len(prompts)}, expansion={expansion_factor}, time={generation_time:.3f}s")
    except Exception as e:
        print("Logging failed:", e)
    choices = resp.choices

    batch_input_tokens = sum(len(target_tokenizer.encode(p)) for p in prompts)
    stats["target_time"] += generation_time
    stats["target_input_tokens"] += batch_input_tokens
    outputs_tokens = 0

    out = []
    for i, prompt in enumerate(prompts):
        anchors = prompt_to_nodes[prompt]
        start_i = i * expansion_factor
        end_i = start_i + expansion_factor
        prompt_choices = choices[start_i:end_i]
        for ch in prompt_choices:
            generated_text = ch.text or ""
            if args.step_word and not generated_text.endswith(args.step_word):
                generated_text = generated_text + args.step_word
            encoded_length = len(target_tokenizer.encode(generated_text))
            outputs_tokens += encoded_length
            for anchor in anchors:
                child = anchor.create_child(
                    (generated_text, 2, getattr(ch, "stop_reason", None)),
                    new_score=args.backtrack_threshold,
                    target_intervened=True,
                )
                anchor_trace = get_target_trace(anchor)
                if anchor_trace["target_mode"] == "partial_cache":
                    child_trace = build_target_trace(
                        "partial_resume",
                        cached_tokens=anchor_trace["target_cached_tokens"],
                        continuation_tokens=encoded_length,
                    )
                else:
                    child_trace = build_target_trace(
                        "direct_generation",
                        continuation_tokens=encoded_length,
                    )
                attach_target_trace(child, child_trace)
                out.append(child)

    stats["target_output_tokens"] += outputs_tokens
    stats["target_tokens"] += batch_input_tokens + outputs_tokens
    max_tokens_in_batch = max(
        (len(target_tokenizer.encode((choice.text or "") + (args.step_word or ""))) for choice in choices),
        default=0,
    )
    stats["target_batches"].append(
        (len(prompts), outputs_tokens, generation_time, max_tokens_in_batch, batch_input_tokens)
    )

    return prefetched_children + out

def _evaluate_with_prm_backtracking(candidates, problems, prm_client, prm_tokenizer, args, stats, all_rewards):
    """Evaluate candidates with PRM, maintaining backtracking structure."""
    if not candidates:
        return candidates
    
    evaluation_data = []
    for candidate in candidates:
        responses = candidate.get_responses_list()
        full_text = ''.join(r[0] for r in responses)
        # Strip trailing step_word if present (PRM should see incomplete step)
        if full_text.endswith(args.step_word):
            full_text = full_text[:-len(args.step_word)]
        evaluation_data.append((problems[candidate.prob_idx], full_text))
    
    processed_data = [
        prepare_input(problem, full_resp, tokenizer=prm_tokenizer, step_token=args.step_word, model_name=args.prm_name_or_path)
        for problem, full_resp in evaluation_data
    ]
    input_ids, steps, reward_flags = zip(*processed_data)
    
    start = time.time()
    rewards = prm_client.embeddings.create(
        input=input_ids,
        model=args.prm_name_or_path.split("/")[-1],
    )
    evaluation_time = time.time() - start
    
    stats["prm_time"] += evaluation_time
    input_tokens = sum(len(input_id) for input_id in input_ids)
    stats["prm_input_tokens"] += input_tokens
    stats["prm_tokens"] += input_tokens
    
    step_rewards = derive_step_rewards_vllm(rewards, reward_flags, args.prm_name_or_path)
    output_tokens = sum(len(reward_sequence) for reward_sequence in step_rewards)
    stats["prm_output_tokens"] += output_tokens
    stats["prm_batches"].append((len(input_ids), input_tokens, evaluation_time,
                               max(len(input_id) for input_id in input_ids), output_tokens))
    
    for candidate, step_reward in zip(candidates, step_rewards):
        # [DraftPipelineChange] Save PRM wall time so rejected draft-prefetch waste can use the PRM window.
        candidate.last_prm_time = evaluation_time
        if len(step_reward) > 0:
            candidate.score = step_reward[-1]
            all_rewards[candidate.prob_idx].append(round(candidate.score, 6))
    
    return candidates

def _evaluate_with_prm_log_scores_backtracking(candidates, problems, prm_client, prm_tokenizer, args, stats, all_rewards):
    """Evaluate candidates with PRM and compute cumulative log scores for backtracking."""
    if not candidates:
        return candidates
    
    # Evaluate with PRM
    candidates = _evaluate_with_prm_backtracking(candidates, problems, prm_client, prm_tokenizer, args, stats, all_rewards)
    
    # Compute cumulative log scores
    for candidate in candidates:
        parent_log_score = candidate.log_score  # Already contains parent's cumulative log score
        current_prm_score = max(candidate.score, 1e-10)  # Avoid log(0)
        candidate.log_score = parent_log_score + math.log(current_prm_score)
    
    return candidates
def _check_completion_beam(beam, args, draft_model, draft_tokenizer, target_tokenizer, stats):
    """Check for completion without using fixed output arrays - return remaining beam and completed nodes"""
    remaining_beam = []
    completed = []
    for node in beam:
        # Prompt/instruction nodes are never considered "completed"; they are
        # only a carrier for context before the next generation step.
        if getattr(node, "response_type", None) == 0 and node.parent is not None:
            remaining_beam.append(node)
            continue
        full_text = node.get_full_text()
        # Get the stop_reason directly from the node
        stop_reason = node.stop_reason
        # Stopping conditions (mirroring main_online.py)
        should_stop = False
        if stop_reason is None:
            should_stop = True
        elif len(draft_tokenizer.encode(full_text)) >= args.max_tokens_per_call:
            should_stop = True
        elif len(target_tokenizer.encode(full_text)) >= args.max_tokens_per_call:
            should_stop = True
        elif node.step >= args.max_steps - 1:
            should_stop = True
        # Note: patience/num_unchanged is not tracked per node in beam/tree search, so we skip it here
        if should_stop:
            # [DraftPipelineChange] Completion can leave a next-step prefetch unused; do not count it as reject waste.
            _stop_draft_pipeline_prefetch_for_nodes(
                [node], draft_model, draft_tokenizer, args, stats, "unused_completion"
            )
            completed.append(node)
        else:
            remaining_beam.append(node)
    return remaining_beam, completed


def get_responses_beam_search(args, target_client, prm_client, draft_tokenizer, target_tokenizer, prm_tokenizer, draft_model, prompt, problem):
    """Run beam search for a single problem."""
    assert isinstance(prompt, str) and isinstance(problem, str), "Beam search now only supports one problem at a time as a string."
    if not args.beam_search:
        raise ValueError("Beam search requested without --beam_search flag.")

    expansion_factor = getattr(args, 'beam_expansion_factor', 1)
    score_type = "log scores" if args.beam_use_log_scores else "regular scores"
    print(
        f"[INFO] Running beam search with width {args.beam_width}, expansion factor {expansion_factor} "
        f"(generates {args.beam_width * expansion_factor} candidates per step), using {score_type}"
    )
    return _beam_search_with_backtracking(
        args,
        target_client,
        prm_client,
        draft_tokenizer,
        target_tokenizer,
        prm_tokenizer,
        draft_model,
        prompt,
        problem,
    )


def get_responses_without_tree(args, target_client, prm_client, draft_tokenizer, target_tokenizer, prm_tokenizer, tree_draft_model, prompt, problem):
    """Direct implementation for non-tree approach using a local draft model, now only accepts one problem at a time (as string)"""
    assert isinstance(prompt, str) and isinstance(problem, str), "get_responses_without_tree now only supports one problem at a time as a string."
    # Use the original logic, but for a single prompt/problem
    outputs = [None]
    token_counts = [(0, 0, 0)]
    step_info = [[]]
    current_prompts = [(0, prompt, [])]
    all_rewards = [[]]
    current_problems = [problem]
    num_step = 0
    pre_num_finished = 0
    num_unchanged = 0

    # Add timing and token stats
    draft_time = 0.0
    target_time = 0.0
    prm_time = 0.0
    draft_tokens = 0
    target_tokens = 0
    prm_tokens = 0

    # Add input/output token tracking
    draft_input_tokens = 0
    draft_output_tokens = 0
    target_input_tokens = 0
    target_output_tokens = 0
    prm_input_tokens_total = 0
    prm_output_tokens = 0

    # Add batch stats
    draft_batch_sizes = []
    target_batch_sizes = []
    prm_batch_sizes = []

    while current_prompts:
        batch_prompts = [p + ''.join(r[0] for r in responses) for _, p, responses in current_prompts]
        batch_size = len(batch_prompts)

        # First generate with local draft model using batch processing
        batch_draft_input_tokens = 0
        batch_draft_output_tokens = 0
        max_tokens_in_batch = 0

        start = time.time()
        print(f"Batch size: {len(batch_prompts)}")
        if hasattr(tree_draft_model, 'completions_create'):
            with nvtx.range("draft_model_generation"):
                response = tree_draft_model.completions_create(
                    prompt=batch_prompts,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens_per_call,
                    stop=[args.step_word],
                )
            draft_responses = response.choices
        else:
            with nvtx.range("draft_model_generation"):
                response = tree_draft_model.completions.create(
                    model=args.draft_model_name_or_path.split("/")[-1],
                    prompt=batch_prompts,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens_per_call,
                    stop=[args.step_word],
                )
            draft_responses = response.choices
        print(f"Generated {len(draft_responses)} responses")

        batch_draft_input_tokens = sum(len(draft_tokenizer.encode(prompt)) for prompt in batch_prompts)
        batch_draft_output_tokens = sum(len(draft_tokenizer.encode(resp.text + args.step_word)) for resp in draft_responses)
        max_tokens_in_batch = max(len(draft_tokenizer.encode(resp.text + args.step_word)) for resp in draft_responses)

        step_time = time.time() - start
        draft_time += step_time
        print(f"Draft generation took {step_time:.2f}s ({batch_draft_output_tokens/step_time:.2f} tokens/sec)")

        draft_input_tokens += batch_draft_input_tokens
        draft_output_tokens += batch_draft_output_tokens
        draft_batch_sizes.append((batch_size, batch_draft_output_tokens, step_time, max_tokens_in_batch, batch_draft_input_tokens))

        # Evaluate draft responses with PRM to determine which ones to accept
        full_responses = [''.join(r[0] for r in prev_resp) + new_resp.text
                    for (_, _, prev_resp), new_resp in zip(current_prompts, draft_responses)]
        processed_data = [
            prepare_input(p, full_resp, tokenizer=prm_tokenizer, step_token=args.step_word, model_name=args.prm_name_or_path)
            for p, full_resp in zip(current_problems, full_responses)
        ]
        input_ids, steps, reward_flags = zip(*processed_data)
        start = time.time()
        with nvtx.range("prm_model_evaluation"):
            rewards = prm_client.embeddings.create(
                input=input_ids,
                model=args.prm_name_or_path.split("/")[-1],
            )
        step_time = time.time() - start
        prm_time += step_time

        batch_prm_input_tokens = sum(len(input_id) for input_id in input_ids)
        max_prm_tokens_in_batch = max(len(input_id) for input_id in input_ids)

        step_rewards = derive_step_rewards_vllm(rewards, reward_flags, args.prm_name_or_path)

        batch_prm_output_tokens = sum(len(reward_sequence) for reward_sequence in step_rewards)

        prm_input_tokens_total += batch_prm_input_tokens
        prm_output_tokens += batch_prm_output_tokens
        prm_tokens += batch_prm_input_tokens
        prm_batch_sizes.append((batch_size, batch_prm_input_tokens, step_time, max_prm_tokens_in_batch, batch_prm_output_tokens))

        good_prompts = []
        bad_prompts = []
        for (orig_idx, prompt, prev_responses), draft_response, step_reward in zip(current_prompts, draft_responses, step_rewards):
            all_rewards[orig_idx].append(round(step_reward[-1], 6))
            if step_reward[-1] >= args.prm_threshold:
                good_prompts.append((orig_idx, prompt, prev_responses, draft_response, True))
            else:
                draft_response_text = draft_response.text + args.step_word
                token_counts[orig_idx] = (
                    token_counts[orig_idx][0],
                    token_counts[orig_idx][1],
                    token_counts[orig_idx][2] + len(draft_tokenizer.encode(draft_response_text))
                )
                bad_prompts.append((orig_idx, prompt, prev_responses))

        if bad_prompts:
            batch_prompts = [p + ''.join(r[0] for r in responses) for _, p, responses in bad_prompts]
            target_batch_size = len(batch_prompts)
            start = time.time()
            with nvtx.range("target_model_generation"):
                target_responses = target_client.completions.create(
                    model=args.target_model_name_or_path.split("/")[-1],
                    prompt=batch_prompts,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens_per_call,
                    n=1,
                    stop=[args.step_word],
                ).choices
            step_time = time.time() - start
            target_time += step_time
            target_responses = sorted(target_responses, key=lambda x: int(x.index))

            batch_target_input_tokens = sum(len(target_tokenizer.encode(prompt)) for prompt in batch_prompts)
            batch_target_output_tokens = sum(len(target_tokenizer.encode(resp.text + args.step_word)) for resp in target_responses)
            max_target_tokens_in_batch = max(len(target_tokenizer.encode(resp.text + args.step_word)) for resp in target_responses)

            target_input_tokens += batch_target_input_tokens
            target_output_tokens += batch_target_output_tokens
            target_batch_sizes.append((target_batch_size, batch_target_output_tokens, step_time, max_target_tokens_in_batch, batch_target_input_tokens))

            for (orig_idx, prompt, prev_responses), target_response in zip(bad_prompts, target_responses):
                good_prompts.append((orig_idx, prompt, prev_responses, target_response, False))

        next_prompts = []
        next_problems = []
        for orig_idx, prompt, prev_responses, response, used_draft in sorted(good_prompts, key=lambda x: x[0]):
            response_text = response.text + args.step_word
            client_id = 1 if used_draft else 2
            tokenizer = draft_tokenizer if client_id == 1 else target_tokenizer
            num_tokens = len(tokenizer.encode(response_text))
            if client_id == 1:
                token_counts[orig_idx] = (token_counts[orig_idx][0] + num_tokens, token_counts[orig_idx][1], token_counts[orig_idx][2])
                draft_tokens += num_tokens
            else:
                token_counts[orig_idx] = (token_counts[orig_idx][0], token_counts[orig_idx][1] + num_tokens, token_counts[orig_idx][2])
                target_tokens += num_tokens
            step_info[orig_idx].append((num_step, client_id))
            full_responses = prev_responses + [(response_text, client_id)]
            full_responses_text = ''.join(r[0] for r in full_responses)
            if (getattr(response, 'stop_reason', None) is None) \
             or len(draft_tokenizer.encode(prompt + full_responses_text)) >= args.max_tokens_per_call \
             or len(target_tokenizer.encode(prompt + full_responses_text)) >= args.max_tokens_per_call \
             or num_step >= args.max_steps - 1 \
             or False: #num_unchanged >= args.patience - 1:
                outputs[orig_idx] = full_responses_text[:-len(args.step_word)]
            else:
                next_prompts.append((orig_idx, prompt, full_responses))
                next_problems.append(problem)

        current_prompts = next_prompts
        current_problems = next_problems
        assert len(current_prompts) == len(current_problems)

        if len(outputs) - len(current_prompts) > pre_num_finished:
            num_unchanged = 0
            pre_num_finished = len(outputs) - len(current_prompts)
        else:
            num_unchanged += 1
        print(f"#### Step {num_step}: Completed {pre_num_finished} / {len(outputs)}, #unchanged {num_unchanged} / {args.patience}")
        num_step += 1

    stats = {
        "draft_time": draft_time,
        "target_time": target_time,
        "prm_time": prm_time,
        "draft_tokens": draft_tokens,
        "target_tokens": target_tokens,
        "prm_tokens": prm_tokens,
        "draft_input_tokens": draft_input_tokens,
        "draft_output_tokens": draft_output_tokens,
        "target_input_tokens": target_input_tokens,
        "target_output_tokens": target_output_tokens,
        "prm_input_tokens": prm_input_tokens_total,
        "prm_output_tokens": prm_output_tokens,
        "draft_batches": draft_batch_sizes,
        "target_batches": target_batch_sizes,
        "prm_batches": prm_batch_sizes,
    }
    
    # Add time distribution percentages
    total_time = draft_time + target_time + prm_time
    if total_time > 0:
        stats["draft_time_percentage"] = (draft_time / total_time) * 100
        stats["target_time_percentage"] = (target_time / total_time) * 100
        stats["prm_time_percentage"] = (prm_time / total_time) * 100
    else:
        stats["draft_time_percentage"] = 0.0
        stats["target_time_percentage"] = 0.0
        stats["prm_time_percentage"] = 0.0
    stats["total_time"] = total_time
    
    return outputs, token_counts, step_info, all_rewards, stats

def main(target_client, prm_client, draft_client, draft_tokenizer, target_tokenizer, prm_tokenizer, draft_model, data_name, args):
    examples, processed_samples, out_file = prepare_data(data_name, args)
    print("=" * 50)
    print("data:", data_name, " ,remain samples:", len(examples))
    if len(examples) > 0:
        print(examples[0])
    elif len(processed_samples) > 0:
        print("No remaining samples to generate; evaluating cached outputs only.")

    # init python executor
    if "pal" in args.prompt_type:
        executor = PythonExecutor(get_answer_expr="solution()")
    else:
        executor = PythonExecutor(get_answer_from_stdout=True)

    samples = []
    # For per-problem latency and intervention metrics
    per_problem_latency = []
    per_problem_target_intervene = []
    per_problem_steps = []
    per_problem_step_latencies = []

    for example in tqdm(examples, total=len(examples)):
        idx = example["idx"]

        # parse question and answer
        example["question"] = parse_question(example, data_name)
        if example["question"] == "":
            continue
        gt_cot, gt_ans = parse_ground_truth(example, data_name)
        example["gt_ans"] = gt_ans
        full_prompt = construct_prompt(example, data_name, args)

        if idx == args.start:
            print(full_prompt)

        sample = {
            "idx": idx,
            "question": example["question"],
            "gt_cot": gt_cot,
            "gt": gt_ans,
            "prompt": full_prompt,
        }

        # add remain fields
        for key in [
            "level",
            "type",
            "unit",
            "solution_type",
            "choices",
            "solution",
            "ques_type",
            "ans_type",
            "answer_type",
            "dataset",
            "subfield",
            "filed",
            "theorem",
            "answer",
        ]:
            if key in example:
                sample[key] = example[key]
        samples.append(sample)

    # repeat n times
    input_prompts = [
        sample["prompt"] for sample in samples for _ in range(args.n_sampling)
    ]
    if args.apply_chat_template:
        input_prompts = [
            draft_tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt.strip()}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in input_prompts
        ]
    remain_prompts = input_prompts
    remain_prompts = [(i, prompt) for i, prompt in enumerate(remain_prompts)]
    end_prompts = []
    model_stats = _init_stats()

    max_func_call = 1 if args.prompt_type in ["cot", "pal"] else 4

    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]

    if args.prompt_type in ["cot"]:
        stop_words.append("\n\nQuestion:")
    if args.prompt_type in ["pal", "tool-integrated", "jiuzhang_tora"]:
        stop_words.extend(["\n\n---", "```output"])
    elif args.prompt_type in ["wizard_zs", "platypus_fs"]:
        stop_words.extend(["Instruction", "Response"])
    elif "jiuzhang" in args.prompt_type:
        stop_words.append("\n\n## Question")
    elif "numina" in args.prompt_type:
        stop_words.append("\n### Problem")
    elif "pure" in args.prompt_type:
        stop_words.append("\n\n\n")

    # start inference
    start_time = time.time()

    for epoch in range(max_func_call):
        print("-" * 20, "Epoch", epoch)
        current_prompts = remain_prompts
        if len(current_prompts) == 0:
            break
        prompts = [item[1] for item in current_prompts]
        problems = [sample["question"] for sample in samples]
        assert len(prompts) == len(problems)


        outputs = []
        token_counts = []
        turn_info = []
        all_rewards = []
        # For per-problem metrics
        problem_start_times = []
        problem_end_times = []
        problem_target_steps = []
        problem_total_steps = []
        problem_step_latencies = []
        problem_counter = 0  # Track problems processed for memory management



        # 在 for prompt, problem in zip(prompts, problems): 之前
        all_model_stats = []
        for prompt, problem in zip(prompts, problems):
            problem_counter += 1
            t0 = time.time()
            if args.beam_search:
                out, tcount, tinfo, areward, mstats = get_responses_beam_search(
                    args,
                    target_client,
                    prm_client,
                    draft_tokenizer,
                    target_tokenizer,
                    prm_tokenizer,
                    draft_model,
                    prompt,
                    problem,
                )
            else:
                out, tcount, tinfo, areward, mstats = get_responses_without_tree(
                    args,
                    target_client,
                    prm_client,
                    draft_tokenizer,
                    target_tokenizer,
                    prm_tokenizer,
                    draft_model,
                    prompt,
                    problem,
                )
            t1 = time.time()

            
            outputs.extend(out)
            all_model_stats.append(mstats)# 每一题的 stats 都存进列表
            
            # Memory cleanup based on frequency setting
            if (args.use_local_models and args.memory_cleanup_freq > 0 and 
                problem_counter % args.memory_cleanup_freq == 0):
                try:
                    import gc
                    import torch
                    # Clear Python garbage
                    gc.collect()
                    # Clear CUDA cache
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        # Force synchronization to ensure cleanup
                        torch.cuda.synchronize()
                        if args.debug:
                            print(f"Memory cleanup after problem {problem_counter}")
                except Exception as e:
                    print(f"Memory cleanup warning: {e}")
            token_counts.extend(tcount)
            turn_info.extend(tinfo)
            all_rewards.extend(areward)
            # Aggregate model stats across all newly processed problems. Starting from
            # _init_stats() keeps reruns over cached outputs from failing when there
            # are no remaining samples in this invocation.
            scalar_stat_keys = [
                "draft_time",
                "target_time",
                "prm_time",
                "draft_tokens",
                "target_tokens",
                "prm_tokens",
                "draft_input_tokens",
                "draft_output_tokens",
                "target_input_tokens",
                "target_output_tokens",
                "prm_input_tokens",
                "prm_output_tokens",
                "bt_forward_good",
                "target_intervention_attempts",
                "target_intervention_successes",
                "bt_anchor_attempts",
                "bt_anchor_count_total",
                "bt_anchor_successes",
                "total_backtrack_attempts",
                "prefetch_consume_time",
                "prefetch_consume_calls",
                "prefetch_collect_wait_time",
                "prefetch_collect_calls",
                "prefetch_stop_time",
                "prefetch_stop_calls",
                "prefetch_stop_request_time",
                "prefetch_stop_requests",
                "prefetch_start_time",
                "prefetch_start_calls",
                # [DraftPipelineChange] Aggregate draft pipeline scalar stats across samples.
                "draft_pipeline_start_calls",
                "draft_pipeline_consume_calls",
                "draft_pipeline_reused_tokens",
                "draft_pipeline_reused_latency",
                "draft_pipeline_wasted_tokens",
                "draft_pipeline_wasted_latency",
                "draft_pipeline_wasted_actual_latency",
                "draft_pipeline_wasted_latency_overhead",
                "draft_pipeline_unused_completion_tokens",
                "draft_pipeline_unused_completion_latency",
            ]
            list_stat_keys = [
                "draft_batches",
                "target_batches",
                "prm_batches",
                "step_details",
                "step_attempts",
                "target_trace_events",
                "backtrack_events",
                # [DraftPipelineChange] Keep detailed per-event draft pipeline traces.
                "draft_pipeline_events",
            ]
            for key in scalar_stat_keys:
                model_stats[key] += mstats.get(key, 0)
            for key in list_stat_keys:
                model_stats[key].extend(mstats.get(key, []))
            # Per-problem latency
            problem_start_times.append(t0)
            problem_end_times.append(t1)
            # Target model intervention and step latency
            if tinfo and len(tinfo) > 0:
                steps = tinfo[0]
                total_steps = len(steps)
                target_steps = sum(1 for _, cid in steps if cid == 2)
                problem_total_steps.append(total_steps)
                problem_target_steps.append(target_steps)
            else:
                problem_total_steps.append(0)
                problem_target_steps.append(0)
            # Per-step latency not available for tree/beam (unless instrumented inside search)
            problem_step_latencies.append([])

        assert len(outputs) == len(current_prompts)

        # process all outputs
        remain_prompts = []
        remain_codes = []
        for (i, query), output in zip(current_prompts, outputs):
            output = output.rstrip()
            query += output
            if args.prompt_type == "pal":
                remain_prompts.append((i, query))
                if "```python" in output:
                    output = extract_program(query)
                remain_codes.append(output)
            elif args.prompt_type == "cot":
                end_prompts.append((i, query))
            elif "boxed" not in output and output.endswith("```"):
                program = extract_program(query)
                remain_prompts.append((i, query))
                remain_codes.append(program)
            else:
                end_prompts.append((i, query))

        # execute the remain prompts
        remain_results = executor.batch_apply(remain_codes)
        for k in range(len(remain_prompts)):
            i, query = remain_prompts[k]
            res, report = remain_results[k]
            exec_result = res if res else report
            if "pal" in args.prompt_type:
                exec_result = "\\boxed{" + exec_result + "}"
            exec_result = f"\n```output\n{exec_result}\n```\n"
            query += exec_result
            # not end
            if epoch == max_func_call - 1:
                query += "\nReach max function call limit."
            remain_prompts[k] = (i, query)

    # unsolved samples
    print("Unsolved samples:", len(remain_prompts))
    end_prompts.extend(remain_prompts)
    # sort by idx
    end_prompts = sorted(end_prompts, key=lambda x: x[0])

    # remove input_prompt from end_prompt
    codes = []
    assert len(input_prompts) == len(end_prompts)
    for i in range(len(input_prompts)):
        _, end_prompt = end_prompts[i]
        code = end_prompt.split(input_prompts[i])[-1].strip()
        for stop_word in stop_words:
            if stop_word in code:
                code = code.split(stop_word)[0].strip()
        codes.append(code)

    # extract preds
    print("\nExtracting predictions from model outputs...")
    if len(codes) > 0:
        print(f"Sample code output: {codes[0][:200]}...\n")
        # Check if the output contains a boxed answer
        if "\\boxed{" in codes[0]:
            print("Found boxed answer in output!")
        else:
            print("WARNING: No boxed answer found in output!")
            # Try to find where the answer might be
            if "answer is" in codes[0].lower():
                print("Found 'answer is' in output, but no boxed format")
            if "therefore" in codes[0].lower():
                print("Found 'therefore' in output, but no boxed format")
        
    results = [
        run_execute(executor, code, args.prompt_type, data_name) for code in codes
    ]
    
    # Print sample result
    if len(results) > 0:
        print(f"Sample extracted answer: {results[0][0]}")
        print(f"Sample report: {results[0][1]}")
        
        # Check if the extracted answer is valid
        if results[0][0] is None or results[0][0] == "":
            print("WARNING: Extracted answer is empty!")
        elif results[0][0] in ["A", "B", "C", "D", "E"]:
            print("Extracted a valid multiple-choice answer")
        elif results[0][0].isdigit() or results[0][0].replace('.', '', 1).isdigit():
            print("Extracted a valid numeric answer")
        else:
            print(f"Extracted answer format: {results[0][0]}")
            
        # Print the ground truth for comparison
        if len(samples) > 0:
            print(f"Ground truth answer: {samples[0]['gt']}")
            print(f"Question: {samples[0]['question'][:100]}...")
            
        # Check if the boxed format is being parsed correctly
        if "\\boxed{" in codes[0]:
            boxed_start = codes[0].find("\\boxed{") + 7
            boxed_end = codes[0].find("}", boxed_start)
            if boxed_end > boxed_start:
                boxed_content = codes[0][boxed_start:boxed_end]
                print(f"Boxed content: {boxed_content}")
                print(f"Does this match extracted answer? {boxed_content == results[0][0]}")
            else:
                print("Malformed boxed answer")
                
        # Check if the answer is being extracted correctly from the code
        print(f"\nCode snippet around potential answer:")
        answer_idx = codes[0].lower().find("answer is")
        if answer_idx >= 0:
            print(codes[0][answer_idx:answer_idx+100])
        boxed_idx = codes[0].find("\\boxed{")
        if boxed_idx >= 0:
            print(codes[0][boxed_idx:boxed_idx+100])
        
    time_use = time.time() - start_time

    # put results back to examples
    all_samples = []
    for i, sample in enumerate(samples):
        code = codes[i * args.n_sampling : (i + 1) * args.n_sampling]
        result = results[i * args.n_sampling : (i + 1) * args.n_sampling]
        preds = [item[0] for item in result]
        reports = [item[1] for item in result]
        for j in range(len(preds)):
            if sample["gt"] in ["A", "B", "C", "D", "E"] and preds[j] not in [
                "A",
                "B",
                "C",
                "D",
                "E",
            ]:
                preds[j] = choice_answer_clean(code[j])
            elif is_multi_choice(sample["gt"]) and not is_multi_choice(preds[j]):
                # remove any non-choice char
                preds[j] = "".join(
                    [c for c in preds[j] if c in ["A", "B", "C", "D", "E"]]
                )

        sample.pop("prompt")
        # Add per-problem metrics
        latency = problem_end_times[i] - problem_start_times[i] if i < len(problem_end_times) else None
        # Compute steps and target intervention from turn_info (step_info)
        step_info = turn_info[i] if i < len(turn_info) else []
        total_steps = len(step_info)
        target_steps = sum(1 for _, cid in step_info if cid == 2)
        target_intervene_pct = (target_steps / total_steps) if total_steps > 0 else 0.0
        per_step_latency = (latency / total_steps) if (latency is not None and total_steps > 0) else None




        # --- 修改后的保存逻辑 ---
        # 检查 sample 中是否已经有了记录，如果有，就用 extend 累加，而不是覆盖

        # 使用 i 从 all_model_stats 中取出属于该 sample 的 stats
        current_mstats = all_model_stats[i] if i < len(all_model_stats) else {}



        sample.update(
            {"code": code, "pred": preds, "report": reports, 
             "token_counts": token_counts[i], "turn_info": turn_info[i], "reward": all_rewards[i],
             "latency": latency, 


             "step_details": current_mstats.get("step_details", []),
             "step_attempts": current_mstats.get("step_attempts", []),
             "target_trace_events": current_mstats.get("target_trace_events", []),
             "backtrack_events": current_mstats.get("backtrack_events", []),
             # [DraftPipelineChange] Persist per-sample draft pipeline traces and summary counters.
             "draft_pipeline_events": current_mstats.get("draft_pipeline_events", []),
             "draft_pipeline_wasted_tokens": current_mstats.get("draft_pipeline_wasted_tokens", 0),
             "draft_pipeline_wasted_latency": current_mstats.get("draft_pipeline_wasted_latency", 0.0),
             "draft_pipeline_wasted_actual_latency": current_mstats.get("draft_pipeline_wasted_actual_latency", 0.0),
             "draft_pipeline_wasted_latency_overhead": current_mstats.get("draft_pipeline_wasted_latency_overhead", 0.0),
             "draft_pipeline_reused_tokens": current_mstats.get("draft_pipeline_reused_tokens", 0),
             "draft_pipeline_reused_latency": current_mstats.get("draft_pipeline_reused_latency", 0.0),
             "draft_pipeline_unused_completion_tokens": current_mstats.get("draft_pipeline_unused_completion_tokens", 0),
             "draft_pipeline_unused_completion_latency": current_mstats.get("draft_pipeline_unused_completion_latency", 0.0),
             # ... 其他字段

             "target_intervene_pct": target_intervene_pct, "total_steps": total_steps, "target_steps": target_steps, "per_step_latency": per_step_latency}
        )
        all_samples.append(sample)


    # add processed samples
    all_samples.extend(processed_samples)
    all_samples, result_json = evaluate(
        samples=all_samples,
        data_name=data_name,
        prompt_type=args.prompt_type,
        execute=True,
    )

    # Add aggregate metrics for latency and intervention
    latencies = [s["latency"] for s in all_samples if "latency" in s and s["latency"] is not None]
    target_intervene_pcts = [s["target_intervene_pct"] for s in all_samples if "target_intervene_pct" in s]
    steps_per_problem = [s["total_steps"] for s in all_samples if "total_steps" in s and s["total_steps"] is not None]
    per_step_latencies = [s["per_step_latency"] for s in all_samples if "per_step_latency" in s and s["per_step_latency"] is not None]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    avg_target_intervene_pct = sum(target_intervene_pcts) / len(target_intervene_pcts) if target_intervene_pcts else 0.0
    avg_steps_per_problem = sum(steps_per_problem) / len(steps_per_problem) if steps_per_problem else 0.0
    avg_per_step_latency = sum(per_step_latencies) / len(per_step_latencies) if per_step_latencies else 0.0
    result_json["avg_problem_latency"] = avg_latency
    result_json["avg_target_intervene_pct"] = avg_target_intervene_pct
    result_json["avg_steps_per_problem"] = avg_steps_per_problem
    result_json["avg_per_step_latency"] = avg_per_step_latency



    # save outputs
    if len(processed_samples) < len(all_samples) and args.save_outputs:
        save_jsonl(all_samples, out_file)

    # save metrics
    result_json["time_use_in_second"] = time_use
    result_json["time_use_in_minite"] = (
        f"{int(time_use // 60)}:{int(time_use % 60):02d}"
    )

    # Write the same output to a text file (rest of summary output below)
    # ...existing code for summary output using wprint...

    # save outputs
    if len(processed_samples) < len(all_samples) and args.save_outputs:
        save_jsonl(all_samples, out_file)

    # save metrics
    result_json["time_use_in_second"] = time_use
    result_json["time_use_in_minite"] = (
        f"{int(time_use // 60)}:{int(time_use % 60):02d}"
    )
    result_json["draft_time"] = model_stats["draft_time"]
    result_json["target_time"] = model_stats["target_time"]
    result_json["prm_time"] = model_stats["prm_time"]
    # Add prefetch instrumentation metrics
    result_json["prefetch_consume_time"] = model_stats.get("prefetch_consume_time", 0.0)
    result_json["prefetch_consume_calls"] = model_stats.get("prefetch_consume_calls", 0)
    result_json["prefetch_collect_wait_time"] = model_stats.get("prefetch_collect_wait_time", 0.0)
    result_json["prefetch_collect_calls"] = model_stats.get("prefetch_collect_calls", 0)
    result_json["prefetch_stop_time"] = model_stats.get("prefetch_stop_time", 0.0)
    result_json["prefetch_stop_calls"] = model_stats.get("prefetch_stop_calls", 0)
    result_json["prefetch_stop_request_time"] = model_stats.get("prefetch_stop_request_time", 0.0)
    result_json["prefetch_stop_requests"] = model_stats.get("prefetch_stop_requests", 0)
    result_json["prefetch_start_time"] = model_stats.get("prefetch_start_time", 0.0)
    result_json["prefetch_start_calls"] = model_stats.get("prefetch_start_calls", 0)
    # [DraftPipelineChange] Add draft-pipeline instrumentation metrics to *_metrics.json.
    result_json["draft_pipeline_start_calls"] = model_stats.get("draft_pipeline_start_calls", 0)
    result_json["draft_pipeline_consume_calls"] = model_stats.get("draft_pipeline_consume_calls", 0)
    result_json["draft_pipeline_reused_tokens"] = model_stats.get("draft_pipeline_reused_tokens", 0)
    result_json["draft_pipeline_reused_latency"] = model_stats.get("draft_pipeline_reused_latency", 0.0)
    result_json["draft_pipeline_wasted_tokens"] = model_stats.get("draft_pipeline_wasted_tokens", 0)
    result_json["draft_pipeline_wasted_latency"] = model_stats.get("draft_pipeline_wasted_latency", 0.0)
    result_json["draft_pipeline_wasted_actual_latency"] = model_stats.get("draft_pipeline_wasted_actual_latency", 0.0)
    result_json["draft_pipeline_wasted_latency_overhead"] = model_stats.get("draft_pipeline_wasted_latency_overhead", 0.0)
    result_json["draft_pipeline_unused_completion_tokens"] = model_stats.get("draft_pipeline_unused_completion_tokens", 0)
    result_json["draft_pipeline_unused_completion_latency"] = model_stats.get("draft_pipeline_unused_completion_latency", 0.0)
    result_json["draft_pipeline_event_count"] = len(model_stats.get("draft_pipeline_events", []))

    # Print prefetch instrumentation summary for quick inspection
    if (
        model_stats.get("prefetch_consume_calls", 0)
        or model_stats.get("prefetch_stop_calls", 0)
        or model_stats.get("prefetch_start_calls", 0)
    ):
        consume_calls = max(model_stats.get("prefetch_consume_calls", 0), 1)
        collect_calls = max(model_stats.get("prefetch_collect_calls", 0), 1)
        stop_calls = max(model_stats.get("prefetch_stop_calls", 0), 1)
        stop_requests = max(model_stats.get("prefetch_stop_requests", 0), 1)
        start_calls = max(model_stats.get("prefetch_start_calls", 0), 1)
        print("\n[Prefetch Timing]")
        print(
            f"start: {model_stats.get('prefetch_start_time', 0.0):.3f}s total "
            f"({model_stats.get('prefetch_start_calls', 0)} calls, "
            f"avg {model_stats.get('prefetch_start_time', 0.0)/start_calls:.4f}s)"
        )
        print(
            f"consume: {model_stats.get('prefetch_consume_time', 0.0):.3f}s total "
            f"({model_stats.get('prefetch_consume_calls', 0)} calls, "
            f"avg {model_stats.get('prefetch_consume_time', 0.0)/consume_calls:.4f}s)"
        )
        print(
            f"collect wait: {model_stats.get('prefetch_collect_wait_time', 0.0):.3f}s total "
            f"({model_stats.get('prefetch_collect_calls', 0)} calls, "
            f"avg {model_stats.get('prefetch_collect_wait_time', 0.0)/collect_calls:.4f}s)"
        )
        print(
            f"stop: {model_stats.get('prefetch_stop_time', 0.0):.3f}s total "
            f"({model_stats.get('prefetch_stop_calls', 0)} calls, "
            f"avg {model_stats.get('prefetch_stop_time', 0.0)/stop_calls:.4f}s)"
        )
        print(
            f"stop requests: {model_stats.get('prefetch_stop_request_time', 0.0):.3f}s total "
            f"({model_stats.get('prefetch_stop_requests', 0)} requests, "
            f"avg {model_stats.get('prefetch_stop_request_time', 0.0)/stop_requests:.4f}s)"
        )
    
    # Calculate token stats first
    llm1_tokens = [0, 0] # (correct, wrong)
    llm1_discarded_tokens = [0, 0]
    llm2_tokens = [0, 0]
    for i, sample in enumerate(all_samples):
        if sample["score"][0]:
            llm1_tokens[0] += sample["token_counts"][0]
            llm2_tokens[0] += sample["token_counts"][1]
            llm1_discarded_tokens[0] += sample["token_counts"][2]
        else:
            llm1_tokens[1] += sample["token_counts"][0]
            llm2_tokens[1] += sample["token_counts"][1]
            llm1_discarded_tokens[1] += sample["token_counts"][2]
    
    # Use the correct token counts from per-sample accounting
    result_json["draft_tokens"] = sum(llm1_tokens)  # Only accepted tokens
    result_json["target_tokens"] = sum(llm2_tokens)
    result_json["prm_tokens"] = model_stats["prm_tokens"]  # PRM tokens are correctly counted
    result_json["draft_tokens_per_sec"] = sum(llm1_tokens) / model_stats["draft_time"] if model_stats["draft_time"] > 0 else 0
    result_json["target_tokens_per_sec"] = sum(llm2_tokens) / model_stats["target_time"] if model_stats["target_time"] > 0 else 0
    result_json["prm_tokens_per_sec"] = model_stats["prm_tokens"] / model_stats["prm_time"] if model_stats["prm_time"] > 0 else 0
    
    # Add input/output token metrics
    result_json["draft_input_tokens"] = model_stats["draft_input_tokens"]
    result_json["draft_output_tokens"] = model_stats["draft_output_tokens"]
    result_json["target_input_tokens"] = model_stats["target_input_tokens"]
    result_json["target_output_tokens"] = model_stats["target_output_tokens"]
    result_json["prm_input_tokens"] = model_stats["prm_input_tokens"]
    result_json["prm_output_tokens"] = model_stats["prm_output_tokens"]
    result_json["prm_output_tokens_per_sec"] = model_stats["prm_output_tokens"] / model_stats["prm_time"] if model_stats["prm_time"] > 0 else 0
    
    # Add time distribution percentages
    total_model_time = model_stats["draft_time"] + model_stats["target_time"] + model_stats["prm_time"]
    if total_model_time > 0:
        result_json["draft_time_percentage"] = (model_stats["draft_time"] / total_model_time) * 100
        result_json["target_time_percentage"] = (model_stats["target_time"] / total_model_time) * 100
        result_json["prm_time_percentage"] = (model_stats["prm_time"] / total_model_time) * 100
    else:
        result_json["draft_time_percentage"] = 0.0
        result_json["target_time_percentage"] = 0.0
        result_json["prm_time_percentage"] = 0.0
    result_json["total_model_time"] = total_model_time
    
    total_tokens = sum(llm1_tokens) + sum(llm2_tokens) + sum(llm1_discarded_tokens)
    total_tokens_for_correct_pred = llm1_discarded_tokens[0] + llm1_tokens[0] + llm2_tokens[0]
    total_tokens_for_wrong_pred = llm1_discarded_tokens[1] + llm1_tokens[1] + llm2_tokens[1]


    # Write the same output to a text file
    summary_file = out_file.replace(".jsonl", f"_{args.prompt_type}_summary.txt")
    with open(summary_file, "w") as fout:

        def wprint(s=""):
            print(s)
            fout.write(str(s) + "\n")

        # Write average per-problem metrics first
        wprint("\n=== Average Per-Problem Metrics ===")
        wprint(f"Average latency: {avg_latency:.3f} s")
        wprint(f"Average target_intervene_pct: {avg_target_intervene_pct:.2%}")
        wprint(f"Average total_steps: {avg_steps_per_problem:.2f}")
        target_steps_list = [s["target_steps"] for s in all_samples if "target_steps" in s and s["target_steps"] is not None]
        avg_target_steps = sum(target_steps_list) / len(target_steps_list) if target_steps_list else 0.0
        wprint(f"Average target_steps: {avg_target_steps:.2f}")
        wprint(f"Average per_step_latency: {avg_per_step_latency:.3f} s")

        # Then write dataset summary
        wprint(f"\n=== Dataset: {data_name} Summary ===")
        wprint(f"Accuracy: {result_json['acc']:.1f}")
        wprint(f"Time: {result_json['time_use_in_minite']} ({time_use:.2f}s)")

        # Calculate time distribution percentages
        total_model_time = model_stats["draft_time"] + model_stats["target_time"] + model_stats["prm_time"]
        if total_model_time > 0:
            draft_time_pct = (model_stats["draft_time"] / total_model_time) * 100
            target_time_pct = (model_stats["target_time"] / total_model_time) * 100
            prm_time_pct = (model_stats["prm_time"] / total_model_time) * 100
        else:
            draft_time_pct = target_time_pct = prm_time_pct = 0.0

        wprint(f"\nTime Distribution:")
        wprint(f"  Total model time: {total_model_time:.2f}s")
        wprint(f"  Draft model:  {model_stats['draft_time']:.2f}s ({draft_time_pct:.1f}%)")
        wprint(f"  Target model: {model_stats['target_time']:.2f}s ({target_time_pct:.1f}%)")
        wprint(f"  PRM model:    {model_stats['prm_time']:.2f}s ({prm_time_pct:.1f}%)")

        # [DraftPipelineChange] Human-readable draft pipeline summary in *_summary.txt.
        if model_stats.get("draft_pipeline_start_calls", 0) or model_stats.get("draft_pipeline_events"):
            wprint("\nDraft Pipeline Timing:")
            wprint(f"  Start calls:                 {model_stats.get('draft_pipeline_start_calls', 0)}")
            wprint(f"  Consume calls:               {model_stats.get('draft_pipeline_consume_calls', 0)}")
            wprint(f"  Reused tokens:               {model_stats.get('draft_pipeline_reused_tokens', 0)}")
            wprint(f"  Reused actual draft latency: {model_stats.get('draft_pipeline_reused_latency', 0.0):.3f}s")
            wprint(f"  Wasted tokens:               {model_stats.get('draft_pipeline_wasted_tokens', 0)}")
            wprint(f"  Wasted latency (PRM window): {model_stats.get('draft_pipeline_wasted_latency', 0.0):.3f}s")
            wprint(f"  Wasted actual draft latency: {model_stats.get('draft_pipeline_wasted_actual_latency', 0.0):.3f}s")
            wprint(f"  Wasted latency overhead:     {model_stats.get('draft_pipeline_wasted_latency_overhead', 0.0):.3f}s")
            wprint(f"  Unused completion tokens:    {model_stats.get('draft_pipeline_unused_completion_tokens', 0)}")
            wprint(f"  Unused completion latency:   {model_stats.get('draft_pipeline_unused_completion_latency', 0.0):.3f}s")
            wprint(f"  Event count:                 {len(model_stats.get('draft_pipeline_events', []))}")

        wprint("\nModel Throughput:")

        # Calculate accepted and rejected tokens
        accepted_draft = sum(llm1_tokens)
        rejected_draft = sum(llm1_discarded_tokens)

        # Draft model stats
        avg_draft_batch = sum(size for size, _, _, _, _ in model_stats['draft_batches']) / len(model_stats['draft_batches']) if model_stats['draft_batches'] else 0

        # Calculate speeds including rejected tokens
        total_draft_tokens = accepted_draft + rejected_draft  # Include both accepted and rejected tokens
        effective_draft_speed = total_draft_tokens / model_stats['draft_time'] if model_stats['draft_time'] > 0 else 0

        # Calculate per-prompt speeds - two approaches:
        # 1. Average tokens per prompt (throughput-based)
        avg_tokens_per_prompt_speed = (
            sum(tokens/time for _, tokens, time, _, _ in model_stats['draft_batches']) /
            sum(size for size, _, _, _, _ in model_stats['draft_batches'])
        ) if model_stats['draft_batches'] else 0

        # 2. Max tokens per prompt in batch (latency-based - more accurate for batched inference)
        max_tokens_per_prompt_speed = (
            sum(max_tokens/time for _, _, time, max_tokens, _ in model_stats['draft_batches']) /
            len(model_stats['draft_batches'])
        ) if model_stats['draft_batches'] else 0

        wprint(f"Draft model:")
        wprint(f"  Raw throughput:        {effective_draft_speed:.2f} tokens/sec (including rejected tokens)")
        wprint(f"  Accepted throughput:   {result_json['draft_tokens_per_sec']:.2f} tokens/sec")
        wprint(f"  Avg per-prompt speed:  {avg_tokens_per_prompt_speed:.2f} tokens/sec/prompt (throughput-based)")
        wprint(f"  Max per-prompt speed:  {max_tokens_per_prompt_speed:.2f} tokens/sec/prompt (latency-based)")
        wprint(f"  Average batch size:    {avg_draft_batch:.1f}")
        wprint(f"  Input tokens:          {model_stats['draft_input_tokens']} ({model_stats['draft_input_tokens']/sum(size for size, _, _, _, _ in model_stats['draft_batches']) if model_stats['draft_batches'] else 0:.1f} avg per prompt)")
        wprint(f"  Output tokens:         {model_stats['draft_output_tokens']}")
        wprint(f"  Total: {total_draft_tokens} tokens ({accepted_draft} accepted + {rejected_draft} rejected) in {model_stats['draft_time']:.2f}s")

        # Target model stats
        avg_target_batch = sum(size for size, _, _, _, _ in model_stats['target_batches']) / len(model_stats['target_batches']) if model_stats['target_batches'] else 0
        avg_target_per_prompt_speed = (
            sum(tokens/time for _, tokens, time, _, _ in model_stats['target_batches']) /
            sum(size for size, _, _, _, _ in model_stats['target_batches'])
        ) if model_stats['target_batches'] else 0
        max_target_per_prompt_speed = (
            sum(max_tokens/time for _, _, time, max_tokens, _ in model_stats['target_batches']) /
            len(model_stats['target_batches'])
        ) if model_stats['target_batches'] else 0

        wprint(f"\nTarget model:")
        wprint(f"  Raw throughput:        {result_json['target_tokens_per_sec']:.2f} tokens/sec (batched)")
        wprint(f"  Avg per-prompt speed:  {avg_target_per_prompt_speed:.2f} tokens/sec/prompt (throughput-based)")
        wprint(f"  Max per-prompt speed:  {max_target_per_prompt_speed:.2f} tokens/sec/prompt (latency-based)")
        wprint(f"  Average batch size:    {avg_target_batch:.1f}")
        wprint(f"  Input tokens:          {model_stats['target_input_tokens']} ({model_stats['target_input_tokens']/sum(size for size, _, _, _, _ in model_stats['target_batches']) if model_stats['target_batches'] else 0:.1f} avg per prompt)")
        wprint(f"  Output tokens:         {model_stats['target_output_tokens']}")
        wprint(f"  Total: {model_stats['target_tokens']} tokens in {model_stats['target_time']:.2f}s")

        # PRM model stats
        avg_prm_batch = sum(size for size, _, _, _, _ in model_stats['prm_batches']) / len(model_stats['prm_batches']) if model_stats['prm_batches'] else 0
        prm_input_speed = result_json['prm_tokens_per_sec']  # This is input token speed
        prm_output_speed = model_stats['prm_output_tokens'] / model_stats['prm_time'] if model_stats['prm_time'] > 0 else 0

        # Input-based per-prompt speeds
        avg_prm_input_per_prompt_speed = (
            sum(size * (tokens/time) for size, tokens, time, _, _ in model_stats['prm_batches']) /
            sum(size for size, _, _, _, _ in model_stats['prm_batches'])
        ) if model_stats['prm_batches'] else 0
        max_prm_input_per_prompt_speed = (
            sum(max_tokens/time for _, _, time, max_tokens, _ in model_stats['prm_batches']) /
            len(model_stats['prm_batches'])
        ) if model_stats['prm_batches'] else 0

        # Output-based per-prompt speeds
        avg_prm_output_per_prompt_speed = (
            sum(size * (output_tokens/time) for size, _, time, _, output_tokens in model_stats['prm_batches']) /
            sum(size for size, _, _, _, _ in model_stats['prm_batches'])
        ) if model_stats['prm_batches'] else 0
        max_prm_output_per_prompt_speed = (
            sum(output_tokens/time for _, _, time, _, output_tokens in model_stats['prm_batches']) /
            len(model_stats['prm_batches'])
        ) if model_stats['prm_batches'] else 0

        wprint(f"\nPRM model:")
        wprint(f"  Input throughput:      {prm_input_speed:.2f} tokens/sec (batched)")
        wprint(f"  Output throughput:     {prm_output_speed:.2f} rewards/sec")
        wprint(f"  Input per-prompt speeds:")
        wprint(f"    Avg per-prompt:      {avg_prm_input_per_prompt_speed:.2f} tokens/sec/prompt (throughput-based)")
        wprint(f"    Max per-prompt:      {max_prm_input_per_prompt_speed:.2f} tokens/sec/prompt (latency-based)")
        wprint(f"  Output per-prompt speeds:")
        wprint(f"    Avg per-prompt:      {avg_prm_output_per_prompt_speed:.2f} rewards/sec/prompt (throughput-based)")
        wprint(f"    Max per-prompt:      {max_prm_output_per_prompt_speed:.2f} rewards/sec/prompt (latency-based)")
        wprint(f"  Average batch size:    {avg_prm_batch:.1f}")
        wprint(f"  Input tokens:          {model_stats['prm_input_tokens']} ({model_stats['prm_input_tokens']/sum(size for size, _, _, _, _ in model_stats['prm_batches']) if model_stats['prm_batches'] else 0:.1f} avg per prompt)")
        wprint(f"  Output tokens:         {model_stats['prm_output_tokens']} (reward scores)")
        wprint(f"  Total: {model_stats['prm_tokens']} tokens in {model_stats['prm_time']:.2f}s")

        # Token usage breakdown
        wprint("\nToken Usage:")
        acceptance_rate = accepted_draft / (accepted_draft + rejected_draft) if (accepted_draft + rejected_draft) > 0 else 0
        wprint(f"Draft accepted:  {accepted_draft} tokens")
        wprint(f"Draft rejected:  {rejected_draft} tokens")
        wprint(f"Target used:     {sum(llm2_tokens)} tokens")
        wprint(f"Draft acceptance rate: {acceptance_rate:.1%}")

        wprint(f"\nCorrect predictions used {total_tokens_for_correct_pred} tokens")
        wprint(f"Wrong predictions used {total_tokens_for_wrong_pred} tokens")
        wprint("=" * 50)

    # Continue with storing metrics in result_json
    result_json["tokens_ratio_overall(llm1,llm2)"] = (
        (sum(llm1_tokens)+sum(llm1_discarded_tokens))/total_tokens, sum(llm2_tokens)/total_tokens
    ) if total_tokens > 0 else (0,0) 
    result_json["tokens_ratio_correct_prediction(llm1,llm2)"] = (
        (llm1_discarded_tokens[0]+llm1_tokens[0])/total_tokens_for_correct_pred, llm2_tokens[0]/total_tokens_for_correct_pred
    ) if total_tokens_for_correct_pred > 0 else (0,0) 
    result_json["tokens_ratio_wrong_prediction(llm1,llm2)"] = (
        (llm1_discarded_tokens[1]+llm1_tokens[1])/total_tokens_for_wrong_pred, llm2_tokens[1]/total_tokens_for_wrong_pred
    ) if total_tokens_for_wrong_pred > 0 else (0,0) 
    result_json["tokens_ratio(correct,wrong)"] = (
        total_tokens_for_correct_pred/total_tokens, total_tokens_for_wrong_pred/total_tokens
    ) if total_tokens > 0 else (0,0) 
    result_json["tokens_ratio_discarded(correct,wrong)"] = (
        llm1_discarded_tokens[0]/total_tokens_for_correct_pred, llm1_discarded_tokens[1]/total_tokens_for_wrong_pred
    ) if (total_tokens_for_correct_pred > 0 and total_tokens_for_wrong_pred > 0)  else (0,0) 
    result_json["acceptance_rate"] = (
        (llm1_tokens[0] + llm1_tokens[1])/(llm1_tokens[0] + llm1_tokens[1] + llm1_discarded_tokens[0] + llm1_discarded_tokens[1])
    ) if ((llm1_tokens[0] + llm1_tokens[1]) > 0)  else 0
    result_json["num_draft_tokens"] = sum(llm1_tokens) + sum(llm1_discarded_tokens)
    result_json["num_target_tokens"] = sum(llm2_tokens)

    with open(
        out_file.replace(".jsonl", f"_{args.prompt_type}_metrics.json"), "w"
    ) as f:
        json.dump(result_json, f, indent=4)
    return result_json

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    setup(args)
