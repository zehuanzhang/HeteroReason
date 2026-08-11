import json
import os
from transformers import AutoTokenizer

# --- 配置 ---
# SOURCE_JSONL = "outputs/outputs/bbeam1_prefetch_cache_single_local/math500/test_qwen25-math-cot_-1_seed43_t0_s0_e-1_delta0.7_maxsteps100.jsonl"
# DATASET_NAME = "math500"
# SOURCE_JSONL = "outputs/outputs/bbeam1_prefetch_cache_single_local/gsm8k/test_qwen25-math-cot_-1_seed43_t0_s0_e-1_delta0.7_maxsteps100.jsonl"
# DATASET_NAME = "gsm8k"
# SOURCE_JSONL = "outputs/outputs/bbeam1_prefetch_cache_single_local/gaokao2023en/test_qwen25-math-cot_-1_seed43_t0_s0_e-1_delta0.7_maxsteps100.jsonl"
# DATASET_NAME = "gaokao2023en"
# SOURCE_JSONL = "outputs/outputs/bbeam1_prefetch_cache_single_local/olympiadbench/test_qwen25-math-cot_-1_seed43_t0_s0_e-1_delta0.7_maxsteps100.jsonl"
# DATASET_NAME = "olympiadbench"


# math500
# SOURCE_JSONL = "outputs/outputs/bbeam1_single_local/math500/test_qwen25-math-cot_-1_seed43_t0_s0_e-1_delta0.7_maxsteps100.jsonl"
# DATASET_NAME = "math500"

# gsm8k
# SOURCE_JSONL = "outputs/outputs/bbeam1_single_local/gsm8k/test_qwen25-math-cot_-1_seed43_t0_s0_e-1_delta0.7_maxsteps100.jsonl"
# DATASET_NAME = "gsm8k"

# # gaokao2023en
# SOURCE_JSONL = "outputs/outputs/bbeam1_single_local/gaokao2023en/test_qwen25-math-cot_-1_seed43_t0_s0_e-1_delta0.7_maxsteps100.jsonl"
# DATASET_NAME = "gaokao2023en"

# # olympiadbench
# SOURCE_JSONL = "outputs/outputs/bbeam1_single_local/olympiadbench/test_qwen25-math-cot_-1_seed43_t0_s0_e-1_delta0.7_maxsteps100.jsonl"
# DATASET_NAME = "olympiadbench"













TOKENIZER_PATH = "/mnt/ccnas2/bdp/zr523/RSD/Qwen2.5-0.5B-Instruct" 

STEP_WORD = "\n\n"

def organize_data():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    output_dir = f"analysis_results/{DATASET_NAME}"
    os.makedirs(output_dir, exist_ok=True)

    with open(SOURCE_JSONL, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            idx = data['idx']
            
            # 获取记录
            raw_attempts = data.get("step_attempts", [])
            # [修改点 1] 获取回退事件列表
            backtrack_events = data.get("backtrack_events", [])
            # 原始 turn_info 记录了最终每一轮采纳的模型 ID
            turn_info = {item[0]: item[1] for item in data.get("turn_info", [])}
            
            # --- [核心调试打印] ---
            # 检查原始数据中 attempts 列表的真实长度
            print(f"DEBUG: Example {idx} | Attempts in JSONL: {len(raw_attempts)} | Final steps: {data.get('total_steps')}")
            
            detailed_steps = []
            for i, att in enumerate(raw_attempts):
                outcome = att["outcome"]
                
                # [修改点 2] 增加对 backtrack_trigger 类型的识别
                if outcome == "backtrack_trigger":
                    detailed_status = "BACKTRACK_EVENT"
                    final_model_name = "N/A"
                else:
                    # 原有的三分类逻辑
                    final_model_id = turn_info.get(att["step"], "N/A")
                    if outcome == "accept":
                        if final_model_id == 2:
                            detailed_status = "accept_via_prefetch"
                        else:
                            detailed_status = "accept_pure_draft"
                    else:
                        detailed_status = "target_rescue"
                    final_model_name = "Target" if final_model_id == 2 else "Draft"
                
                # 组装单步字典
                step_info = {
                    "event_idx": i, # 动作发生的先后顺序
                    "logical_step": att["step"], # 逻辑步索引
                    "status": detailed_status,
                    "final_model": final_model_name,
                    "draft_tokens_spent": att["draft_tokens"],
                    "target_tokens_spent": att["target_tokens"],
                    "latency": att.get("latency", 0), # 每一个动作的绝对耗时
                }

                # [修改点 3] 如果是回退，记录回退的起止点
                if detailed_status == "BACKTRACK_EVENT":
                    step_info.update({
                        "from_step": att.get("from_step"),
                        "to_step": att.get("to_step"),
                        "steps_dropped": att.get("from_step", 0) - att.get("to_step", 0)
                    })
                
                detailed_steps.append(step_info)

            # 构造该 Example 的独立数据包
            example_report = {
                "idx": idx,
                "summary": {
                    "is_correct": data['score'][0],
                    # [修改点 4] 汇总回退总次数
                    "backtrack_count": len(backtrack_events),
                    "total_tokens_breakdown": data['token_counts'] 
                },
                "execution_trace": detailed_steps
            }

            with open(f"{output_dir}/example_{idx}.json", 'w') as out_f:
                json.dump(example_report, out_f, indent=4)
    
    print(f"完成！数据已整理至 analysis_results/{DATASET_NAME}/")

if __name__ == "__main__":
    organize_data()