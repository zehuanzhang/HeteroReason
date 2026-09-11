#!/usr/bin/env python3

import argparse
import csv
import sys
from collections import OrderedDict
from dataclasses import dataclass

# FPGA modeling, cross validate with on-board testing results with KV port and timing scaling
FPGA_PER_TOKEN_MS = 5.74     # linear term, per generated draft token
FPGA_ATTN_COEF_MS = 0.0024   # quadratic (attention) term coefficient
MAX_DRAFT_LATENCY_MS = 2500

# Measured using 'nvidia-smi' with 500ms sampling rate
# Load_Idle is not pure idle as it take around 5~10s after decoding peak performance to reach 35W, 
#   Here we assume some average case where load idle is usually half of peak performance
PRM_7B_Load_Idle = 175 
PRM_7B_Active = 349 

Target_7B_Load_Idle = 175 
Target_7B_Active = 349 

Target_1B5_Load_Idle = 125 
Target_1B5_Active = 249 

FPGA_Power_Active = 60 # scaling to 300Mhz, 45W measured at 200MHz
FPGA_Power_Load_Idle = 60 

REQUIRED_COLUMNS = [
    "problem_ordinal", "step", "phase", "draft_latency_ms", "prm_latency_ms",
    "target_latency_ms", "draft_input_tokens", "draft_output_tokens",
    "prm_calls", "target_calls",
]

# A step whose phase is exactly this ran draft + PRM verify only; 
DRAFT_ONLY_PHASE = "draft|draft_verify"

@dataclass
class Step:
    problem_ordinal: str
    example_idx: str
    step: int
    phase: str               # e.g. "draft|draft_verify" or "…|target|target_rescue_verify"
    draft_ms: float          # measured GPU draft latency
    prm_ms: float            # PRM verify latency
    target_ms: float         # target latency; 0 when no rescue/backtrack fired
    draft_in_tokens: float
    draft_out_tokens: float
    prm_calls: int
    target_calls: int
    fpga_draft_ms: float = 0.0   # filled in by add_fpga_draft_time()

# ---------------------------------------------------------------------------
# The following use AI to format the code.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 1. read
# ---------------------------------------------------------------------------
def read_steps(path, max_draft_ms=MAX_DRAFT_LATENCY_MS, verbose=True):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        raw = [r for r in csv.DictReader(fh) if r.get("step") not in (None, "")]
    if not raw:
        sys.exit("ERROR: no data rows found in %s" % path)

    missing = [c for c in REQUIRED_COLUMNS if c not in raw[0]]
    if missing:
        sys.exit("ERROR: CSV is missing required column(s): %s" % ", ".join(missing))

    n_raw = len(raw)
    # Due to the measurement spike, to avoid unrealistic FPGA speedup, we claim the draft latency to a certain threshold.
    # In most of cases, as draft model is small, the draft_latency_ms won't be very high.
    raw = [r for r in raw if float(r["draft_latency_ms"]) <= max_draft_ms]
    if not raw:
        sys.exit("ERROR: every row was dropped by the draft_latency_ms <= %.0f ms filter"
                 % max_draft_ms)
    if verbose:
        print("draft latency filter: kept %d / %d rows (dropped %d with "
              "draft_latency_ms > %.0f ms)"
              % (len(raw), n_raw, n_raw - len(raw), max_draft_ms))

    return [
        Step(
            problem_ordinal=r["problem_ordinal"],
            example_idx=r.get("example_idx", ""),
            step=int(float(r["step"])),
            phase=r["phase"].strip(),
            draft_ms=float(r["draft_latency_ms"]),
            prm_ms=float(r["prm_latency_ms"]),
            target_ms=float(r["target_latency_ms"]),
            draft_in_tokens=float(r["draft_input_tokens"]),
            draft_out_tokens=float(r["draft_output_tokens"]),
            prm_calls=int(float(r["prm_calls"])),
            target_calls=int(float(r["target_calls"])),
        )
        for r in raw
    ]

# ---------------------------------------------------------------------------
# 2. FPGA draft time
# ---------------------------------------------------------------------------
def add_fpga_draft_time(steps):
    """Arithmetic progression by considering KV cache loading time"""
    for s in steps:
        K = s.draft_in_tokens
        L = s.draft_out_tokens
        s.fpga_draft_ms = (
            FPGA_PER_TOKEN_MS * L
            + FPGA_ATTN_COEF_MS * (L - 1.0) * (2.0 * K + L) / 2.0 # Arithmetic progression 
        )
    return steps


# ---------------------------------------------------------------------------
# 3. group by problem
# ---------------------------------------------------------------------------
def group_by_problem(steps):
    """OrderedDict: problem_ordinal -> list of its Steps, in file order."""
    groups = OrderedDict()
    for s in steps:
        groups.setdefault(s.problem_ordinal, []).append(s)
    return groups


# ---------------------------------------------------------------------------
# 4. cost models
# ---------------------------------------------------------------------------
def compute_our_or_strong(steps, draft_ms, target_overlap, args, is_fpga=True):
    last = len(steps) - 1
    costs = []
    power, target_power, draft_power, prm_power = 0.0, 0.0, 0.0, 0.0
    if (is_fpga): 
        draft_active = args.power["draft_fpga_active"]
        draft_load_idle = args.power["draft_fpga_load_idle"]
    else: 
        draft_active = args.power["draft_gpu_active"]
        draft_load_idle = args.power["draft_gpu_load_idle"]
    for i, s in enumerate(steps):
        rescue = s.target_ms > 0.0
        is_last = i == last

        if i == 0:
            front = s.draft_ms
        else:
            prev = steps[i - 1]
            prev_rescue = prev.target_ms > 0.0

            if prev_rescue:
                if target_overlap and prev.target_calls > 1:
                    if prev.prm_calls <= 0:
                        raise ValueError(
                            "multi-rescue step has prm_calls <= 0: problem=%s step=%s"
                            % (prev.problem_ordinal, prev.step)
                        )
                    prev_tail_prm = prev.prm_ms / prev.prm_calls
                else:
                    prev_tail_prm = prev.prm_ms / 2.0
                front = max(draft_ms[i], prev_tail_prm)
            else:
                front = max(draft_ms[i], prev.prm_ms)

        if not rescue:
            cost = front + (s.prm_ms if is_last else 0.0)

            ######### Calculate Power ######### 
            prev = steps[i - 1]
            draft_power += draft_active * min(front, draft_ms[i]) + draft_load_idle * max(0, front - draft_ms[i]) # Draft Power
            prm_power += args.power["prm_active"] * min(front, prev.prm_ms) + args.power["prm_load_idle"] * max(0, front - prev.prm_ms) # PRM Power
            target_power += args.power["target_load_idle"] * front # Target Power
            if is_last:
              draft_power += draft_load_idle * s.prm_ms
              prm_power += args.power["prm_active"] * s.prm_ms # PRM Power
              target_power += args.power["target_active"] * s.prm_ms # Target Power
            ###################################
        elif not target_overlap:
            half_prm = s.prm_ms / 2.0
            # The target time can be further overlaped with PRM[i-2], but here we compute a conversative speedup
            cost = front + half_prm + s.target_ms
            if is_last:
                cost += half_prm
            # Assume overlap, now power calculation here

        elif s.target_calls > 1:
            if s.prm_calls <= 0:
                raise ValueError(
                    "multi-rescue step has prm_calls <= 0: problem=%s step=%s"
                    % (s.problem_ordinal, s.step)
                )
            p = s.prm_ms / s.prm_calls
            t = s.target_ms / s.target_calls

            cost = max(front + p, t)
            ######### Calculate Power ######### 
            draft_power += (
                draft_load_idle * t
                + draft_active * max(0.0, front + p - t)
            )
            target_power += (
                args.power["target_active"] * t
                + args.power["target_load_idle"] * max(0.0, front + p - t)
            )
            prm_power += (
                args.power["prm_load_idle"] * t
                + args.power["prm_active"] * max(0.0, front + p - t)
            )
            ###################################

            cost += (s.target_calls - 1) * t
            ######### Calculate Power ######### 
            remaining_target = (s.target_calls - 1) * t
            draft_power += draft_load_idle * remaining_target
            prm_power += args.power["prm_load_idle"] * remaining_target
            target_power += args.power["target_active"] * remaining_target
            ###################################

            cost += max(s.prm_calls - 2, 0) * p
            ######### Calculate Power ######### 
            remaining_prm = max(s.prm_calls - 2, 0) * p
            draft_power += draft_load_idle * remaining_prm
            prm_power += args.power["prm_active"] * remaining_prm
            target_power += args.power["target_load_idle"] * remaining_prm
            ###################################
            if is_last:
                cost += p
                ######### Calculate Power ######### 
                draft_power += draft_load_idle * p
                prm_power += args.power["prm_active"] * p
                target_power += args.power["target_load_idle"] * p
                ###################################

        else:
            half_prm = s.prm_ms / 2.0
            cost = max(front + half_prm, s.target_ms)
            ######### Calculate Power ######### 
            prm_side = front + half_prm
            draft_power += draft_active * prm_side # Assume draft is run even some idle cycle, to get lower bound result
            draft_power += (
                draft_load_idle
                * max(0.0, s.target_ms - prm_side)
            )
            prm_power += args.power["prm_active"] * prm_side # Current first-half PRM.
            prm_power += (
                args.power["prm_load_idle"]
                * max(0.0, s.target_ms - prm_side)
            ) # PRM idle if target is longer.
            target_power += (
                args.power["target_active"] * s.target_ms
                + args.power["target_load_idle"]
                * max(0.0, prm_side - s.target_ms)
            ) # Target active, idle if PRM side is longer.

            ###################################
            if is_last:
                cost += half_prm
                ######### Calculate Power ######### 
                draft_power += draft_load_idle * half_prm
                prm_power += args.power["prm_active"] * half_prm
                target_power += args.power["target_load_idle"] * half_prm
                ###################################

        costs.append(cost)
        power = draft_power + prm_power + target_power
    return costs, power / 1000.0 # to Joule


def compute_weak(steps):
    """Per-step cost for one problem with no overlap: draft + PRM + target."""
    return [s.draft_ms + s.prm_ms + s.target_ms for s in steps]

def get_power_weak(steps, weak_time, args):
    weak_power = 0.0
    for i, s in enumerate(steps):
        weak_power += args.power["draft_gpu_active"] * s.draft_ms + args.power["draft_gpu_load_idle"] * (weak_time[i] - s.draft_ms) # Draft power, assume always full load
        weak_power += args.power["prm_active"] * s.prm_ms + args.power["prm_load_idle"] * (weak_time[i] - s.prm_ms) #PRM power
        weak_power += args.power["target_active"] * s.target_ms + args.power["target_load_idle"] * (weak_time[i] - s.target_ms) #Target power
    return weak_power
# ---------------------------------------------------------------------------
# 5. aggregate + report
# ---------------------------------------------------------------------------
def analyse(groups, target_overlap, args):
    """Per-problem totals and speedups, plus the per-step cost breakdown."""
    per_problem = OrderedDict()
    per_step = []
    weak_power, strong_power, our_power = 0,0,0
    for pid, steps in groups.items():
        our, our_power = compute_our_or_strong(steps, [s.fpga_draft_ms for s in steps], target_overlap, args)
        strong, strong_power = compute_our_or_strong(steps, [s.draft_ms for s in steps], target_overlap, args, is_fpga=False)
        weak = compute_weak(steps)
        weak_power = get_power_weak(steps, weak, args) / 1000.0 # to Joule
        sum_our, sum_weak, sum_strong = sum(our), sum(weak), sum(strong)
        print (weak_power, strong_power, our_power)
        # speedup_vs_stong = sum_strong / sum_our if sum_our else float("nan")
        # if (speedup_vs_stong > 1.5): continue # measuring spike

        for s, o, wk, st in zip(steps, our, weak, strong):
            per_step.append((s, o, wk, st))
        
        per_problem[pid] = {
            "problem_ordinal": pid,
            "example_idx": steps[0].example_idx,
            "steps": len(steps),
            "our": sum_our,
            "weak": sum_weak,
            "strong": sum_strong,
            "our_power": our_power,
            "weak_power": weak_power,
            "strong_power": strong_power,
            "speedup_vs_weak": sum_weak / sum_our if sum_our else float("nan"),
            "speedup_vs_strong": sum_strong / sum_our if sum_our else float("nan"),
            "power_vs_weak": weak_power / our_power,
            "power_vs_strong": strong_power / our_power ,
        }

    return per_problem, per_step


def report(csv_path, per_problem, n_steps):
    ids = list(per_problem)
    tot_our = sum(a["our"] for a in per_problem.values())
    tot_weak = sum(a["weak"] for a in per_problem.values())
    tot_strong = sum(a["strong"] for a in per_problem.values())
    mean_weak = sum(per_problem[p]["speedup_vs_weak"] for p in ids) / len(ids)
    mean_strong = sum(per_problem[p]["speedup_vs_strong"] for p in ids) / len(ids)
    mean_power_weak = sum(per_problem[p]["power_vs_weak"] for p in ids) / len(ids)
    mean_power_strong = sum(per_problem[p]["power_vs_strong"] for p in ids) / len(ids)

    print("Source: %s   (%d steps, %d problems)\n" % (csv_path, n_steps, len(ids)))
    print("%-8s %-7s %6s %13s %13s %13s %15s %15s %15s %15s" % (
        "problem", "ex_idx", "steps", "our (ms)", "weak (ms)", "strong (ms)",
        "speed vs weak", "speed vs strong", "power vs weak", "power vs strong"))
    print("-" * 92)
    for p in ids:
        a = per_problem[p]
        print("%-8s %-7s %6d %13.2f %13.2f %13.2f %13.4fx %13.4fx %13.4fx %13.4fx" % (
            a["problem_ordinal"], a["example_idx"], a["steps"],
            a["our"], a["weak"], a["strong"],
            a["speedup_vs_weak"], a["speedup_vs_strong"], a["power_vs_weak"], a["power_vs_strong"]))
    print("-" * 92)
    print("%-8s %-7s %6d %13.2f %13.2f %13.2f %13.4fx %13.4fx %13.4fx %13.4fx" % (
        "TOTAL", "", n_steps, tot_our, tot_weak, tot_strong,
        tot_weak / tot_our, tot_strong / tot_our, mean_power_weak, mean_power_strong))

    print("\nOverall (pooled totals):")
    print("    our vs weak    = %.6fx" % (tot_weak / tot_our))
    print("    our vs strong  = %.6fx" % (tot_strong / tot_our))
    print("\nAverage speedup across the %d problem ordinals (unweighted mean):" % len(ids))
    print("    our vs weak    = %.6fx" % mean_weak)
    print("    our vs strong  = %.6fx" % mean_strong)

    return {"our": tot_our, "weak": tot_weak, "strong": tot_strong,
            "speedup_vs_weak": tot_weak / tot_our,
            "speedup_vs_strong": tot_strong / tot_our,
            "mean_vs_weak": mean_weak, "mean_vs_strong": mean_strong}


# ---------------------------------------------------------------------------
# 6. csv output
# ---------------------------------------------------------------------------
def write_per_problem(path, per_problem, totals, n_steps):
    cols = ["problem_ordinal", "example_idx", "steps", "our", "weak", "strong", "our_power", "weak_power", "strong_power",
            "speedup_vs_weak", "speedup_vs_strong", "power_vs_weak", "power_vs_strong"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for a in per_problem.values():
            w.writerow(a)
        w.writerow({"problem_ordinal": "TOTAL", "example_idx": "", "steps": n_steps,
                    "our": totals["our"], "weak": totals["weak"],
                    "strong": totals["strong"],
                    "speedup_vs_weak": totals["speedup_vs_weak"],
                    "speedup_vs_strong": totals["speedup_vs_strong"]})
        w.writerow({"problem_ordinal": "MEAN_OF_PROBLEMS", "example_idx": "", "steps": "",
                    "our": "", "weak": "", "strong": "",
                    "speedup_vs_weak": totals["mean_vs_weak"],
                    "speedup_vs_strong": totals["mean_vs_strong"]})


def write_per_step(path, per_step):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["problem_ordinal", "example_idx", "step", "phase",
                    "draft_latency_ms", "prm_latency_ms", "target_latency_ms",
                    "draft_input_tokens", "draft_output_tokens",
                    "fpga_draft_ms", "our", "weak", "strong"])
        for s, o, wk, st in per_step:
            w.writerow([s.problem_ordinal, s.example_idx, s.step, s.phase,
                        s.draft_ms, s.prm_ms, s.target_ms,
                        s.draft_in_tokens, s.draft_out_tokens,
                        s.fpga_draft_ms, o, wk, st])


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-csv_path", help="step-profile CSV")
    ap.add_argument("-o", "--out_prefix", default="speedup",
                    help="prefix for the two output CSVs (default: speedup)")
    ap.add_argument("-target_overlap", dest="target_overlap", action="store_true", default=False,
                    help="enable target overlap with prm and draft")
    ap.add_argument(
        "--target_size",
        choices=["7b", "1.5b"],
        default="7b",
        help="model size (default: 7b)"
    )
    args = ap.parse_args()

    if args.target_size == "7b":
        target_load_idle = Target_7B_Load_Idle
        target_active = Target_7B_Active
    else:  # 1.5b
        target_load_idle = Target_1B5_Load_Idle
        target_active = Target_1B5_Active


    args.power = {
        "prm_active": PRM_7B_Active,
        "prm_load_idle": PRM_7B_Load_Idle,

        "target_active": target_active,
        "target_load_idle": target_load_idle,

        "draft_gpu_active": PRM_7B_Active,
        "draft_gpu_load_idle": PRM_7B_Load_Idle,

        "draft_fpga_active": FPGA_Power_Active,
        "draft_fpga_load_idle": FPGA_Power_Load_Idle,
    }

    target_overlap = args.target_overlap
    print (target_overlap)
    steps = add_fpga_draft_time(read_steps(args.csv_path))
    groups = group_by_problem(steps)
    per_problem, per_step = analyse(groups, target_overlap, args)
    totals = report(args.csv_path, per_problem, len(steps))

    per_problem_path = args.out_prefix + "_per_problem.csv"
    per_step_path = args.out_prefix + "_per_step.csv"
    write_per_problem(per_problem_path, per_problem, totals, len(steps))
    write_per_step(per_step_path, per_step)
    print("\nWrote %s and %s" % (per_problem_path, per_step_path))


if __name__ == "__main__":
    main()

