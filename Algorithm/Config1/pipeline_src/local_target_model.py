import itertools
import queue
import threading
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, Union

from vllm import LLM, SamplingParams
from vllm.sampling_params import RequestOutputKind


class TargetGenerationJob:
    """Internal record for tracking in-flight target generations."""

    def __init__(
        self,
        job_id: str,
        prompt: str,
        sampling_params: SamplingParams,
        prompt_token_count: int,
    ) -> None:
        self.job_id = job_id
        self.prompt = prompt
        self.sampling_params = sampling_params
        self.prompt_token_count = prompt_token_count
        self.created_at = time.time()
        self.finished_at: Optional[float] = None
        self.cancelled = False
        self.error: Optional[Exception] = None

        self._lock = threading.Lock()
        self._done = threading.Event()

        self.partial_outputs: Dict[int, Dict[str, Any]] = {}
        self.finished_outputs: Dict[int, Dict[str, Any]] = {}
        self.generated_token_counts: Dict[int, int] = {}

    def update_from_request_output(self, request_output) -> None:
        with self._lock:
            for completion in request_output.outputs:
                payload = {
                    "text": completion.text,
                    "index": completion.index,
                    "stop_reason": completion.stop_reason,
                    "finish_reason": completion.finish_reason,
                    "token_ids": list(completion.token_ids),
                }
                self.partial_outputs[completion.index] = payload
                if completion.finish_reason is not None:
                    self.finished_outputs[completion.index] = payload
                    self.generated_token_counts[completion.index] = len(
                        completion.token_ids
                    )

            if request_output.finished:
                self.finished_at = time.time()
                self._done.set()

    def mark_cancelled(self) -> None:
        with self._lock:
            self.cancelled = True
            self.finished_at = time.time()
            self._done.set()

    def mark_error(self, exc: Exception) -> None:
        with self._lock:
            self.error = exc
            self.finished_at = time.time()
            self._done.set()

    @property
    def finished(self) -> bool:
        return self.finished_at is not None

    def wait(self, timeout: Optional[float] = None) -> bool:
        return self._done.wait(timeout)

    def snapshot(
        self,
        *,
        include_partial: bool,
        include_finished: bool,
    ) -> List[SimpleNamespace]:
        with self._lock:
            entries: Dict[int, Dict[str, Any]] = {}
            if include_partial:
                entries.update(self.partial_outputs)
            if include_finished:
                entries.update(self.finished_outputs)

            snapshots: List[SimpleNamespace] = []
            for idx in sorted(entries.keys()):
                payload = entries[idx]
                snapshots.append(
                    SimpleNamespace(
                        text=payload["text"],
                        index=payload["index"],
                        stop_reason=payload["stop_reason"],
                        finish_reason=payload["finish_reason"],
                        token_ids=list(payload["token_ids"]),
                    )
                )
            return snapshots

    def output_token_count(self, include_partial: bool = False) -> int:
        with self._lock:
            if self.finished_outputs:
                return sum(
                    len(payload["token_ids"]) for payload in self.finished_outputs.values()
                )
            if include_partial:
                return sum(
                    len(payload["token_ids"]) for payload in self.partial_outputs.values()
                )
            return 0

    def elapsed_time(self) -> float:
        end_time = self.finished_at or time.time()
        return max(end_time - self.created_at, 0.0)

    def max_output_token_length(self, include_partial: bool = False) -> int:
        with self._lock:
            candidates = self.finished_outputs
            if not candidates and include_partial:
                candidates = self.partial_outputs
            if not candidates:
                return 0
            return max(len(payload["token_ids"]) for payload in candidates.values())


class LocalTargetCompletions:
    """Wrapper matching the OpenAI-style completions.create(...) signature."""

    def __init__(self, target_model: "LocalTargetModel"):
        self._target_model = target_model

    def create(self, **kwargs: Any):
        return self._target_model.completions_create(**kwargs)


class LocalTargetModel:
    """Local target model wrapper with asynchronous prefetch support."""

    def __init__(
        self,
        model_name_or_path: str,
        tokenizer=None,
        device: str = "cuda",
    ) -> None:
        self.device = device
        self.model_name = model_name_or_path.split("/")[-1]

        self.model = LLM(
            model=model_name_or_path,
            trust_remote_code=True,
            dtype="float16",
            tensor_parallel_size=1,
        )

        if tokenizer is None:
            tokenizer = self.model.get_tokenizer()
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.completions = LocalTargetCompletions(self)
        self._request_counter = itertools.count()
        self._command_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._jobs_lock = threading.Lock()
        self._jobs: Dict[str, TargetGenerationJob] = {}
        self._completed_jobs: Dict[str, TargetGenerationJob] = {}
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="LocalTargetModelWorker",
            daemon=True,
        )
        self._worker_thread.start()

    # ------------------------------------------------------------------
    # Public API (OpenAI compatible)
    # ------------------------------------------------------------------
    def completions_create(
        self,
        *,
        model: Optional[str] = None,
        prompt: Union[str, List[str]] = "",
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 100,
        n: int = 1,
        stop: Optional[Union[str, List[str]]] = None,
        **_: Any,
    ):
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        stop_tokens = stop if isinstance(stop, list) else [stop] if stop else None

        job_ids: List[str] = []
        for prompt_text in prompts:
            params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stop=stop_tokens,
                n=n,
                output_kind=RequestOutputKind.CUMULATIVE,
            )
            job_ids.append(self._schedule_job(prompt_text, params))

        choices: List[SimpleNamespace] = []
        accumulated_prompt_tokens = 0
        accumulated_output_tokens = 0
        for prompt_idx, job_id in enumerate(job_ids):
            job = self._wait_for_completion(job_id)
            if job.error:
                raise RuntimeError("Target generation failed") from job.error

            outputs = job.snapshot(include_partial=False, include_finished=True)
            outputs.sort(key=lambda item: item.index)
            accumulated_prompt_tokens += job.prompt_token_count
            accumulated_output_tokens += job.output_token_count()

            for completion_idx, completion in enumerate(outputs):
                choices.append(
                    SimpleNamespace(
                        text=completion.text,
                        index=prompt_idx * n + completion_idx,
                        stop_reason=completion.stop_reason,
                        finish_reason=completion.finish_reason,
                    )
                )
            self._cleanup_job(job_id)

        response = SimpleNamespace(choices=choices)
        response.usage = {
            "prompt_tokens": accumulated_prompt_tokens,
            "completion_tokens": accumulated_output_tokens,
            "total_tokens": accumulated_prompt_tokens + accumulated_output_tokens,
        }
        return response

    # ------------------------------------------------------------------
    # Public API (prefetch helpers)
    # ------------------------------------------------------------------
    def start_async_generation(
        self,
        *,
        prompts: Union[str, List[str]],
        temperature: float,
        top_p: float,
        max_tokens: int,
        n: int,
        stop: Optional[List[str]] = None,
    ) -> List[str]:
        prompt_list = [prompts] if isinstance(prompts, str) else list(prompts)
        job_ids: List[str] = []
        for prompt in prompt_list:
            params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stop=stop,
                n=n,
                output_kind=RequestOutputKind.CUMULATIVE,
            )
            job_ids.append(self._schedule_job(prompt, params))
        return job_ids

    def collect_async_generation(
        self,
        job_id: str,
        *,
        wait: bool = True,
        timeout: Optional[float] = None,
        allow_partial: bool = False,
        remove: bool = True,
    ) -> Tuple[List[SimpleNamespace], Dict[str, Any]]:
        job = self._get_job(job_id)
        if job is None:
            return [], {"status": "missing"}

        if wait and not job.finished:
            job.wait(timeout)

        if job.finished:
            outputs = job.snapshot(include_partial=True, include_finished=True)
            meta = self._build_metadata(job, include_partial=False)
            if remove:
                self._cleanup_job(job_id)
            return outputs, meta

        if allow_partial:
            outputs = job.snapshot(include_partial=True, include_finished=False)
            meta = self._build_metadata(job, include_partial=True)
            return outputs, meta

        return [], {"status": "running"}

    def stop_async_generation(
        self,
        job_id: str,
        *,
        save_partial: bool = True,
    ) -> Tuple[List[SimpleNamespace], Dict[str, Any]]:
        job = self._get_job(job_id)
        if job is None:
            return [], {"status": "missing"}

        if not job.finished:
            self._command_queue.put({"kind": "abort", "job_id": job_id})
            job.wait()

        include_partial = save_partial or not job.finished_outputs
        outputs = job.snapshot(
            include_partial=include_partial,
            include_finished=True,
        )
        meta = self._build_metadata(job, include_partial=include_partial)
        self._cleanup_job(job_id)
        return outputs, meta

    def request_async_stop(
        self,
        job_id: str,
        *,
        save_partial: bool = True,
    ) -> Tuple[List[SimpleNamespace], Dict[str, Any]]:
        job = self._get_job(job_id)
        if job is None:
            return [], {"status": "missing"}

        if job.finished:
            include_partial = save_partial or not job.finished_outputs
            outputs = job.snapshot(
                include_partial=include_partial,
                include_finished=True,
            )
            meta = self._build_metadata(job, include_partial=include_partial)
            self._cleanup_job(job_id)
            return outputs, meta

        self._command_queue.put({
            "kind": "abort",
            "job_id": job_id,
            "save_partial": save_partial,
        })
        return [], {"status": "pending"}

    def is_async_job_active(self, job_id: str) -> bool:
        with self._jobs_lock:
            return job_id in self._jobs

    def shutdown(self) -> None:
        if self._stop_event.is_set():
            return
        self._stop_event.set()
        self._command_queue.put({"kind": "shutdown"})
        self._worker_thread.join(timeout=5)

    def __del__(self) -> None:
        try:
            self.shutdown()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _schedule_job(self, prompt: str, sampling_params: SamplingParams) -> str:
        job_id = f"target-{next(self._request_counter)}"
        prompt_tokens = len(self.tokenizer.encode(prompt)) if self.tokenizer else 0
        job = TargetGenerationJob(job_id, prompt, sampling_params, prompt_tokens)

        with self._jobs_lock:
            self._jobs[job_id] = job

        self._command_queue.put({"kind": "add", "job": job})
        return job_id

    def _wait_for_completion(self, job_id: str) -> TargetGenerationJob:
        job = self._get_job(job_id)
        if job is None:
            raise RuntimeError(f"Unknown job_id {job_id}")
        job.wait()
        return job

    def _get_job(self, job_id: str) -> Optional[TargetGenerationJob]:
        with self._jobs_lock:
            job = self._jobs.get(job_id)
            if job is None:
                job = self._completed_jobs.get(job_id)
        return job

    def _cleanup_job(self, job_id: str) -> None:
        with self._jobs_lock:
            self._jobs.pop(job_id, None)
            self._completed_jobs.pop(job_id, None)

    def _build_metadata(
        self,
        job: TargetGenerationJob,
        *,
        include_partial: bool,
    ) -> Dict[str, Any]:
        status = "cancelled" if job.cancelled else "finished"
        if job.error is not None:
            status = "error"
        return {
            "status": status,
            "prompt_tokens": job.prompt_token_count,
            "output_tokens": job.output_token_count(include_partial=include_partial),
            "generation_time": job.elapsed_time(),
            "error": job.error,
            "expansion_factor": getattr(job.sampling_params, "n", 1),
            "num_outputs": len(job.finished_outputs)
            if job.finished_outputs
            else len(job.partial_outputs),
            "max_output_tokens": job.max_output_token_length(include_partial=include_partial),
            "batch_size": 1,
        }

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            processed_command = self._drain_commands()

            if self._stop_event.is_set():
                break

            if self._has_active_jobs():
                self._step_engine()
                continue

            if not processed_command:
                try:
                    command = self._command_queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                self._handle_command(command)

    def _has_active_jobs(self) -> bool:
        with self._jobs_lock:
            return bool(self._jobs)

    def _drain_commands(self) -> bool:
        processed = False
        while True:
            try:
                command = self._command_queue.get_nowait()
            except queue.Empty:
                break
            self._handle_command(command)
            processed = True
        return processed

    def _handle_command(self, command: Dict[str, Any]) -> None:
        kind = command.get("kind")

        if kind == "add":
            job: TargetGenerationJob = command["job"]
            try:
                self.model.llm_engine.add_request(
                    job.job_id,
                    job.prompt,
                    job.sampling_params,
                )
            except Exception as exc:
                job.mark_error(exc)
                with self._jobs_lock:
                    self._completed_jobs[job.job_id] = job
                    self._jobs.pop(job.job_id, None)

        elif kind == "abort":
            job_id = command["job_id"]
            save_partial = command.get("save_partial", True)
            job = self._get_job(job_id)
            if job is None:
                return
            try:
                self.model.llm_engine.abort_request([job.job_id])
            except Exception as exc:
                job.mark_error(exc)
            else:
                job.mark_cancelled()
                if not save_partial:
                    job.partial_outputs.clear()
                    job.finished_outputs.clear()
            with self._jobs_lock:
                self._jobs.pop(job.job_id, None)
                self._completed_jobs[job.job_id] = job

        elif kind == "shutdown":
            self._stop_event.set()

    def _step_engine(self) -> None:
        try:
            outputs = self.model.llm_engine.step()
        except Exception as exc:
            with self._jobs_lock:
                active_jobs = list(self._jobs.values())
                self._jobs.clear()
                for job in active_jobs:
                    job.mark_error(exc)
                    self._completed_jobs[job.job_id] = job
            return

        for output in outputs:
            job = self._locate_job(output.request_id)
            if job is None:
                continue
            job.update_from_request_output(output)
            if output.finished:
                with self._jobs_lock:
                    self._jobs.pop(job.job_id, None)
                    self._completed_jobs[job.job_id] = job

    def _locate_job(self, request_id: str) -> Optional[TargetGenerationJob]:
        with self._jobs_lock:
            job = self._jobs.get(request_id)
            if job is None and "_" in request_id:
                parent_id = request_id.split("_", 1)[1]
                job = self._jobs.get(parent_id)
        return job
