import argparse
import json
import os
import runpy
import subprocess
import sys
import time
from pathlib import Path


def active_pids(device):
    result = subprocess.run(["nvidia-smi", "-i", str(device), "--query-compute-apps=pid",
                             "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    return {int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit()}


def main():
    parser = argparse.ArgumentParser(description="Wait for an assigned empty GPU, claim a CUDA context and run a local experiment queue")
    parser.add_argument("--physical-gpu", type=int, choices=range(8), required=True)
    parser.add_argument("--job-file", required=True)
    parser.add_argument("--stop-file", required=True)
    parser.add_argument("--poll-seconds", type=float, default=.25)
    parser.add_argument("--max-wait-minutes", type=float, default=180)
    parser.add_argument("--job-wait-minutes", type=float, default=30)
    args = parser.parse_args()
    if args.poll_seconds < .1:
        raise ValueError("Polling faster than 0.1 s is unnecessary")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.physical_gpu)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "4"
    import torch
    torch.set_num_threads(4)
    repo = Path(__file__).resolve().parents[1]
    job_path, stop_path = Path(args.job_file).resolve(), Path(args.stop_file).resolve()
    started = time.monotonic()
    def event(name, **values):
        print(json.dumps({"event": name, "physical_gpu": args.physical_gpu, "pid": os.getpid(),
                          "elapsed_seconds": round(time.monotonic() - started, 2), **values}), flush=True)
    event("waiting_for_empty_gpu")
    while active_pids(args.physical_gpu):
        if stop_path.exists():
            event("stopped_before_claim")
            return
        if time.monotonic() - started > args.max_wait_minutes * 60:
            raise TimeoutError("Assigned GPU did not become empty in the allowed interval")
        time.sleep(args.poll_seconds)
    if stop_path.exists():
        return
    # Import is already warm; initialize on the assigned device in this process.
    # A small real CUDA allocation makes the owner visible while the job loads.
    torch.cuda.set_device(0)
    claim = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
    claim.zero_()
    torch.cuda.synchronize()
    competing = active_pids(args.physical_gpu) - {os.getpid()}
    if competing:
        event("another_process_won_race_releasing_context", other_pids=sorted(competing))
        return
    event("claimed_cuda_context", job_file=str(job_path))
    claimed = time.monotonic()
    while not job_path.exists():
        if stop_path.exists():
            event("stopped_while_waiting_for_job")
            return
        if time.monotonic() - claimed > args.job_wait_minutes * 60:
            raise TimeoutError("No verified experiment queue arrived; releasing CUDA context")
        time.sleep(.25)
    queue = json.loads(job_path.read_text())
    if queue["physical_gpu"] != args.physical_gpu:
        raise ValueError("Queue belongs to a different physical GPU")
    del claim
    torch.cuda.empty_cache()
    os.chdir(repo)
    for job in queue["jobs"]:
        if stop_path.exists():
            event("stopped_between_jobs")
            return
        script = (repo / job["script"]).resolve()
        if not script.is_relative_to(repo / "scripts") or not script.is_file() or script.suffix != ".py":
            raise ValueError("Queue script must be an existing Python script in this repository")
        arguments = job["arguments"]
        if "--device" in arguments and arguments[arguments.index("--device") + 1] != "cuda:0":
            raise ValueError("The assigned physical GPU is exposed as logical cuda:0")
        event("running", script=job["script"], arguments=arguments)
        sys.argv = [str(script), *arguments]
        begin = time.monotonic()
        try:
            runpy.run_path(str(script), run_name="__main__")
        except SystemExit as error:
            if error.code not in (None, 0):
                raise
        event("job_completed", script=job["script"], seconds=round(time.monotonic() - begin, 2))
    event("queue_completed")


if __name__ == "__main__":
    main()
