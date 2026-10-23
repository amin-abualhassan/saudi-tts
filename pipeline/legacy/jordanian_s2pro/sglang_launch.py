"""Prepare an isolated SGLang runtime before switching the voice studio."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import uuid

from common import HERE, atomic_json, file_hash, identity, read_json
from sglang_export import prepare_exports
from sglang_worker import docker_mounts
from voice_server import selected_bundle


def image_tag():
    inputs = ("sglang.Dockerfile", "sglang-overrides.txt", "sglang_build_check.py", ".dockerignore")
    return "jordanian-s2pro-sglang:" + identity({name: file_hash(HERE / name) for name in inputs})[:16]


def ensure_image():
    if not shutil.which("docker"):
        raise RuntimeError("Docker is required. Install Docker with NVIDIA Container Toolkit on the host, then run this launcher again")
    subprocess.run(["docker", "info"], check=True, stdout=subprocess.DEVNULL)
    tag = image_tag()
    found = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", tag],
                           capture_output=True, text=True)
    if found.returncode:
        print("Building the isolated SGLang image; the first build downloads CUDA dependencies", flush=True)
        subprocess.run(["docker", "build", "--progress=plain", "-f", str(HERE / "sglang.Dockerfile"),
                        "-t", tag, str(HERE)], check=True)
    # Use immutable local image ID for all engines and record it in receipts.
    return subprocess.check_output(["docker", "image", "inspect", "--format", "{{.Id}}", tag], text=True).strip()


def prepare_nvme_docker():
    from sglang_docker_storage import DOCKER_DEST, plan
    if os.environ.get("DOCKER_HOST") not in (None, "", "unix:///var/run/docker.sock", "unix:///run/docker.sock") or os.environ.get("DOCKER_CONTEXT") not in (None, "", "default"):
        raise RuntimeError("--nvme-docker targets this host's default Docker service; use its local default context")
    settings = plan()
    if settings["already_configured"]:
        try:
            actual = json.loads(subprocess.check_output(["docker", "info", "--format", "{{json .}}"], text=True))
            if Path(actual["DockerRootDir"]).resolve() == DOCKER_DEST:
                print("Docker and containerd storage already use NVMe", flush=True)
                return
        except subprocess.CalledProcessError:
            pass  # The privileged helper can start services after NVMe is mounted.
    print("Preparing NVMe Docker storage. sudo is needed once to migrate idle services and retain their cached files.", flush=True)
    subprocess.run(["bash", str(HERE / "setup_sglang_docker_nvme.sh")], check=True)


def check_ports(ports, host="127.0.0.1"):
    held = []
    try:
        for port in ports:
            sock = socket.socket()
            held.append(sock)
            sock.bind((host, port))
    finally:
        for sock in held:
            sock.close()


def running_server():
    """Read-only ownership check used to permit a restart's occupied ports."""
    try:
        pid = int((HERE / "voice/server.pid").read_text().strip())
        process = Path("/proc") / str(pid)
        args = (process / "cmdline").read_bytes().decode().split("\0")
        cwd = (process / "cwd").resolve()
        scripts = {HERE / "voice_server.py", HERE / "sglang_server.py"}
        script = next((cwd / a).resolve() for a in args if a.endswith(".py") and (cwd / a).resolve() in scripts)
        port = 7860
        for i, arg in enumerate(args):
            if arg == "--port":
                port = int(args[i + 1])
            elif arg.startswith("--port="):
                port = int(arg.split("=", 1)[1])
        return {"pid": pid, "script": script.name, "port": port}
    except (OSError, ValueError, StopIteration, IndexError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=3000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--check", action="store_true", help="Verify inputs and exports without Docker, CUDA or stopping the server")
    parser.add_argument("--skip-benchmark", action="store_true", help="Skip paired generation; real cloning canaries still run")
    parser.add_argument("--nvme-docker", action="store_true", help="Move idle local Docker/containerd stores to NVMe before building; needs sudo once")
    args = parser.parse_args()
    if not 1 <= args.port <= 65533 or args.gpu < 0:
        parser.error("Choose a port from 1 to 65533 and a nonnegative GPU index")
    bundle = selected_bundle(args.campaign)
    root = HERE / "sglang"
    root.mkdir(exist_ok=True)
    with (root / "setup.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if args.nvme_docker and not args.check:
            prepare_nvme_docker()
        image = None if args.check else ensure_image()
        print("Verifying/exporting the exact selected checkpoints on CPU", flush=True)
        exports = prepare_exports(bundle)
        if args.check:
            print(json.dumps({"exports": exports, "gpu_tested": False}, indent=2))
            return 0
        ports = [args.port + 1, args.port + 2]
        live = running_server()
        if not live or live["port"] != args.port:
            check_ports([args.port], args.host)
        previous = read_json(root / "manifest.json", {})
        if not live or live["script"] != "sglang_server.py" or previous.get("ports") != ports:
            check_ports(ports)
        # Do not overwrite a live worker's manifest. Preflight writes to a staging directory.
        stage = root / "preflight" / uuid.uuid4().hex
        stage.mkdir(parents=True)
        manifest = {"image": image, "exports": exports, "ports": ports,
                    "config_dir": str(stage / "config"),
                    "checkpoint_hashes": {n: m["checkpoint_sha256"] for n, m in bundle["models"].items()}}
        atomic_json(stage / "manifest.json", manifest)
        print("Checking container dependencies, CUDA driver, BF16 and engine limits before switching", flush=True)
        subprocess.run(["docker", "run", "--rm", "--gpus", f"device={args.gpu}",
            *docker_mounts(stage), image, "python3", str(HERE / "sglang_container.py"),
            "--manifest", "/runtime/manifest.json", "--output", "/runtime/config"], check=True)
        manifest["preflight"] = read_json(stage / "config/preflight.json")
        atomic_json(stage / "manifest.json", manifest)
        if args.prepare_only:
            print(f"Prepared: {stage / 'manifest.json'}. Existing voice server was not stopped.", flush=True)
            return 0
    print(f"Starting SGLang voice studio on port {args.port}. First startup compiles kernels and generates paired samples.", flush=True)
    print("The idle original studio is stopped to release GPU memory. Its launcher remains available.", flush=True)
    command = [sys.executable, str(HERE / "sglang_server.py"), "--port", str(args.port),
               "--host", args.host, "--gpu", str(args.gpu), "--campaign", bundle["campaign"],
               "--replace-voice-server", "--replace-dashboard"]
    os.environ["JORDANIAN_SGLANG_SKIP_BENCHMARK"] = "1" if args.skip_benchmark else "0"
    os.environ["JORDANIAN_SGLANG_MANIFEST"] = str(stage / "manifest.json")
    os.execv(sys.executable, command)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f"SGLang setup failed: {error}\nInspect the message above. The original launcher is start_voice_server.sh.", file=sys.stderr)
        raise SystemExit(1)

