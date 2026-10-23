"""Move idle local Docker/containerd stores to NVMe; retain originals and rollback."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import time

try:
    import tomllib
except ImportError:
    import tomli as tomllib

ROOT = Path("/opt/dlami/nvme")
STORAGE = ROOT / "sglang_docker_storage"
DOCKER_DEST = STORAGE / "docker"
CONTAINERD_DEST = STORAGE / "containerd"
DOCKER_CONFIG = Path("/etc/docker/daemon.json")
CONTAINERD_CONFIG = Path("/etc/containerd/config.toml")
MOUNT_GUARDS = (
    Path("/etc/systemd/system/docker.service.d/sglang-nvme-storage.conf"),
    Path("/etc/systemd/system/containerd.service.d/sglang-nvme-storage.conf"),
)
MIN_FREE = 200 * 2**30


def load_settings():
    docker_text = DOCKER_CONFIG.read_text() if DOCKER_CONFIG.exists() else "{}\n"
    containerd_text = CONTAINERD_CONFIG.read_text() if CONTAINERD_CONFIG.exists() else ""
    docker = json.loads(docker_text)
    containerd = tomllib.loads(containerd_text)
    return docker_text, containerd_text, docker, containerd


def containerd_config_with_root(text, destination):
    """Change only the top-level root; retain plugin/runtime configuration."""
    table = re.search(r"(?m)^[ \t]*\[", text)
    boundary = table.start() if table else len(text)
    header, rest = text[:boundary], text[boundary:]
    line = "root = " + json.dumps(str(destination))
    pattern = r"(?m)^[ \t]*root[ \t]*=.*$"
    if re.search(pattern, header):
        header = re.sub(pattern, lambda _: line, header)
    else:
        header = line + "\n" + header
    updated = header + rest
    parsed = tomllib.loads(updated)
    if parsed.get("root") != str(destination):
        raise ValueError("Failed to set containerd storage root")
    return updated


def plan():
    docker_text, containerd_text, docker, containerd = load_settings()
    docker_source = Path(docker.get("data-root", "/var/lib/docker")).resolve()
    containerd_source = Path(containerd.get("root", "/var/lib/containerd")).resolve()
    updated_docker = dict(docker, **{"data-root": str(DOCKER_DEST)})
    return {"docker_source": str(docker_source), "containerd_source": str(containerd_source),
            "docker_destination": str(DOCKER_DEST), "containerd_destination": str(CONTAINERD_DEST),
            "nvme_free_bytes": shutil.disk_usage(ROOT).free,
            "already_configured": docker_source == DOCKER_DEST and containerd_source == CONTAINERD_DEST,
            "docker_original": docker_text, "containerd_original": containerd_text,
            "docker_updated": json.dumps(updated_docker, indent=2) + "\n",
            "containerd_updated": containerd_config_with_root(containerd_text, CONTAINERD_DEST)}


def validate_disk(root=ROOT, minimum=MIN_FREE):
    if not root.is_mount() or root.stat().st_dev == Path("/").stat().st_dev:
        raise RuntimeError("NVMe must be mounted separately at /opt/dlami/nvme; refusing to fill the system disk")
    free = shutil.disk_usage(root).free
    if free < minimum:
        raise RuntimeError(f"NVMe needs at least {minimum / 2**30:.0f} GiB free; available {free / 2**30:.1f} GiB")
    return free


def local_environment():
    env = os.environ.copy()
    env.pop("DOCKER_CONTEXT", None)
    env["DOCKER_HOST"] = "unix:///var/run/docker.sock"
    return env


def output(command):
    return subprocess.check_output(command, text=True, env=local_environment(), timeout=30).strip()


def check_idle(containerd_settings):
    if output(["docker", "ps", "--quiet"]):
        raise RuntimeError("Docker has running containers. Storage was not changed; let those workloads finish first")
    # Restarting Docker can start an 'always' container that was manually stopped.
    containers = output(["docker", "ps", "--all", "--quiet"]).split()
    if containers:
        policies = output(["docker", "inspect", "--format", "{{.HostConfig.RestartPolicy.Name}}", *containers]).splitlines()
        if any(policy == "always" for policy in policies):
            raise RuntimeError("A stopped container has restart policy 'always'; automatic storage migration would start it. Storage was not changed")
    address = containerd_settings.get("grpc", {}).get("address", "/run/containerd/containerd.sock")
    namespaces = output(["ctr", "--address", address, "namespaces", "list", "--quiet"]).split()
    for namespace in namespaces:
        tasks = output(["ctr", "--address", address, "--namespace", namespace, "tasks", "list", "--quiet"])
        if tasks:
            raise RuntimeError(f"containerd has running tasks in {namespace}; storage was not changed")


def atomic_text(path, text, mode=None):
    path = Path(path)
    if mode is None:
        mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o600
    temporary = path.with_name(path.name + ".sglang-tmp")
    try:
        with temporary.open("w") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def restart_services():
    subprocess.run(["systemctl", "start", "containerd.service"], check=True)
    subprocess.run(["systemctl", "start", "docker.socket", "docker.service"], check=True)


def install():
    if os.geteuid() != 0:
        raise RuntimeError("Storage migration needs sudo to stop idle Docker, copy its files and update its configuration")
    for binary in ("docker", "dockerd", "containerd", "ctr", "rsync", "systemctl"):
        if not shutil.which(binary):
            raise RuntimeError(f"Required command is missing: {binary}")
    validate_disk()
    STORAGE.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (STORAGE / "migration.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        current = plan()
        if current["already_configured"]:
            restart_services()  # Starts inactive units; does not restart active services.
        info = json.loads(output(["docker", "info", "--format", "{{json .}} "]))
        if current["already_configured"]:
            if Path(info["DockerRootDir"]).resolve() != DOCKER_DEST:
                raise RuntimeError("Configuration is on NVMe but the running Docker service has a different root; inspect the existing migration before restarting")
            print("Docker and containerd storage are already configured on NVMe", flush=True)
            return
        if Path(info["DockerRootDir"]).resolve() != Path(current["docker_source"]):
            raise RuntimeError("Docker's running root differs from daemon.json; automatic migration cannot override its command-line configuration")
        _, _, docker_settings, containerd_settings = load_settings()
        if docker_settings.get("features", {}).get("containerd-in-dockerd"):
            raise RuntimeError("This helper targets the standalone containerd store shown in the error")
        for service, flag, config in (("docker.service", "--data-root", DOCKER_CONFIG),
                                       ("containerd.service", "--root", CONTAINERD_CONFIG)):
            command = output(["systemctl", "show", service, "--property=ExecStart", "--value"])
            if flag in command:
                raise RuntimeError(f"{service} sets {flag} on its command line; inspect that service before migration")
            custom = re.search(r"(?:--config(?:-file)?|-c)(?:=|[ \t]+)([^\s;]+)", command)
            if custom and Path(custom.group(1).strip("\"'")).resolve() != config.resolve():
                raise RuntimeError(f"{service} uses a different configuration file; automatic migration would not affect it")
        check_idle(containerd_settings)
        allocated = 0
        for key in ("docker_source", "containerd_source"):
            source = Path(current[key])
            if source.exists():
                allocated += int(output(["du", "-sx", "--block-size=1", str(source)]).split()[0])
        if shutil.disk_usage(ROOT).free < allocated + MIN_FREE:
            raise RuntimeError("NVMe needs space for both existing stores plus 200 GiB for the image build; services were not stopped")
        sources = {key: current[key] for key in ("docker_source", "containerd_source")}
        receipt_path = STORAGE / "migration.json"
        prior = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
        for destination in (DOCKER_DEST, CONTAINERD_DEST):
            if destination.exists() and any(destination.iterdir()) and prior.get("sources") != sources:
                raise RuntimeError(f"Destination contains unverified data: {destination}; refusing to overwrite it")
        backup = STORAGE / ("backup-" + time.strftime("%Y%m%d-%H%M%S"))
        backup.mkdir(mode=0o700)
        mount_guard = f"[Unit]\nRequiresMountsFor={ROOT}\n\n[Service]\nExecStartPre=/usr/bin/mountpoint -q {ROOT}\n"
        prior_guards = {path: path.read_text() if path.exists() else None for path in MOUNT_GUARDS}
        if any(text is not None and text != mount_guard for text in prior_guards.values()):
            raise RuntimeError("A different NVMe service override already exists; refusing to overwrite it")
        original_exists = {"docker": DOCKER_CONFIG.exists(), "containerd": CONTAINERD_CONFIG.exists()}
        (backup / "daemon.json").write_text(current["docker_original"])
        (backup / "containerd.toml").write_text(current["containerd_original"])
        atomic_text(receipt_path, json.dumps({"sources": sources, "status": "copying", "backup": str(backup)}, indent=2))
        stopped = False
        try:
            # Stop socket activation too, so inspection cannot restart Docker during the copy.
            print("Stopping idle Docker/containerd and copying their existing stores to NVMe", flush=True)
            stopped = True
            subprocess.run(["systemctl", "stop", "docker.socket", "docker.service"], check=True)
            subprocess.run(["systemctl", "stop", "containerd.service"], check=True)
            for source_key, destination in (("docker_source", DOCKER_DEST), ("containerd_source", CONTAINERD_DEST)):
                source = Path(current[source_key])
                if source == destination or source in destination.parents or destination in source.parents:
                    raise RuntimeError("Source and destination storage trees overlap")
                # A prior incomplete copy is retained separately, never merged with a newer DB.
                if destination.exists() and any(destination.iterdir()):
                    destination.rename(backup / (destination.name + "-previous-copy"))
                destination.mkdir(mode=0o700, exist_ok=True)
                if source.exists():
                    subprocess.run(["rsync", "-aHAXS", "--numeric-ids", "--info=progress2",
                                    str(source) + "/", str(destination) + "/"], check=True)
            DOCKER_CONFIG.parent.mkdir(parents=True, exist_ok=True)
            CONTAINERD_CONFIG.parent.mkdir(parents=True, exist_ok=True)
            docker_mode = stat.S_IMODE(DOCKER_CONFIG.stat().st_mode) if DOCKER_CONFIG.exists() else 0o644
            containerd_mode = stat.S_IMODE(CONTAINERD_CONFIG.stat().st_mode) if CONTAINERD_CONFIG.exists() else 0o644
            atomic_text(DOCKER_CONFIG, current["docker_updated"], docker_mode)
            atomic_text(CONTAINERD_CONFIG, current["containerd_updated"], containerd_mode)
            subprocess.run(["dockerd", "--validate", "--config-file", str(DOCKER_CONFIG)], check=True)
            # Parse the actual containerd configuration with its installed binary.
            subprocess.run(["containerd", "--config", str(CONTAINERD_CONFIG), "config", "dump"],
                           check=True, stdout=subprocess.DEVNULL)
            for guard in MOUNT_GUARDS:
                guard.parent.mkdir(parents=True, exist_ok=True)
                atomic_text(guard, mount_guard, 0o644)
            subprocess.run(["systemctl", "daemon-reload"], check=True)
            restart_services()
            verified = json.loads(output(["docker", "info", "--format", "{{json .}} "]))
            if Path(verified["DockerRootDir"]).resolve() != DOCKER_DEST:
                raise RuntimeError("Docker restarted with the wrong storage root")
            parsed = tomllib.loads(output(["containerd", "--config", str(CONTAINERD_CONFIG), "config", "dump"]))
            if Path(parsed["root"]).resolve() != CONTAINERD_DEST:
                raise RuntimeError("containerd restarted with the wrong configured storage root")
            atomic_text(receipt_path, json.dumps({"sources": sources, "status": "complete", "backup": str(backup),
                "docker_destination": str(DOCKER_DEST), "containerd_destination": str(CONTAINERD_DEST),
                "original_stores_retained": True}, indent=2))
            print("Docker and containerd now use NVMe. Original stores and configuration backups are retained.", flush=True)
        except BaseException:
            if stopped:
                print("Migration failed; restoring the original configuration and services", flush=True)
                subprocess.run(["systemctl", "stop", "docker.socket", "docker.service", "containerd.service"], check=False)
                for key, path, text in (("docker", DOCKER_CONFIG, current["docker_original"]),
                                        ("containerd", CONTAINERD_CONFIG, current["containerd_original"])):
                    if original_exists[key]:
                        atomic_text(path, text)
                    else:
                        path.unlink(missing_ok=True)
                for guard, text in prior_guards.items():
                    if text is None:
                        guard.unlink(missing_ok=True)
                    else:
                        atomic_text(guard, text, 0o644)
                subprocess.run(["systemctl", "daemon-reload"], check=True)
                restart_services()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    if args.install:
        def stopped(*_):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, stopped)
        install()
    else:
        values = plan()
        # Configuration backups can contain proxy credentials; never print their contents.
        print(json.dumps({k: v for k, v in values.items() if not k.endswith(("_original", "_updated"))}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as error:
        raise SystemExit(f"Docker NVMe setup failed: {error}")
