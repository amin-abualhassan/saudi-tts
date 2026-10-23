"""Storage migration guards, configuration preservation and recovery, without root."""
import json
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sglang_docker_storage as storage


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.docker = self.root / "etc/docker/daemon.json"
        self.containerd = self.root / "etc/containerd/config.toml"
        self.docker.parent.mkdir(parents=True)
        self.containerd.parent.mkdir(parents=True)
        self.source = self.root / "original"
        self.source.mkdir()
        self.original_docker = json.dumps({"data-root": str(self.source / "docker"),
            "runtimes": {"nvidia": {"path": "nvidia-container-runtime", "args": []}}, "log-driver": "local"})
        self.original_containerd = f'root = "{self.source / "containerd"}"\ndisabled_plugins = ["cri"]\n[grpc]\naddress = "/run/containerd/containerd.sock"\n'
        self.docker.write_text(self.original_docker)
        self.containerd.write_text(self.original_containerd)
        self.docker.chmod(0o644)
        self.containerd.chmod(0o644)
        self.destination = self.root / "nvme/storage"
        self.patches = []
        for key, value in (("ROOT", self.root), ("STORAGE", self.destination),
                           ("DOCKER_DEST", self.destination / "docker"),
                           ("CONTAINERD_DEST", self.destination / "containerd"),
                           ("DOCKER_CONFIG", self.docker), ("CONTAINERD_CONFIG", self.containerd)):
            item = patch.object(storage, key, value)
            item.start()
            self.patches.append(item)
        item = patch.object(storage, "MOUNT_GUARDS", (self.root / "docker.guard", self.root / "containerd.guard"))
        item.start()
        self.patches.append(item)

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    def test_plan_preserves_nvidia_and_containerd_plugin_root(self):
        self.containerd.write_text(self.original_containerd + '[plugins.example]\nroot = "/plugin-root"\n')
        plan = storage.plan()
        docker = json.loads(plan["docker_updated"])
        containerd = storage.tomllib.loads(plan["containerd_updated"])
        self.assertEqual(docker["runtimes"], json.loads(self.original_docker)["runtimes"])
        self.assertEqual(docker["log-driver"], "local")
        self.assertEqual(containerd["root"], str(self.destination / "containerd"))
        self.assertEqual(containerd["plugins"]["example"]["root"], "/plugin-root")
        self.assertEqual(containerd["disabled_plugins"], ["cri"])
        self.assertEqual(self.docker.read_text(), self.original_docker)

    def test_atomic_update_retains_read_permissions(self):
        storage.atomic_text(self.docker, "updated")
        self.assertEqual(stat.S_IMODE(self.docker.stat().st_mode), 0o644)
        self.assertFalse(self.docker.with_name(self.docker.name + ".sglang-tmp").exists())

    def test_running_containers_are_rejected_before_service_calls(self):
        with patch.object(storage, "output", return_value="running-container"), \
                patch.object(storage.subprocess, "run") as execute:
            with self.assertRaisesRegex(RuntimeError, "running containers"):
                storage.check_idle({})
            execute.assert_not_called()

    def test_stopped_always_container_and_foreign_containerd_task_are_rejected(self):
        for outputs, error in ((["", "stopped-container", "always"], "restart policy"),
                               (["", "", "k8s.io", "running-task"], "running tasks")):
            with patch.object(storage, "output", side_effect=outputs), self.assertRaisesRegex(RuntimeError, error):
                storage.check_idle({})

    def test_missing_nvme_mount_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "mounted separately"):
            storage.validate_disk(self.root)

    def test_rsync_failure_restores_original_configuration_and_services(self):
        (self.source / "docker").mkdir()
        (self.source / "containerd").mkdir()
        def output(args):
            if args[0] == "du":
                return "1024 directory"
            if args[:2] == ["docker", "info"]:
                return json.dumps({"DockerRootDir": str(self.source / "docker")})
            return ""
        calls = []
        def run(args, **kwargs):
            calls.append(args)
            if args[0] == "rsync":
                raise subprocess.CalledProcessError(1, args)
            return subprocess.CompletedProcess(args, 0)
        with patch.object(storage.os, "geteuid", return_value=0), \
                patch.object(storage.shutil, "which", return_value="command"), \
                patch.object(storage, "validate_disk", return_value=10**12), \
                patch.object(storage.shutil, "disk_usage", return_value=(0, 0, 10**12)) as usage, \
                patch.object(storage, "output", side_effect=output), \
                patch.object(storage.subprocess, "run", side_effect=run), \
                patch.object(storage, "restart_services") as restart:
            usage.return_value = type("Usage", (), {"free": 10**12})()
            with self.assertRaises(subprocess.CalledProcessError):
                storage.install()
            restart.assert_called_once()
        self.assertEqual(self.docker.read_text(), self.original_docker)
        self.assertEqual(self.containerd.read_text(), self.original_containerd)
        self.assertTrue((self.source / "docker").exists())
        self.assertTrue((self.source / "containerd").exists())
        self.assertTrue(any(args[:2] == ["systemctl", "stop"] for args in calls))

    def run_copy(self, fail_start=False):
        for name in ("docker", "containerd"):
            directory = self.source / name
            directory.mkdir()
            (directory / "cached-data").write_bytes(name.encode())
        def output(args):
            if args[0] == "du":
                return "1024 directory"
            if args[:2] == ["docker", "info"]:
                return json.dumps({"DockerRootDir": json.loads(self.docker.read_text())["data-root"]})
            if args[0] == "containerd":
                return self.containerd.read_text()
            return ""
        real_run = subprocess.run
        def run(args, **kwargs):
            if args[0] == "rsync":
                return real_run(args, **kwargs, stdout=subprocess.DEVNULL)
            return subprocess.CompletedProcess(args, 0)
        starts = []
        def start():
            starts.append(json.loads(self.docker.read_text())["data-root"])
            if fail_start and len(starts) == 1:
                raise RuntimeError("Simulated restart failure")
        with patch.object(storage.os, "geteuid", return_value=0), \
                patch.object(storage.shutil, "which", return_value="command"), \
                patch.object(storage, "validate_disk", return_value=10**12), \
                patch.object(storage.shutil, "disk_usage", return_value=type("Usage", (), {"free": 10**12})()), \
                patch.object(storage, "output", side_effect=output), \
                patch.object(storage.subprocess, "run", side_effect=run), \
                patch.object(storage, "restart_services", side_effect=start):
            if fail_start:
                with self.assertRaisesRegex(RuntimeError, "restart failure"):
                    storage.install()
            else:
                storage.install()
        return starts

    def test_real_rsync_preserves_stores_and_commit_updates_both_roots(self):
        self.run_copy()
        self.assertEqual(json.loads(self.docker.read_text())["data-root"], str(self.destination / "docker"))
        self.assertEqual(storage.tomllib.loads(self.containerd.read_text())["root"], str(self.destination / "containerd"))
        for name in ("docker", "containerd"):
            self.assertEqual((self.destination / name / "cached-data").read_bytes(), name.encode())
            self.assertTrue((self.source / name / "cached-data").is_file())
        receipt = json.loads((self.destination / "migration.json").read_text())
        self.assertEqual(receipt["status"], "complete")
        self.assertTrue(receipt["original_stores_retained"])
        self.assertEqual(stat.S_IMODE(self.docker.stat().st_mode), 0o644)
        self.assertTrue(all("RequiresMountsFor=" in path.read_text() for path in storage.MOUNT_GUARDS))

    def test_restart_failure_restores_updated_configs_and_mount_guards(self):
        starts = self.run_copy(fail_start=True)
        self.assertEqual(len(starts), 2)
        self.assertEqual(starts[-1], str(self.source / "docker"))
        self.assertEqual(self.docker.read_text(), self.original_docker)
        self.assertEqual(self.containerd.read_text(), self.original_containerd)
        self.assertTrue(all(not path.exists() for path in storage.MOUNT_GUARDS))


if __name__ == "__main__":
    unittest.main()
