#!/usr/bin/env python3
"""Stop only CPU preparation workers writing the original dataset directory."""
import fcntl
import os
from pathlib import Path
import signal
import time

HERE = Path(__file__).resolve().parent
TARGET = HERE / 'prepare_dataset.py'
OUTPUT = (HERE / '../wesam_dataset').resolve()


def matches(arguments, cwd):
    if not any((Path(a) if a.startswith('/') else cwd / a).resolve() == TARGET
               for a in arguments[1:] if a.endswith('prepare_dataset.py')):
        return False
    if '--device' not in arguments or arguments[arguments.index('--device') + 1] != 'cpu':
        return False
    if '--out' in arguments:
        value = Path(arguments[arguments.index('--out') + 1])
        output = (value if value.is_absolute() else cwd / value).resolve()
        if output != OUTPUT:
            return False
    return True


def locked():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / '.prepare.lock').open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return False
        except BlockingIOError:
            return True


def main():
    workers = []
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():
            continue
        try:
            args = [a.decode() for a in (directory / 'cmdline').read_bytes().split(b'\0') if a]
            if args and matches(args, (directory / 'cwd').resolve()):
                workers.append(int(directory.name))
        except (OSError, UnicodeError, IndexError):
            continue
    for pid in workers:
        print(f'Stopping CPU dataset worker PID {pid}', flush=True)
        try:
            os.kill(pid, signal.SIGINT)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 20
    while locked() and workers and time.monotonic() < deadline:
        time.sleep(.5)
    if locked():
        for pid in workers:
            # Verify identity again before escalating the signal.
            try:
                directory = Path('/proc') / str(pid)
                args = [a.decode() for a in (directory / 'cmdline').read_bytes().split(b'\0') if a]
                if matches(args, (directory / 'cwd').resolve()):
                    os.kill(pid, signal.SIGTERM)
            except (OSError, UnicodeError, IndexError):
                pass
        deadline = time.monotonic() + 10
        while locked() and workers and time.monotonic() < deadline:
            time.sleep(.5)
    if locked():
        raise RuntimeError('CPU output lock is still held. Run this launcher in the host terminal, '
                           'where the existing worker is visible.')
    print('CPU output lock is free.', flush=True)


if __name__ == '__main__':
    main()
