"""Defer an early pause signal until the unchanged target installs its handler.

This bootstrap changes signal delivery only. The original target source and
arguments execute through runpy; it never patches model or optimizer behavior.
"""
import argparse
from pathlib import Path
import runpy
import signal
import sys


class DeferredPause:
    def __init__(self, target):
        self.target = Path(target).resolve()
        self.original_signal = signal.signal
        self.previous = {}
        self.pending = False
        self.delivered = False
        self.target_handlers = {}

    def defer(self, signum, frame):
        self.pending = True

    def install(self, signum, handler):
        result = self.original_signal(signum, handler)
        code = getattr(handler, "__code__", None)
        is_target = code is not None and Path(code.co_filename).resolve() == self.target
        if signum in (signal.SIGINT, signal.SIGTERM) and is_target:
            self.target_handlers[signum] = handler
            if self.pending and not self.delivered:
                self.delivered = True
                handler(signal.SIGINT, None)
        elif signum in (signal.SIGINT, signal.SIGTERM) and signum not in self.target_handlers:
            # Retain startup protection if an imported library installs a handler.
            self.original_signal(signum, self.defer)
        return result

    def __enter__(self):
        for signum in (signal.SIGINT, signal.SIGTERM):
            self.previous[signum] = self.original_signal(signum, self.defer)
        signal.signal = self.install
        return self

    def __exit__(self, *unused):
        signal.signal = self.original_signal
        for signum, previous in self.previous.items():
            self.original_signal(signum, previous)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    target = args.program.resolve(strict=True)
    if remaining[:1] == ["--"]:
        remaining = remaining[1:]
    sys.argv = [str(target), *remaining]
    sys.path.insert(0, str(target.parent))
    with DeferredPause(target):
        runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
