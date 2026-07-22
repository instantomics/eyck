"""Command-line entry point and restartable local supervisor."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import webbrowser
from collections.abc import Callable, Sequence
from urllib.parse import urlsplit

import uvicorn

from .app import create_app


RESTART_CHILD_ENV = "EYCK_RESTART_CHILD"
RESTART_GENERATION_ENV = "EYCK_RESTART_GENERATION"
RESTART_EXIT_CODE = 75


def supervisor_loop(serve_once: Callable[[threading.Event], bool], *, restart: bool = True) -> None:
    """Run server generations until a normal stop or restart is disabled."""
    while True:
        requested = threading.Event()
        should_restart = serve_once(requested)
        if not restart or not should_restart or not requested.is_set():
            return


def serve(
    roots: list[str],
    *,
    host: str,
    port: int,
    external_origin: str | None,
    restart: bool,
    open_browser: bool,
    proxy_prefix: str,
) -> None:
    loopback = host in {"127.0.0.1", "localhost", "::1"}
    if not loopback and not external_origin:
        raise SystemExit("non-loopback binding requires --external-origin and an authenticated proxy policy")
    origin = (external_origin or f"http://{host}:{port}").rstrip("/")
    parsed = urlsplit(origin)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise SystemExit("--external-origin must be an absolute HTTP(S) origin")
    allowed_host = parsed.hostname or host
    def request_restart() -> None:
        threading.Timer(0.2, os._exit, args=(RESTART_EXIT_CODE,)).start()

    app = create_app(
        roots,
        allowed_hosts=(host, allowed_host, "localhost", "127.0.0.1"),
        allowed_origins=(origin,),
        restart_callback=request_restart if restart else None,
        proxy_prefix=proxy_prefix,
    )
    if open_browser and os.environ.get(RESTART_GENERATION_ENV, "1") == "1":
        webbrowser.open(origin)
    uvicorn.run(app, host=host, port=port, log_level="info")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eyck")
    commands = parser.add_subparsers(dest="command", required=True)
    annotations = commands.add_parser("annotations")
    annotation_commands = annotations.add_subparsers(dest="annotation_command", required=True)
    serve_parser = annotation_commands.add_parser("serve")
    serve_parser.add_argument("roots", nargs="+")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--external-origin")
    serve_parser.add_argument("--proxy-prefix", default="")
    serve_parser.add_argument("--no-restart", action="store_true")
    serve_parser.add_argument("--no-open", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_arguments = list(argv) if argv is not None else sys.argv[1:]
    arguments = build_parser().parse_args(raw_arguments)
    if arguments.command == "annotations" and arguments.annotation_command == "serve":
        if not arguments.no_restart and os.environ.get(RESTART_CHILD_ENV) != "1":
            return _restart_process_loop(raw_arguments)
        serve(
            arguments.roots,
            host=arguments.host,
            port=arguments.port,
            external_origin=arguments.external_origin,
            restart=not arguments.no_restart,
            open_browser=not arguments.no_open,
            proxy_prefix=arguments.proxy_prefix,
        )
        return 0
    raise AssertionError("unreachable command")


def _restart_process_loop(arguments: list[str]) -> int:
    generation = 0
    while True:
        generation += 1
        environment = dict(os.environ)
        environment[RESTART_CHILD_ENV] = "1"
        environment[RESTART_GENERATION_ENV] = str(generation)
        child = subprocess.run(
            [sys.executable, "-m", "eyck.cli", *arguments],
            env=environment,
            check=False,
        )
        if child.returncode != RESTART_EXIT_CODE:
            return child.returncode
        print("Restarting Eyck annotation server...", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
