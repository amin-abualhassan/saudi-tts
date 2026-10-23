"""Local status dashboard and matched-sample listening comparison on port 7860."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import threading
from urllib.parse import parse_qs, urlsplit

from common import HERE, atomic_json, read_json, read_rows


def safe_asset(value):
    path = Path(value).resolve()
    if not path.is_relative_to(HERE.resolve()) or path.suffix != ".wav" or not path.is_file():
        raise ValueError("Only generated WAVs within this suite may be served")
    return path


def snapshot(monitor):
    state = read_json(monitor.directory / "status.json", monitor.snapshot())
    campaign = state.get("campaign")
    if campaign:
        directory = Path(campaign)
        if directory.resolve().is_relative_to((HERE / "suites").resolve()):
            state["models"] = read_json(directory / "comparison/models.json", [])
            state["ratings"] = read_json(directory / "comparison/ratings.json", {})
            cases = directory / "comparison/cases.jsonl"
            if cases.is_file():
                state["cases"] = read_rows(cases)
    return state


def handler(monitor):
    class Handler(BaseHTTPRequestHandler):
        def send(self, payload, content_type, code=200):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            route = urlsplit(self.path)
            if route.path == "/":
                self.send((HERE / "dashboard.html").read_bytes(), "text/html; charset=utf-8")
            elif route.path == "/api/status":
                self.send(json.dumps(snapshot(monitor), allow_nan=False).encode(), "application/json; charset=utf-8")
            elif route.path == "/audio":
                try:
                    path = safe_asset(parse_qs(route.query).get("path", [""])[0])
                    data = path.read_bytes()
                    self.send(data, mimetypes.guess_type(path)[0] or "audio/wav")
                except (ValueError, OSError):
                    self.send_error(404)
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path != "/api/rating":
                self.send_error(404)
                return
            # Browser writes stay on this dashboard origin.
            origin = self.headers.get("Origin")
            host = self.headers.get("Host")
            if origin and urlsplit(origin).netloc != host:
                self.send_error(403)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 4096:
                    raise ValueError("Invalid payload size")
                item = json.loads(self.rfile.read(length))
                state = snapshot(monitor)
                model = item["model"]
                case = item["case"]
                if model not in {m["name"] for m in state.get("models", [])} or case not in {c["id"] for c in state.get("cases", [])}:
                    raise ValueError("Unknown model or sample")
                ratings = {k: item[k] for k in ("accent", "clarity", "voice", "stability")}
                if any(type(v) is not int or not 1 <= v <= 5 for v in ratings.values()):
                    raise ValueError("Ratings must be integers from 1 to 5")
                path = Path(state["campaign"]) / "comparison/ratings.json"
                with monitor.lock:
                    saved = read_json(path, {})
                    saved[model+":"+case] = ratings
                    atomic_json(path, saved)
                self.send(b'{"saved":true}', "application/json")
            except (ValueError, KeyError, TypeError, OSError):
                self.send_error(400)

        def log_message(self, *args):
            pass
    return Handler


def serve(monitor, host, port):
    server = ThreadingHTTPServer((host, port), handler(monitor))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
