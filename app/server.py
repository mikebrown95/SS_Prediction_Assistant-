from __future__ import annotations

import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from model import predict_award_probability


ROOT = Path(__file__).resolve().parent
STATIC_ROOT = ROOT / "static"
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8000"))


class AppHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._send_file(STATIC_ROOT / "index.html", "text/html; charset=utf-8")
            return
        if path == "/styles.css":
            self._send_file(STATIC_ROOT / "styles.css", "text/css; charset=utf-8")
            return
        if path == "/app.js":
            self._send_file(STATIC_ROOT / "app.js", "text/javascript; charset=utf-8")
            return
        if path.startswith("/assets/"):
            asset_path = (STATIC_ROOT / path.removeprefix("/")).resolve()
            if not asset_path.is_relative_to(STATIC_ROOT.resolve()):
                self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
                return
            self._send_file(asset_path, _content_type(asset_path))
            return
        self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/api/predict":
            self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
            return

        try:
            payload = self._read_json()
            prediction = predict_award_probability(payload)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        except json.JSONDecodeError:
            self._send_json({"error": "Request body must be valid JSON"}, HTTPStatus.BAD_REQUEST)
            return

        self._send_json(
            {
                "probability": prediction.probability,
                "band": prediction.band,
                "score": prediction.score,
                "factors": prediction.factors,
                "awardLevels": prediction.award_levels,
                "predictedAwardLevel": prediction.predicted_award_level,
                "predictedOutcome": prediction.predicted_outcome,
                "ruleFindings": prediction.rule_findings,
            }
        )

    def log_message(self, format: str, *args: object) -> None:
        return

    def _read_json(self) -> dict[str, object]:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        payload = json.loads(body or b"{}")
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object")
        return payload

    def _send_file(self, path: Path, content_type: str) -> None:
        if not path.exists():
            self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
            return

        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict[str, object], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), AppHandler)
    print(f"SSDI Predictor running at http://{HOST}:{PORT}")
    server.serve_forever()


def _content_type(path: Path) -> str:
    if path.suffix == ".svg":
        return "image/svg+xml"
    if path.suffix == ".png":
        return "image/png"
    if path.suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if path.suffix == ".webp":
        return "image/webp"
    return "application/octet-stream"


if __name__ == "__main__":
    main()
