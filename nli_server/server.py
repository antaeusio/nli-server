"""System One server for a natural-language-inference (NLI) classifier.

It answers yes/no ("noul") questions about a JSON state. The state is
rendered as text and used as the premise; each question's instructions are the
hypothesis; the answer is the model's entailment probability.
"""

import hmac
import json
import math
import os
import re
import signal
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MAX_REQUEST_BYTES = 4 << 20
# Antaeus allows up to 256 rules of up to 16,384 code points each.
MAX_QUESTIONS = 256
MAX_QUESTION_ID_CHARS = 128
MAX_INSTRUCTIONS_CHARS = 16384
MAX_DEPTH = 32
# The model reads at most 512 tokens, so a longer premise is never used.
# Rendering stops at this budget, which also bounds memory for any input.
MAX_PREMISE_CHARS = 16384
MAX_KEY_CHARS = 256
# How long a request waits for the model before the server reports it busy.
QUEUE_TIMEOUT_SECONDS = float(os.environ.get("NLI_QUEUE_TIMEOUT_SECONDS", "10"))
# The model names the Antaeus System One client accepts.
MODEL_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,255}")
METHODS = {"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"}
PATHS = {"/healthz", "/v1/systemone"}


class InstructionsTooLong(Exception):
    """Raised by a scorer when a hypothesis leaves no room for the premise."""


class RequestError(Exception):
    def __init__(self, status, code, message):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def render_state(state):
    """Render JSON as "path: value" lines in document order, within a budget."""
    lines = []
    used = 0

    def visit(value, path, depth):
        nonlocal used
        if depth > MAX_DEPTH:
            raise RequestError(HTTPStatus.BAD_REQUEST, "state_too_deep", "state is nested too deeply")
        if used >= MAX_PREMISE_CHARS:
            return
        if isinstance(value, dict):
            for key, item in value.items():
                key = key[:MAX_KEY_CHARS]
                visit(item, f"{path}.{key}" if path else key, depth + 1)
                if used >= MAX_PREMISE_CHARS:
                    return
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]", depth + 1)
                if used >= MAX_PREMISE_CHARS:
                    return
        else:
            text = value if isinstance(value, str) else json.dumps(value)
            # Keep the end of long paths: it names the value most directly.
            line = f"{path[-MAX_KEY_CHARS:]}: {text}" if path else text
            line = line[: MAX_PREMISE_CHARS - used]
            lines.append(line)
            used += len(line) + 1

    visit(state, "", 0)
    return "\n".join(lines)


def parse_request(body, served_model):
    """Validate a System One request; return (premise, [(id, hypothesis)])."""
    try:
        request = json.loads(body)
    except (ValueError, UnicodeDecodeError, RecursionError):
        raise RequestError(HTTPStatus.BAD_REQUEST, "invalid_json", "request body is not valid JSON")
    if not isinstance(request, dict) or set(request) != {"state", "model", "questions"}:
        raise RequestError(HTTPStatus.BAD_REQUEST, "invalid_request", "request must contain exactly state, model, and questions")
    if request["model"] != served_model:
        raise RequestError(HTTPStatus.NOT_FOUND, "model_not_found", "requested model is not served here")
    questions = request["questions"]
    if not isinstance(questions, dict) or not 1 <= len(questions) <= MAX_QUESTIONS:
        raise RequestError(HTTPStatus.BAD_REQUEST, "invalid_questions", f"questions must be an object with 1 to {MAX_QUESTIONS} entries")
    pairs = []
    for question_id, question in questions.items():
        if not 1 <= len(question_id) <= MAX_QUESTION_ID_CHARS:
            raise RequestError(HTTPStatus.BAD_REQUEST, "invalid_question", "question id has an invalid length")
        if not isinstance(question, dict) or set(question) != {"type", "instructions"}:
            raise RequestError(HTTPStatus.BAD_REQUEST, "invalid_question", "each question must contain exactly type and instructions")
        if question["type"] != "noul":
            raise RequestError(HTTPStatus.BAD_REQUEST, "unsupported_question_type", "only noul questions are supported")
        instructions = question["instructions"]
        if not isinstance(instructions, str) or not instructions.strip() or len(instructions) > MAX_INSTRUCTIONS_CHARS:
            raise RequestError(HTTPStatus.BAD_REQUEST, "invalid_question", f"instructions must be 1 to {MAX_INSTRUCTIONS_CHARS} characters")
        pairs.append((question_id, instructions))
    return render_state(request["state"]), pairs


class Service:
    """Transport-independent request handling around a scorer."""

    def __init__(self, scorer, model_name, api_key=None, queue_timeout=QUEUE_TIMEOUT_SECONDS):
        self.scorer = scorer
        self.model_name = model_name
        self.api_key = api_key.encode() if api_key else None
        self.queue_timeout = queue_timeout
        # The model is not safe to run concurrently; requests wait here briefly.
        self.lock = threading.Lock()

    def authorized(self, header):
        if self.api_key is None:
            return True
        prefix = "Bearer "
        if not header or not header.startswith(prefix):
            return False
        return hmac.compare_digest(header[len(prefix):].encode(), self.api_key)

    def answer(self, body):
        premise, pairs = parse_request(body, self.model_name)
        if not self.lock.acquire(timeout=self.queue_timeout):
            raise RequestError(HTTPStatus.SERVICE_UNAVAILABLE, "busy", "the model is busy; retry later")
        try:
            probabilities = self.scorer.score(premise, [hypothesis for _, hypothesis in pairs])
        except InstructionsTooLong:
            raise RequestError(HTTPStatus.BAD_REQUEST, "instructions_too_long", "instructions use too many model tokens")
        finally:
            self.lock.release()
        answers = {}
        for (question_id, _), p in zip(pairs, probabilities, strict=True):
            p = float(p)
            if not math.isfinite(p):
                raise ValueError("model returned a non-finite probability")
            answers[question_id] = {"type": "noul", "noul": min(1.0, max(0.0, p))}
        return {"model": self.model_name, "answers": answers}


def make_handler(service):
    class Handler(BaseHTTPRequestHandler):
        server_version = "antaeus-nli-server"
        sys_version = ""
        # Drop connections that stall while sending a request.
        timeout = 30

        def log_message(self, format, *args):
            sys.stderr.write("%s %s\n" % (self.log_date_time_string(), format % args))

        def log_request(self, code="-", size="-"):
            # Log only a known method, a known path, and the status; never the
            # raw request line, headers, or body.
            method = self.command if self.command in METHODS else "-"
            path = getattr(self, "path", "")
            self.log_message("%s %s %s", method, path if path in PATHS else "-", str(code))

        def log_error(self, format, *args):
            # Standard error messages can include the raw request line.
            pass

        def send_error(self, code, message=None, explain=None):
            # Replace the standard HTML error page, which echoes the request line.
            self.close_connection = True
            error = "internal_error" if code >= 500 else "bad_request"
            self.send_json(code, {"error": {"code": error, "message": HTTPStatus(code).phrase}})

        def send_json(self, status, payload):
            data = json.dumps(payload, separators=(",", ":")).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            if status == HTTPStatus.SERVICE_UNAVAILABLE:
                self.send_header("Retry-After", "1")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(data)

        def send_error_json(self, status, code, message):
            self.send_json(status, {"error": {"code": code, "message": message}})

        def do_GET(self):
            if self.path == "/healthz":
                self.send_json(HTTPStatus.OK, {"status": "ok", "model": service.model_name})
            else:
                self.send_error_json(HTTPStatus.NOT_FOUND, "not_found", "no such path")

        do_HEAD = do_GET

        def do_POST(self):
            if self.path != "/v1/systemone":
                self.close_connection = True
                self.send_error_json(HTTPStatus.NOT_FOUND, "not_found", "no such path")
                return
            try:
                length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                self.close_connection = True
                self.send_error_json(HTTPStatus.LENGTH_REQUIRED, "length_required", "Content-Length is required")
                return
            if length < 0 or length > MAX_REQUEST_BYTES:
                self.close_connection = True
                self.send_error_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "request_too_large", "request body is too large")
                return
            # Read the bounded body before any rejection so the client sees the
            # response instead of a reset connection.
            body = self.rfile.read(length)
            if not service.authorized(self.headers.get("Authorization")):
                self.send_error_json(HTTPStatus.UNAUTHORIZED, "unauthorized", "a valid bearer key is required")
                return
            try:
                self.send_json(HTTPStatus.OK, service.answer(body))
            except RequestError as error:
                self.send_error_json(error.status, error.code, error.message)
            except Exception:
                self.log_message("evaluation failed: %s", sys.exc_info()[0].__name__)
                self.send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, "internal_error", "evaluation failed")

        def do_PUT(self):
            self.close_connection = True
            self.send_error_json(HTTPStatus.METHOD_NOT_ALLOWED, "method_not_allowed", "method not allowed")

        do_DELETE = do_PATCH = do_OPTIONS = do_PUT

    return Handler


def cpu_quota():
    """Return the container's CPU limit (cgroup v2), or the CPU count."""
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            quota, period = f.read().split()
        if quota != "max":
            return max(1, math.ceil(int(quota) / int(period)))
    except (OSError, ValueError):
        pass
    return os.cpu_count() or 1


def main():
    from nli_server.model import MODEL_NAME, NLIScorer

    # Check configuration before the slow model load.
    model_name = os.environ.get("NLI_MODEL_NAME") or MODEL_NAME
    if not MODEL_NAME_PATTERN.fullmatch(model_name):
        sys.exit("NLI_MODEL_NAME must be 1 to 256 letters, digits, or ._:/@+- and start with a letter or digit")
    threads = int(os.environ.get("NLI_THREADS", "0")) or cpu_quota()
    scorer = NLIScorer(os.environ.get("NLI_MODEL_DIR", "/opt/model"), threads)
    service = Service(scorer, model_name, os.environ.get("NLI_API_KEY") or None)
    host = os.environ.get("NLI_HOST", "127.0.0.1")
    port = int(os.environ.get("NLI_PORT", "8080"))
    server = ThreadingHTTPServer((host, port), make_handler(service))

    def stop(signum, frame):
        # shutdown() waits for serve_forever, so it must run on another thread.
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    sys.stderr.write(f"serving {model_name} on {host}:{port} with {threads} threads\n")
    server.serve_forever()
    server.server_close()


if __name__ == "__main__":
    main()
