import http.client
import json
import socket
import threading
import unittest
from http.server import ThreadingHTTPServer

from nli_server.server import (
    MAX_PREMISE_CHARS,
    MODEL_NAME_PATTERN,
    InstructionsTooLong,
    RequestError,
    Service,
    make_handler,
    render_state,
)

MODEL = "test-model@0000000"


class FakeScorer:
    def __init__(self):
        self.calls = []

    def score(self, premise, hypotheses):
        self.calls.append((premise, hypotheses))
        if any("too long" in h for h in hypotheses):
            raise InstructionsTooLong()
        if any("explode" in h for h in hypotheses):
            raise RuntimeError("secret detail")
        if any("nan" in h for h in hypotheses):
            return [float("nan")]
        return [0.9 if "watch" in h else 0.1 for h in hypotheses]


def request(questions=None, **overrides):
    body = {
        "state": {"title": "Replica watch", "price": 95},
        "model": MODEL,
        "questions": questions if questions is not None else {"counterfeit": {"type": "noul", "instructions": "The item is a watch."}},
    }
    body.update(overrides)
    return body


class ServerTest(unittest.TestCase):
    api_key = None
    served_name = MODEL

    def setUp(self):
        self.scorer = FakeScorer()
        self.service = Service(self.scorer, self.served_name, self.api_key)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.service))
        self.server.RequestHandlerClass.log_message = lambda *args: None
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def post(self, body, headers=None, path="/v1/systemone"):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_address[1], timeout=5)
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        connection.request("POST", path, data, {"Content-Type": "application/json", **(headers or {})})
        response = connection.getresponse()
        payload = json.loads(response.read())
        connection.close()
        return response.status, payload


class ProtocolTest(ServerTest):
    def test_answers_every_question_with_a_probability(self):
        status, payload = self.post(request({
            "a": {"type": "noul", "instructions": "The item is a watch."},
            "b": {"type": "noul", "instructions": "The item is a car."},
        }))
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"model": MODEL, "answers": {
            "a": {"type": "noul", "noul": 0.9},
            "b": {"type": "noul", "noul": 0.1},
        }})
        premise, hypotheses = self.scorer.calls[0]
        self.assertEqual(premise, "title: Replica watch\nprice: 95")
        self.assertEqual(hypotheses, ["The item is a watch.", "The item is a car."])

    def test_rejects_invalid_requests_without_scoring(self):
        cases = [
            (b"{", 400, "invalid_json"),
            (request(extra=1), 400, "invalid_request"),
            (request(model="other"), 404, "model_not_found"),
            (request(questions={}), 400, "invalid_questions"),
            (request({"a": {"type": "choice", "instructions": "x"}}), 400, "unsupported_question_type"),
            (request({"a": {"type": "noul", "instructions": " "}}), 400, "invalid_question"),
            (request({"a": {"type": "noul", "instructions": "x", "choices": []}}), 400, "invalid_question"),
            (request({"a": {"type": "noul", "instructions": "x" * 16385}}), 400, "invalid_question"),
            (request({"q" * 129: {"type": "noul", "instructions": "x"}}), 400, "invalid_question"),
            (request({"q%d" % i: {"type": "noul", "instructions": "x"} for i in range(257)}), 400, "invalid_questions"),
        ]
        for body, want_status, want_code in cases:
            with self.subTest(code=want_code):
                status, payload = self.post(body)
                self.assertEqual((status, payload["error"]["code"]), (want_status, want_code))
        self.assertEqual(self.scorer.calls, [])

    def test_maps_scorer_failures_without_leaking_details(self):
        status, payload = self.post(request({"a": {"type": "noul", "instructions": "too long"}}))
        self.assertEqual((status, payload["error"]["code"]), (400, "instructions_too_long"))
        status, payload = self.post(request({"a": {"type": "noul", "instructions": "explode"}}))
        self.assertEqual((status, payload["error"]["code"]), (500, "internal_error"))
        self.assertNotIn("secret", json.dumps(payload))

    def test_accepts_as_many_rules_as_antaeus(self):
        questions = {"q%d" % i: {"type": "noul", "instructions": "The item is a watch."} for i in range(256)}
        status, payload = self.post(request(questions))
        self.assertEqual(status, 200)
        self.assertEqual(len(payload["answers"]), 256)
        for answer in payload["answers"].values():
            self.assertEqual(set(answer), {"type", "noul"})

    def test_deeply_nested_json_is_a_client_error(self):
        status, payload = self.post(b'{"state":' + b"[" * 100000 + b"]" * 100000 + b',"model":"%s","questions":{}}' % MODEL.encode())
        self.assertEqual(status, 400)

    def test_non_finite_probability_is_a_server_error(self):
        status, payload = self.post(request({"a": {"type": "noul", "instructions": "nan"}}))
        self.assertEqual((status, payload["error"]["code"]), (500, "internal_error"))

    def test_reports_busy_when_the_model_stays_locked(self):
        self.service.queue_timeout = 0.05
        self.service.lock.acquire()
        try:
            status, payload = self.post(request())
        finally:
            self.service.lock.release()
        self.assertEqual((status, payload["error"]["code"]), (503, "busy"))

    def test_other_methods_and_missing_length(self):
        for method in ("PUT", "DELETE", "PATCH", "OPTIONS"):
            with self.subTest(method=method):
                connection = http.client.HTTPConnection("127.0.0.1", self.server.server_address[1], timeout=5)
                connection.request(method, "/v1/systemone")
                self.assertEqual(connection.getresponse().status, 405)
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_address[1], timeout=5)
        connection.putrequest("POST", "/v1/systemone")
        connection.endheaders()
        self.assertEqual(connection.getresponse().status, 411)

    def test_malformed_request_line_is_not_echoed(self):
        with socket.create_connection(("127.0.0.1", self.server.server_address[1]), timeout=5) as sock:
            sock.sendall(b"GET /secret-token HTTP/9.9\r\n\r\n")
            response = sock.recv(65536)
        self.assertIn(b"\"error\"", response)
        self.assertNotIn(b"secret-token", response)

    def test_unknown_path_and_health(self):
        status, payload = self.post(request(), path="/v1/other")
        self.assertEqual(status, 404)
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_address[1], timeout=5)
        connection.request("GET", "/healthz")
        response = connection.getresponse()
        self.assertEqual((response.status, json.loads(response.read())), (200, {"status": "ok", "model": MODEL}))

    def test_rejects_oversized_bodies_before_reading(self):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_address[1], timeout=5)
        connection.putrequest("POST", "/v1/systemone")
        connection.putheader("Content-Length", str((4 << 20) + 1))
        connection.endheaders()
        response = connection.getresponse()
        self.assertEqual(response.status, 413)


class AuthTest(ServerTest):
    api_key = "s3cret"

    def test_requires_the_bearer_key(self):
        for headers in ({}, {"Authorization": "Bearer wrong"}, {"Authorization": "s3cret"}):
            with self.subTest(headers=headers):
                status, payload = self.post(request(), headers)
                self.assertEqual((status, payload["error"]["code"]), (401, "unauthorized"))
        status, _ = self.post(request(), {"Authorization": "Bearer s3cret"})
        self.assertEqual(status, 200)


class AliasTest(ServerTest):
    served_name = "antaeus-semantic-v1"

    def test_answers_and_reports_only_the_served_name(self):
        status, payload = self.post(request(model="antaeus-semantic-v1"))
        self.assertEqual((status, payload["model"]), (200, "antaeus-semantic-v1"))
        status, payload = self.post(request(model="deberta-v3-large-zeroshot-v2.0-c@b2730f1"))
        self.assertEqual((status, payload["error"]["code"]), (404, "model_not_found"))

    def test_name_pattern_matches_the_client(self):
        for name in ("antaeus-semantic-v1", "deberta-v3-large-zeroshot-v2.0-c@b2730f1"):
            self.assertTrue(MODEL_NAME_PATTERN.fullmatch(name))
        for name in ("", "-x", "has space", "x" * 257):
            self.assertFalse(MODEL_NAME_PATTERN.fullmatch(name))


class RenderTest(unittest.TestCase):
    def test_renders_nested_values_in_document_order(self):
        state = {"listing": {"title": "Bike", "tags": ["kids", True], "seller": None}, "price": 60.5}
        self.assertEqual(
            render_state(state),
            "listing.title: Bike\nlisting.tags[0]: kids\nlisting.tags[1]: true\nlisting.seller: null\nprice: 60.5",
        )
        self.assertEqual(render_state("plain text"), "plain text")

    def test_rejects_deep_nesting(self):
        state = value = {}
        for _ in range(40):
            value["a"] = {}
            value = value["a"]
        with self.assertRaises(RequestError) as caught:
            render_state(state)
        self.assertEqual(caught.exception.code, "state_too_deep")

    def test_bounds_the_premise_for_amplifying_input(self):
        # A long key repeated across a large array would render to gigabytes.
        premise = render_state({"k" * 500000: [0] * 100000})
        self.assertLessEqual(len(premise), MAX_PREMISE_CHARS)
        self.assertTrue(premise.startswith("k" * 10))
        self.assertLessEqual(len(render_state("x" * 100000)), MAX_PREMISE_CHARS)


if __name__ == "__main__":
    unittest.main()
