# Antaeus NLI server

A small [System One](https://github.com/antaeusio/antaeus/blob/main/docs/systemone-adapter.md)
server that answers yes/no policy questions on CPU with an open
natural-language-inference (NLI) model. It lets
[Antaeus](https://github.com/antaeusio/antaeus) evaluate semantic policy rules
without sending inputs to a hosted AI provider.

> **Status: experimental.** The model's accuracy on policy conditions has not
> been measured on a real evaluation set. On a small internal check with
> synthetic marketplace listings it answered 11 of 16 rule checks correctly,
> and text inside an input can steer its answer. Gate low-confidence answers to human review, and do
> not use it for enforcement without testing it on your own cases.

## How it works

`POST /v1/systemone` receives a JSON `state` and a set of `noul` (yes/no)
questions. The server renders the state as text, one `path: value` line per
value, and uses it as the NLI premise. Each question's instructions are the
hypothesis. The answer is the model's entailment probability:

```json
{
  "state": {"title": "Luxury watch, 1:1 replica", "price": "95 USD"},
  "model": "deberta-v3-large-zeroshot-v2.0-c@b2730f1",
  "questions": {
    "prohibited-item": {
      "type": "noul",
      "instructions": "The listing offers weapons, counterfeit goods, or prescription drugs."
    }
  }
}
```

```json
{
  "model": "deberta-v3-large-zeroshot-v2.0-c@b2730f1",
  "answers": {"prohibited-item": {"type": "noul", "noul": 0.87}}
}
```

The model is
[`MoritzLaurer/deberta-v3-large-zeroshot-v2.0-c`](https://huggingface.co/MoritzLaurer/deberta-v3-large-zeroshot-v2.0-c)
(MIT license, trained on commercially usable data), pinned to revision
`b2730f1`. The image downloads it at build time, checks every file against
[`model.sha256`](model.sha256), installs hash-locked dependencies, and runs
with the Hugging Face libraries offline.

## Run it

```sh
scripts/build-image antaeus-nli-server:dev
docker run --rm -p 127.0.0.1:8080:8080 antaeus-nli-server:dev
curl -s localhost:8080/healthz
```

The image is about 3 GB and uses about 2.2 GB of memory. A request with two
rules takes about 200 ms on an unrestricted laptop CPU and about 1 second with
2 CPUs.

| Variable | Default | Meaning |
| --- | --- | --- |
| `NLI_HOST` | `0.0.0.0` in the image | Listen address |
| `NLI_PORT` | `8080` | Listen port |
| `NLI_MODEL_NAME` | `deberta-v3-large-zeroshot-v2.0-c@b2730f1` | Name the server answers to and reports, for example a versioned alias of your own; change it whenever the model or image changes |
| `NLI_API_KEY` | unset | When set, requests need `Authorization: Bearer <key>` |
| `NLI_THREADS` | container CPU limit | PyTorch threads |
| `NLI_QUEUE_TIMEOUT_SECONDS` | `10` | How long a request waits for the model before a 503 |
| `NLI_MODEL_DIR` | `/opt/model` | Model directory |

## Limits and errors

A request may contain up to 4 MiB and 256 questions, and instructions may be up
to 16,384 characters, matching Antaeus's own limits. Instructions longer than
256 model tokens are rejected, because the rest of the 512-token window is kept
for the input. The input is rendered up to 16,384 characters and then cut to
fit the window, so values late in a long input may be ignored. The model runs
one request at a time, in batches of 8 questions. Time grows with the number of rules, at
about half a second per rule with 2 CPUs, so a policy with 100 rules holds the
model for close to a minute; set client timeouts and container size with that
in mind. A request keeps running even if its client disconnects.

Errors return `{"error": {"code": "...", "message": "..."}}` and never echo
the input:

| Status | When |
| --- | --- |
| 400 | Invalid JSON or request, or instructions too long |
| 401 | Missing or wrong key |
| 404 | Another model or path |
| 405 | A method other than GET, HEAD, or POST |
| 411 | No `Content-Length` |
| 413 | Body over 4 MiB |
| 500 | Evaluation failed |
| 503 | The model stayed busy for the queue timeout (`Retry-After: 1`) |

The server logs the method, a known path, and the status, never bodies,
headers, or keys. It stops cleanly on SIGTERM.

## Development

```sh
scripts/test                      # unit tests, standard library only
scripts/build-image               # container build, under the build lock
scripts/lock                      # regenerate the hash-pinned dependency locks
```

## License

Apache License 2.0. The model weights are distributed by their author under
the MIT license.
