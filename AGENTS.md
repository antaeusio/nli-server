# AGENTS.md

Working rules for humans and coding agents in this public repository.

## Scope

This repository contains a small System One server that runs an open NLI model
on CPU for Antaeus. Keep it small: protocol handling, input limits, the pinned
model, and the container image. Policy semantics, decision contracts, and
routing belong in the public `antaeusio/antaeus` repository. Do not copy
private roadmaps, hosted-service architecture, customer data, or unpublished
evaluation data here.

## Commit policy

- Never create a commit without explicit human approval of the exact commit
  message and the changes to be committed.
- Commit messages are short, imperative, and exactly one line, with no
  conversation links, session IDs, generated-by notices, or co-author trailers
  unless the human asks for them.
- Do not force-push, rewrite shared history, merge a pull request, publish an
  image, or deploy unless the human explicitly authorizes that action.

## Review

Pull requests with meaningful logic (request handling, limits, authentication,
model loading, the container image, or CI) need an independent review from a
fresh session using a different model family from the author, recorded in the
pull-request description, as in `antaeusio/antaeus`.

## One build at a time

Container builds run through `scripts/with-build-lock` (use
`scripts/build-image`). It holds `.tmp/antaeus-build.lock` and refuses to break
a possibly live lock.

## Engineering rules

- Never log request bodies, headers, or keys, and never echo input in errors.
- Keep the model pinned by revision and checksum; changing the model is a
  reviewed change with a before-and-after comparison on the same cases.
- Keep unit tests free of the model so they run with the standard library.
- Never read, print, commit, or modify `.env` files unless the human asks.
