# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Compare a rollout result against a reference for token and logprob drift."""

import argparse
import json
from pathlib import Path


def _iteration(result):
    return result["iterations"][-1]


def _flatten_logprobs(values):
    if values is None:
        return None
    if values and isinstance(values[0], list):
        return [value for row in values for value in row]
    return list(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--max-logprob-diff", type=float, default=1.0e-2)
    args = parser.parse_args()

    with open(args.reference) as handle:
        reference = json.load(handle)
    with open(args.candidate) as handle:
        candidate = json.load(handle)

    reference_ids = reference.get("response_ids")
    candidate_ids = candidate.get("response_ids")
    if reference_ids is None or candidate_ids is None:
        raise SystemExit("both results need --save-response-ids")
    if reference_ids != candidate_ids:
        for row, (expected, actual) in enumerate(zip(reference_ids, candidate_ids)):
            if expected != actual:
                raise SystemExit(f"token mismatch at row {row}: reference={expected} candidate={actual}")
        raise SystemExit("token rows differ in length")

    reference_iteration = _iteration(reference)
    candidate_iteration = _iteration(candidate)
    if reference_iteration.get("response_sha256") != candidate_iteration.get("response_sha256"):
        raise SystemExit("response SHA differs despite matching response_ids")

    reference_logprobs = _flatten_logprobs(reference_iteration.get("response_logprobs"))
    candidate_logprobs = _flatten_logprobs(candidate_iteration.get("response_logprobs"))
    if reference_logprobs is not None and candidate_logprobs is not None:
        if len(reference_logprobs) != len(candidate_logprobs):
            raise SystemExit("logprob lengths differ")
        max_diff = 0.0
        compared = 0
        for expected, actual in zip(reference_logprobs, candidate_logprobs):
            if expected == 0.0 and actual == 0.0:
                continue
            if expected == 0.0 or actual == 0.0:
                raise SystemExit(f"logprob presence mismatch: reference={expected} candidate={actual}")
            max_diff = max(max_diff, abs(expected - actual))
            compared += 1
        print(f"tokens identical; logprob max abs diff={max_diff:.6g} over {compared} recorded positions")
        if max_diff > args.max_logprob_diff:
            raise SystemExit(f"logprob diff {max_diff} exceeds {args.max_logprob_diff}")
    else:
        print("tokens identical; logprobs were not recorded")

    print("PASS")


if __name__ == "__main__":
    main()
