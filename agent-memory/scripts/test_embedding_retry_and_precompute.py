#!/usr/bin/env python3
import os
import sys
import time
import types
from pathlib import Path

# Load the lightweight limiter module directly.
import importlib.util
spec = importlib.util.spec_from_file_location(
    'embedding_rate_limiter_under_test',
    Path(__file__).resolve().parents[1] / 'memrl/service/embedding_rate_limiter.py',
)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

class ConnectionFailure(Exception):
    pass

class RateFailure(Exception):
    status_code = 429


def test_connection_failure_uses_small_attempt_limit():
    old = dict(os.environ)
    try:
        os.environ['MEMRL_EMBED_CONNECTION_MAX_RETRIES'] = '2'
        os.environ['MEMRL_EMBED_CONNECTION_RETRY_BUDGET_S'] = '20'
        os.environ['MEMRL_EMBED_CONNECTION_BASE_DELAY'] = '0'
        os.environ['MEMRL_EMBED_RETRY_JITTER'] = '0'
        os.environ['MEMRL_EMBED_GLOBAL_MIN_INTERVAL'] = '0'
        calls = {'n': 0}
        def fail():
            calls['n'] += 1
            raise ConnectionFailure('Connection error.')
        try:
            mod.call_embedding_with_retry(fail)
        except ConnectionFailure:
            pass
        else:
            raise AssertionError('connection failure should propagate')
        assert calls['n'] == 2, calls
    finally:
        os.environ.clear(); os.environ.update(old)


def test_429_keeps_outer_max_attempts():
    old = dict(os.environ)
    try:
        os.environ['MEMRL_EMBED_MAX_RETRIES'] = '4'
        os.environ['MEMRL_EMBED_429_BASE_DELAY'] = '0'
        os.environ['MEMRL_EMBED_429_MAX_DELAY'] = '0'
        os.environ['MEMRL_EMBED_RETRY_JITTER'] = '0'
        os.environ['MEMRL_EMBED_GLOBAL_MIN_INTERVAL'] = '0'
        calls = {'n': 0}
        def fail():
            calls['n'] += 1
            raise RateFailure('429 rate limit')
        try:
            mod.call_embedding_with_retry(fail)
        except RateFailure:
            pass
        else:
            raise AssertionError('429 should propagate after max attempts')
        assert calls['n'] == 4, calls
    finally:
        os.environ.clear(); os.environ.update(old)

if __name__ == '__main__':
    test_connection_failure_uses_small_attempt_limit()
    test_429_keeps_outer_max_attempts()
    print('OK: embedding retry policy separates connection failures from 429')
