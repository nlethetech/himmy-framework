"""Chaos suite: failure-injection tests (real timeouts, mid-stream death, dead storage).

Unlike the unit suites, these tests inject *genuine* failures — handlers that
really sleep past their timeout, providers that die mid-stream, storage that
raises like an unreachable database — and lock the framework's failure-handling
contract: typed classification, no hangs, no orphan tasks. Every awaited
scenario is wrapped in an ``asyncio.wait_for`` guard so a regression to a hang
fails in seconds instead of stalling the suite.
"""
