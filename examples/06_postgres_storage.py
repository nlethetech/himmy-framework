"""Example 06: Postgres storage — wire the PostgresStorageService scaffold.

Gated by ``HIMMY_TEST_POSTGRES_DSN``: when it is unset this prints a skip
notice and exits 0 (the offline default), so the example suite stays green
without a database. When the DSN is set (and the ``[postgres]`` extra is
installed) it connects, applies the idempotent schema, and runs one agent task
persisting through Postgres.

    HIMMY_TEST_POSTGRES_DSN=postgresql://himmy:himmy@localhost:5433/himmy \\
        python examples/06_postgres_storage.py
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _runtime import build_inference  # noqa: E402

from himmy.agents.base_agent.task import Task  # noqa: E402
from himmy.agents.personas.persona import Persona  # noqa: E402
from himmy.entities.registry import EntityRegistry  # noqa: E402
from himmy.runtime.single_agent import SingleAgentRuntime  # noqa: E402
from himmy.services.storage.postgres import (  # noqa: E402
    STORAGE_DDL,
    PostgresStorageService,
)


async def main() -> None:
    """Run the example against Postgres, or print a skip notice and exit 0."""
    dsn = os.environ.get("HIMMY_TEST_POSTGRES_DSN")

    print("=== Example 06: Postgres storage ===")
    if not dsn:
        print("HIMMY_TEST_POSTGRES_DSN is not set — skipping (offline default).")
        print(
            "Set it to a running pgvector DB (see docker/docker-compose.yml) and "
            "install the [postgres] extra to exercise this example."
        )
        print(f"target schema: {STORAGE_DDL.count('CREATE TABLE')} tables defined.")
        return

    # --- live path: requires the [postgres] extra + a running database ------
    storage = await PostgresStorageService.connect(dsn)
    await storage.create_schema()
    print(f"connected to Postgres; schema applied. DSN={dsn}")

    runtime = SingleAgentRuntime(
        inference_service=build_inference(),
        memory_store=storage,
        entity_registry=EntityRegistry(),
    )
    persona = Persona(name="db-backed-agent", metadata={"role": "Persistence Demo"})
    task = Task(title="hello-postgres", prompt="Say hello, persisted to Postgres.")

    thread = await runtime.run_task(persona, task)
    last = thread.last_message
    assert last is not None
    print("assistant reply:", last.content)
    print(f"thread {thread.thread_id} persisted via PostgresStorageService.")


if __name__ == "__main__":
    asyncio.run(main())
