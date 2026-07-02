"""The 'deploy MY agent' topology artifacts stand up an agent — not Studio.

Verified bait-and-switch (P1 #6): the README 'Deploying' section + the runbook pointed
at compose/helm that deploy Himmy *Studio* (the admin GUI), so a newcomer who followed
them ended with a GUI and ZERO agents. These checks pin the fix — a compose file and a
Helm chart that actually run the user's agent.yaml as a durable service:

* ``deploy/compose/agent-compose.yml`` parses as YAML and has an ``api`` service (``himmy
  serve`` the agent) AND a ``worker`` service (``himmy worker`` on the SAME store), both
  mounting the user's ``agent.yaml``, over ONE named durable volume;
* the api stays fail-closed (binds 127.0.0.1 by default; no unconditional published port);
* the ``himmy-agent`` Helm chart is DISTINCT from ``himmy-studio`` and ships an api + a
  worker Deployment that mount an agent.yaml ConfigMap and run the agent commands.

No docker daemon / helm binary needed — these are pure file + YAML-parse assertions.
"""

from __future__ import annotations

from pathlib import Path

import yaml


def _repo_root() -> Path:
    """Locate the repo root (holds ``deploy/``) from this test file's path."""
    return Path(__file__).resolve().parents[2]


def _compose_path() -> Path:
    return _repo_root() / "deploy" / "compose" / "agent-compose.yml"


def _agent_chart_dir() -> Path:
    return _repo_root() / "deploy" / "helm" / "himmy-agent"


# ------------------------------------------------------------------ compose file


def test_agent_compose_exists_and_parses() -> None:
    """The agent compose file exists and is well-formed YAML."""
    path = _compose_path()
    assert path.exists(), f"missing {path}"
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(doc, dict)
    assert "services" in doc


def test_agent_compose_has_api_and_worker_running_the_agent() -> None:
    """An ``api`` service serves the agent and a ``worker`` service runs on the same store."""
    doc = yaml.safe_load(_compose_path().read_text(encoding="utf-8"))
    services = doc["services"]
    assert "api" in services, "compose must define an 'api' service"
    assert "worker" in services, "compose must define a 'worker' service"

    def _cmd_str(svc: dict) -> str:
        cmd = svc.get("command", [])
        return " ".join(cmd) if isinstance(cmd, list) else str(cmd)

    api_cmd = _cmd_str(services["api"])
    worker_cmd = _cmd_str(services["worker"])
    # api = `himmy serve -f /app/agent.yaml`; worker = `himmy worker -f /app/agent.yaml`.
    assert "serve" in api_cmd and "/app/agent.yaml" in api_cmd
    assert "worker" in worker_cmd and "/app/agent.yaml" in worker_cmd


def test_agent_compose_commands_invoke_the_himmy_binary() -> None:
    """The exec target is `himmy` — the image has NO ENTRYPOINT, so `command:` must
    start with the binary or Docker execs a nonexistent "serve"/"worker"."""
    doc = yaml.safe_load(_compose_path().read_text(encoding="utf-8"))
    for name in ("api", "worker"):
        cmd = doc["services"][name].get("command", [])
        assert isinstance(cmd, list) and cmd, f"{name} command must be a non-empty list"
        assert cmd[0] == "himmy", f"{name} must exec the himmy binary, got {cmd[0]!r}"


def test_agent_compose_mounts_agent_yaml_in_both_services() -> None:
    """Both api and worker bind-mount the user's agent.yaml at the stable CMD path."""
    doc = yaml.safe_load(_compose_path().read_text(encoding="utf-8"))
    for name in ("api", "worker"):
        vols = doc["services"][name].get("volumes", [])
        assert any(
            "/app/agent.yaml" in str(v) for v in vols
        ), f"{name} must mount agent.yaml"


def test_agent_compose_shares_one_durable_store_volume() -> None:
    """api + worker share a single named volume for the durable .himmy store."""
    doc = yaml.safe_load(_compose_path().read_text(encoding="utf-8"))
    named = set(doc.get("volumes", {}) or {})
    assert named, "compose must declare named volumes for the durable store"

    def _state_vol(svc: dict) -> set[str]:
        out = set()
        for v in svc.get("volumes", []):
            src = str(v).split(":", 1)[0]
            if src in named and "/app/.himmy" in str(v):
                out.add(src)
        return out

    api_state = _state_vol(doc["services"]["api"])
    worker_state = _state_vol(doc["services"]["worker"])
    assert api_state, "api must mount a named volume at /app/.himmy"
    assert api_state == worker_state, "api + worker must share the SAME .himmy volume"


def test_agent_compose_is_fail_closed_by_default() -> None:
    """The api binds loopback and publishes no port unconditionally (security-first)."""
    doc = yaml.safe_load(_compose_path().read_text(encoding="utf-8"))
    api = doc["services"]["api"]
    cmd = " ".join(api.get("command", []))
    # Default bind is 127.0.0.1 (the ${AGENT_BIND:-127.0.0.1} interpolation default).
    assert "127.0.0.1" in cmd
    # No active `ports:` mapping — exposing is an opt-in the operator uncomments.
    assert not api.get("ports"), "api must not publish a port by default (fail-closed)"


def test_agent_compose_is_not_the_studio_stack() -> None:
    """This file deploys the agent, not Studio — no `studio` service, distinct volumes."""
    text = _compose_path().read_text(encoding="utf-8")
    doc = yaml.safe_load(text)
    assert "studio" not in doc["services"]
    # Volumes are namespaced away from the Studio stack (himmy_state / himmy_pgdata_full).
    assert "himmy_agent_state" in (doc.get("volumes") or {})


# -------------------------------------------------------------------- helm chart


def test_agent_chart_exists_and_is_distinct_from_studio() -> None:
    """A ``himmy-agent`` chart exists and is a different chart than ``himmy-studio``."""
    chart_yaml = _agent_chart_dir() / "Chart.yaml"
    assert chart_yaml.exists(), f"missing {chart_yaml}"
    meta = yaml.safe_load(chart_yaml.read_text(encoding="utf-8"))
    assert meta["name"] == "himmy-agent"
    assert meta["name"] != "himmy-studio"


def test_agent_chart_values_parse() -> None:
    """values.yaml is well-formed YAML and carries api + worker + agent blocks."""
    values = yaml.safe_load(
        (_agent_chart_dir() / "values.yaml").read_text(encoding="utf-8")
    )
    assert "api" in values and "worker" in values and "agent" in values
    # Fail-closed default bind.
    assert values["api"]["bindHost"] == "127.0.0.1"


def test_agent_chart_has_api_and_worker_deployments() -> None:
    """The chart ships a distinct api Deployment and worker Deployment."""
    tpl = _agent_chart_dir() / "templates"
    api = (tpl / "deployment-api.yaml").read_text(encoding="utf-8")
    worker = (tpl / "deployment-worker.yaml").read_text(encoding="utf-8")
    for text in (api, worker):
        assert "kind: Deployment" in text
    # api runs `himmy serve`; worker runs `himmy worker`. Both mount the agent.yaml.
    assert '"serve"' in api and "/app/agent.yaml" in api
    assert '"worker"' in worker and "/app/agent.yaml" in worker


def test_agent_chart_deployments_pin_the_himmy_command() -> None:
    """Both Deployments pin `command: ["himmy"]` — the image has NO ENTRYPOINT, so a
    bare `args:` would exec a nonexistent "serve"/"worker" and CrashLoop the pod."""
    tpl = _agent_chart_dir() / "templates"
    for fname in ("deployment-api.yaml", "deployment-worker.yaml"):
        text = (tpl / fname).read_text(encoding="utf-8")
        assert "command:" in text and '- "himmy"' in text, (
            f"{fname} must pin command: [\"himmy\"] so args don't exec a missing binary"
        )


def test_agent_chart_api_probes_are_exec_not_httpget() -> None:
    """The api liveness/readiness probes are `exec` (in-pod curl), not `httpGet`.

    With the fail-closed bindHost 127.0.0.1 default uvicorn listens only on the
    container loopback, so a kubelet httpGet to the pod IP would be refused and
    CrashLoop the pod. exec runs in the pod netns and reaches loopback at any bind.
    """
    api = (_agent_chart_dir() / "templates" / "deployment-api.yaml").read_text(
        encoding="utf-8"
    )
    assert "httpGet:" not in api, "api probes must not use httpGet (loopback bind)"
    assert api.count("exec:") >= 2, "api needs exec liveness + readiness probes"
    assert "/health" in api and "/readyz" in api


def test_agent_chart_worker_colocates_with_api() -> None:
    """The worker template hard-co-locates with the api pod (shared RWO PVC).

    A RWO PVC attaches to one node at a time; without podAffinity the worker can land
    on a different node and hang with a Multi-Attach error.
    """
    worker = (_agent_chart_dir() / "templates" / "deployment-worker.yaml").read_text(
        encoding="utf-8"
    )
    assert "podAffinity:" in worker
    assert "requiredDuringSchedulingIgnoredDuringExecution" in worker
    assert "kubernetes.io/hostname" in worker
    assert "component: api" in worker


def test_agent_chart_mounts_agent_yaml_from_configmap() -> None:
    """The agent spec is delivered via a ConfigMap mounted at /app/agent.yaml."""
    tpl = _agent_chart_dir() / "templates"
    helpers = (tpl / "_helpers.tpl").read_text(encoding="utf-8")
    configmap = (tpl / "configmap.yaml").read_text(encoding="utf-8")
    assert "kind: ConfigMap" in configmap
    # The shared volumeMounts helper mounts the ConfigMap at the stable CMD path.
    assert "/app/agent.yaml" in helpers
    assert "configMap" in helpers
