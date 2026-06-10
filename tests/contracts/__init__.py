"""Provider contract tests: SDK shape parity for the inference adapters.

Two layers per provider (Anthropic / OpenAI):

* ``test_*_shapes.py`` — ALWAYS run. Hand-built fixture objects mirror the exact
  SDK request/response shapes (verified against pinned SDK versions) and are fed
  through the adapters' normalization paths via injected fake clients, so the
  adapter side of the provider contract is guarded even in the offline base
  install where the SDK extras are absent.
* ``test_*_sdk.py`` — run only when the SDK extra is installed
  (``pytest.importorskip``). They rebuild the same scenarios from the REAL SDK
  types and validate the adapter's call kwargs against the SDK's own request
  TypedDicts, so an SDK upgrade that changes shapes fails loudly here.
"""
