"""Unit tests for the OpenAPI → HttpToolSpec loader (pure parsing, offline).

The e2e coverage lives in ``test_openapi_connector_e2e.py`` (real loopback server); here
we lock the spec-derivation contract: operation naming, base-URL selection, parameter /
body / response extraction, security-scheme → auth mapping, ``$ref`` resolution, method
allow-listing, and the approve-mutations gate. No network is touched.
"""

from __future__ import annotations

from typing import Any

from himmy.config.openapi_spec import load_openapi_tools
from himmy.services.tools.models import HttpAuthMode


def _doc(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "openapi": "3.0.0",
        "info": {"title": "Demo", "version": "1"},
        "servers": [{"url": "https://api.demo.example/v1"}],
        "paths": {
            "/users/{id}": {
                "get": {
                    "operationId": "get_user",
                    "summary": "Fetch a user.",
                    "parameters": [
                        {"name": "id", "in": "path", "required": True,
                         "schema": {"type": "string"}},
                        {"name": "expand", "in": "query",
                         "schema": {"type": "string"}},
                    ],
                    "responses": {
                        "200": {
                            "content": {
                                "application/json": {
                                    "schema": {"type": "object"}
                                }
                            }
                        }
                    },
                }
            },
            "/users": {
                "post": {
                    "operationId": "create_user",
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "required": ["email"],
                                    "properties": {
                                        "email": {"type": "string"},
                                        "age": {"type": "integer"},
                                    },
                                }
                            }
                        }
                    },
                    "responses": {"201": {"description": "created"}},
                }
            },
        },
    }
    base.update(overrides)
    return base


def test_loader_derives_one_spec_per_operation() -> None:
    specs = load_openapi_tools(_doc())
    by_name = {s.name: s for s in specs}
    assert set(by_name) == {"get_user", "create_user"}
    assert by_name["get_user"].method == "GET"
    assert by_name["create_user"].method == "POST"


def test_loader_uses_server_url_as_base() -> None:
    specs = load_openapi_tools(_doc())
    assert all(s.base_url == "https://api.demo.example/v1" for s in specs)


def test_loader_base_url_override_wins() -> None:
    specs = load_openapi_tools(_doc(), base_url="https://override.example/")
    assert all(s.base_url == "https://override.example" for s in specs)


def test_path_and_query_params_become_typed_args() -> None:
    spec = next(s for s in load_openapi_tools(_doc()) if s.name == "get_user")
    schema = spec.args_schema
    assert schema is not None
    assert set(schema["properties"]) == {"id", "expand"}
    assert schema["required"] == ["id"]  # path placeholder is required


def test_body_fields_carry_types_and_required() -> None:
    spec = next(s for s in load_openapi_tools(_doc()) if s.name == "create_user")
    schema = spec.args_schema
    assert schema is not None
    assert schema["properties"]["age"]["type"] == "integer"
    assert "email" in schema["required"]
    assert spec.body == ["email", "age"]


def test_security_scheme_maps_to_bearer_auth() -> None:
    doc = _doc(
        components={
            "securitySchemes": {"b": {"type": "http", "scheme": "bearer"}}
        }
    )
    spec = load_openapi_tools(doc, secret="TOK")[0]
    assert spec.to_config().auth.mode is HttpAuthMode.BEARER
    assert spec.to_config().auth.env_var == "TOK"


def test_apikey_query_scheme_maps_to_query_auth() -> None:
    doc = _doc(
        components={
            "securitySchemes": {
                "k": {"type": "apiKey", "in": "query", "name": "api_key"}
            }
        }
    )
    spec = load_openapi_tools(doc, secret="K")[0]
    auth = spec.to_config().auth
    assert auth.mode is HttpAuthMode.API_KEY_QUERY
    assert auth.query_param == "api_key"


def test_no_secret_means_no_auth_block() -> None:
    """Without a supplied secret NAME we never invent a credential."""
    specs = load_openapi_tools(_doc())
    assert all(s.auth is None for s in specs)


def test_ref_in_request_body_is_resolved() -> None:
    doc = _doc()
    doc["components"] = {
        "schemas": {
            "NewUser": {
                "type": "object",
                "required": ["email"],
                "properties": {"email": {"type": "string"}},
            }
        }
    }
    doc["paths"]["/users"]["post"]["requestBody"]["content"][
        "application/json"
    ]["schema"] = {"$ref": "#/components/schemas/NewUser"}
    spec = next(s for s in load_openapi_tools(doc) if s.name == "create_user")
    assert spec.body == ["email"]


def test_approve_mutations_gates_write_methods_only() -> None:
    specs = {s.name: s for s in load_openapi_tools(_doc(), approve_mutations=True)}
    assert specs["create_user"].requires_approval is True  # POST
    assert specs["get_user"].requires_approval is False  # GET


def test_include_methods_filters_operations() -> None:
    specs = load_openapi_tools(_doc(), include_methods=frozenset({"GET"}))
    assert {s.name for s in specs} == {"get_user"}


def test_disallowed_method_in_spec_is_skipped() -> None:
    doc = _doc()
    doc["paths"]["/trace"] = {"trace": {"operationId": "t"}}  # not an OpenAPI verb key
    names = {s.name for s in load_openapi_tools(doc)}
    assert "t" not in names


def test_yaml_text_document_is_parsed() -> None:
    yaml_text = (
        "openapi: 3.0.0\n"
        "info: {title: y, version: '1'}\n"
        "servers: [{url: https://y.example}]\n"
        "paths:\n"
        "  /ping:\n"
        "    get: {operationId: ping, responses: {'200': {description: ok}}}\n"
    )
    specs = load_openapi_tools(yaml_text)
    assert {s.name for s in specs} == {"ping"}
    assert specs[0].base_url == "https://y.example"


def test_validate_responses_attaches_response_schema() -> None:
    spec = next(
        s
        for s in load_openapi_tools(_doc(), validate_responses=True)
        if s.name == "get_user"
    )
    assert spec.response_schema == {"type": "object"}


def test_duplicate_operation_names_are_disambiguated() -> None:
    doc = {
        "openapi": "3.0.0",
        "info": {"title": "d", "version": "1"},
        "servers": [{"url": "https://d.example"}],
        "paths": {
            "/a": {"get": {"summary": "a"}},
            "/a/": {"get": {"summary": "a-slash"}},
        },
    }
    names = [s.name for s in load_openapi_tools(doc)]
    assert len(names) == len(set(names))  # no collisions
