"""HTTP-level tests for the Studio KB file-upload router.

Drives the FastAPI app in-process via TestClient (Host=testserver passes the
studio guard) — no network, no server. Covers the happy paths for every
supported extension that needs no extra (txt/md/csv), the validation 4xx
matrix (type, content-type, size, empty, missing part, unknown KB), the
python-multipart-missing degradation, dedup on re-upload, filename
sanitization, and the documents listing.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from himmy.api import studio_knowledge
from himmy.api.app import create_app
from himmy.api.routers import studio_knowledge_upload as sku


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.chdir(tmp_path)
    studio_knowledge.reset_kb_service()
    yield TestClient(create_app())
    studio_knowledge.reset_kb_service()


def _mk_kb(client: TestClient, name: str = "docs") -> str:
    r = client.post("/api/studio/knowledge", json={"name": name})
    assert r.status_code == 200
    return r.json()["kb_id"]


def _upload(
    client: TestClient,
    kb_id: str,
    filename: str,
    data: bytes,
    content_type: str,
    extra: dict[str, str] | None = None,
):
    return client.post(
        f"/api/studio/kb/{kb_id}/upload",
        files={"file": (filename, data, content_type)},
        data=extra or {},
    )


# ---- happy paths ----------------------------------------------------------


def test_upload_txt_then_search_and_list(client: TestClient) -> None:
    kb_id = _mk_kb(client)
    body = b"Khaki Campbell ducks lay around 300 eggs a year.\n" * 4
    r = _upload(client, kb_id, "ducks.txt", body, "text/plain")
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["document_id"]
    assert out["chunks"] >= 1
    assert out["characters"] == len(body.decode())
    assert out["source_uri"] == "ducks.txt"
    assert out["title"] == "ducks"  # filename stem when no title is sent

    # KB list document count reflects the upload.
    kbs = client.get("/api/studio/knowledge").json()
    assert kbs[0]["documents"] == 1

    # The upload is searchable through the existing search endpoint, and the
    # hit carries the filename as its source.
    hits = client.post(
        f"/api/studio/knowledge/{kb_id}/search",
        json={"query": "ducks eggs", "top_k": 3},
    ).json()
    assert isinstance(hits, list) and hits
    assert hits[0]["source_uri"] == "ducks.txt"

    # The documents listing shows it with chunk/char counts.
    docs = client.get(f"/api/studio/kb/{kb_id}/documents").json()
    assert len(docs) == 1
    assert docs[0]["source_uri"] == "ducks.txt"
    assert docs[0]["chunks"] == out["chunks"]
    assert docs[0]["characters"] == out["characters"]


def test_upload_md_with_title_field(client: TestClient) -> None:
    kb_id = _mk_kb(client)
    r = _upload(
        client,
        kb_id,
        "notes.md",
        b"# Pond\n\nThe pond holds carp and ducks.",
        "text/markdown",
        extra={"title": "Farm pond notes"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["title"] == "Farm pond notes"


def test_upload_csv_is_flattened(client: TestClient) -> None:
    kb_id = _mk_kb(client)
    r = _upload(
        client,
        kb_id,
        "trees.csv",
        b"species,count\nmango,12\nlitchi,4\n",
        "text/csv",
    )
    assert r.status_code == 200, r.text
    assert r.json()["chunks"] >= 1
    # CsvReader flattens rows to header=value pairs — visible in retrieval.
    hits = client.post(
        f"/api/studio/knowledge/{kb_id}/search",
        json={"query": "mango", "top_k": 3},
    ).json()
    assert hits and "species=mango" in hits[0]["text"]


def test_reupload_same_file_dedupes(client: TestClient) -> None:
    kb_id = _mk_kb(client)
    body = b"The orchard has 12 mango trees."
    first = _upload(client, kb_id, "orchard.txt", body, "text/plain").json()
    second = _upload(client, kb_id, "orchard.txt", body, "text/plain").json()
    assert second["document_id"] == first["document_id"]
    assert client.get("/api/studio/knowledge").json()[0]["documents"] == 1
    assert len(client.get(f"/api/studio/kb/{kb_id}/documents").json()) == 1


def test_filename_is_sanitized_to_basename(client: TestClient) -> None:
    kb_id = _mk_kb(client)
    r = _upload(client, kb_id, "../../etc/notes.txt", b"hello there", "text/plain")
    assert r.status_code == 200, r.text
    assert r.json()["source_uri"] == "notes.txt"  # no path components survive


# ---- validation / error paths ----------------------------------------------


def test_unsupported_extension_415(client: TestClient) -> None:
    kb_id = _mk_kb(client)
    r = _upload(client, kb_id, "evil.exe", b"MZ....", "application/octet-stream")
    assert r.status_code == 415
    assert ".pdf" in r.json()["detail"]


def test_mismatched_content_type_415(client: TestClient) -> None:
    kb_id = _mk_kb(client)
    r = _upload(client, kb_id, "notes.txt", b"hello", "application/pdf")
    assert r.status_code == 415


def test_empty_file_422(client: TestClient) -> None:
    kb_id = _mk_kb(client)
    r = _upload(client, kb_id, "empty.txt", b"", "text/plain")
    assert r.status_code == 422
    assert "empty" in r.json()["detail"]


def test_whitespace_only_file_422(client: TestClient) -> None:
    kb_id = _mk_kb(client)
    r = _upload(client, kb_id, "blank.txt", b"   \n\n  ", "text/plain")
    assert r.status_code == 422
    assert "no extractable text" in r.json()["detail"]


def test_oversize_file_413(client: TestClient) -> None:
    kb_id = _mk_kb(client)
    blob = b"x" * (sku.MAX_UPLOAD_BYTES + 1024 * 1024)
    r = _upload(client, kb_id, "big.txt", blob, "text/plain")
    assert r.status_code == 413


def test_corrupt_pdf_is_4xx_not_500(client: TestClient) -> None:
    pytest.importorskip("pypdf")
    kb_id = _mk_kb(client)
    r = _upload(client, kb_id, "broken.pdf", b"not a pdf at all", "application/pdf")
    assert 400 <= r.status_code < 500
    assert "broken.pdf" in r.json()["detail"] or "pdf" in r.json()["detail"].lower()


def test_missing_file_part_422(client: TestClient) -> None:
    kb_id = _mk_kb(client)
    r = client.post(f"/api/studio/kb/{kb_id}/upload", data={"title": "no file"})
    assert r.status_code == 422
    assert "file" in r.json()["detail"]


def test_unknown_kb_404(client: TestClient) -> None:
    r = _upload(client, "nope-123", "a.txt", b"hello", "text/plain")
    assert r.status_code == 404


def test_invalid_utf8_text_422(client: TestClient) -> None:
    kb_id = _mk_kb(client)
    r = _upload(client, kb_id, "bin.txt", b"\xff\xfe\x00\x01\x80", "text/plain")
    assert r.status_code == 422
    assert "UTF-8" in r.json()["detail"]


def test_multipart_missing_degrades_to_400(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    kb_id = _mk_kb(client)
    monkeypatch.setattr(sku, "_multipart_available", lambda: False)
    r = _upload(client, kb_id, "a.txt", b"hello", "text/plain")
    assert r.status_code == 400
    assert "python-multipart" in r.json()["detail"]


def test_documents_unknown_kb_404(client: TestClient) -> None:
    assert client.get("/api/studio/kb/nope-123/documents").status_code == 404


def test_documents_empty_kb_is_empty_list(client: TestClient) -> None:
    kb_id = _mk_kb(client)
    assert client.get(f"/api/studio/kb/{kb_id}/documents").json() == []


def test_kb_id_bounds_422(client: TestClient) -> None:
    r = _upload(client, "k" * 500, "a.txt", b"hello", "text/plain")
    assert r.status_code == 422
