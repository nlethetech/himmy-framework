"""Tests for the CSV and Excel document readers."""

from __future__ import annotations

from pathlib import Path

import pytest

from himmy.services.knowledge.readers import CsvReader, DocumentReaderFactory


def test_csv_reader_flattens_rows(tmp_path: Path) -> None:
    """A CSV becomes one ``col=value | …`` line per row."""
    path = tmp_path / "data.csv"
    path.write_text("name,qty\napple,3\npear,5\n")
    text = CsvReader().read(str(path))
    assert "name=apple | qty=3" in text
    assert "name=pear | qty=5" in text


def test_factory_routes_csv(tmp_path: Path) -> None:
    """The factory picks the CSV reader by extension."""
    path = tmp_path / "d.csv"
    path.write_text("a,b\n1,2\n")
    assert "a=1 | b=2" in DocumentReaderFactory().read(str(path))


def test_excel_reader_reads_sheets(tmp_path: Path) -> None:
    """An .xlsx workbook flattens each sheet's rows (skips if openpyxl absent)."""
    openpyxl = pytest.importorskip("openpyxl")
    path = tmp_path / "book.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    ws.append(["name", "qty"])
    ws.append(["apple", 3])
    wb.save(path)

    text = DocumentReaderFactory().read(str(path))
    assert "# Sheet1" in text
    assert "name\tqty" in text
    assert "apple\t3" in text
