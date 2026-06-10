"""Tests for the operational scripts under ``scripts/`` (loaded by file path).

The ``scripts/`` directory is not an importable package, so these tests load each
``ops_*.py`` module via :func:`importlib.util.spec_from_file_location`.
"""
