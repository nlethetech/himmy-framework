"""Regression tests for the guardrails-redos-phone bundle.

Covers two findings in ``himmy/services/guardrails/builtins.py``:

* Finding #4 (High, ReDoS): the email PII regex used unbounded quantifiers, so an
  adversarial ``"a-a-a..."`` string drove catastrophic backtracking that pinned a core
  for minutes and blocked the event loop. The fix bounds the quantifiers and caps the
  scanned input length. These tests assert a large hostile input is inspected in well
  under 100 ms.
* Residual loop-starvation (re-attack): even with the bounded (linear) pattern, ``re``
  holds the GIL while matching, so a large SUB-CAP input still pinned the event loop
  (~900 ms for 2 MiB, even through ``asyncio.to_thread``). The fix lowers the per-scan
  cap to 128 KiB so worst-case ``inspect`` CPU drops to ~tens of ms. These tests assert a
  hostile worst-case input AT the cap completes fast and that an over-cap input is
  truncated (only the head is scanned, the tail re-appended verbatim).
* Finding #13 (Low, phone bypass): the phone validator exempted ANY value containing a
  decimal anywhere, so a dotted phone like ``4155550.132`` slipped through unredacted.
  The fix exempts only whole-span clean number tokens, so a dotted phone redacts while a
  bare ``3.14`` stays data.
"""

from __future__ import annotations

import time

from himmy.services.guardrails.builtins import _MAX_PII_SCAN_LEN, PIIGuardrail


def _inspect(text: str) -> tuple[str, list[str]]:
    verdict = PIIGuardrail().inspect(text, context={})
    return verdict.text, list(verdict.flags)


# --------------------------------------------------------------- Finding #4: ReDoS


def test_url_dense_input_is_not_quadratic() -> None:
    """A 128 KiB string packed with ``http://a x@y.z`` tokens must inspect fast.

    The "keep links intact" guard collected URL spans and, PER match, scanned the whole
    span list (``any(s <= m.start() and m.end() <= e ...)``) — O(matches x spans), i.e.
    quadratic. A URL-dense 128 KiB leaf (~8.7k spans x ~8.7k matches x ~13 rules) stalled a
    single ``inspect`` ~6 s even though it fit inside every length/leaf cap. The fix makes
    containment a binary search over the sorted, non-overlapping spans (+ a span-count cap),
    so it is ~linear. Budget: < 400 ms (was ~5900 ms).
    """
    hostile = ("http://a x@y.z " * 8740)[:_MAX_PII_SCAN_LEN]
    start = time.perf_counter()
    redacted, _flags = _inspect(hostile)
    elapsed = time.perf_counter() - start
    assert elapsed < 0.4, f"URL-dense scan is not linear: took {elapsed * 1000:.0f} ms"
    assert "[REDACTED-EMAIL]" in redacted  # emails outside URLs still redacted


def test_url_protection_keeps_real_links_intact() -> None:
    """The binary-search containment must still keep PII *inside* a URL un-redacted."""
    redacted, _flags = _inspect("mail a@b.com and link http://x.com/u?cc=keep@me.com")
    assert "[REDACTED-EMAIL]" in redacted  # the bare email redacted
    assert "http://x.com/u?cc=keep@me.com" in redacted  # the in-URL email kept


def test_email_redos_input_completes_fast() -> None:
    """A ~120 KB adversarial ``a-a-a...@`` string must not hang the regex scan.

    Against the old unbounded ``[\\w.+-]+@`` pattern this backtracks for minutes; with the
    bounded quantifier + input cap it returns near-instantly. Budget: <100 ms.
    """
    hostile = ("a-" * 30_000) + "@"
    start = time.perf_counter()
    redacted, _flags = _inspect(hostile)
    elapsed = time.perf_counter() - start
    assert elapsed < 0.1, f"email inspect took {elapsed:.3f}s (ReDoS not fixed)"
    # It is NOT a valid email (no domain), so nothing is redacted; text is preserved.
    assert redacted == hostile


def test_oversized_input_is_short_circuited_but_head_redacted() -> None:
    """An input far over the scan cap still inspects fast and keeps its full text.

    PII in the visible head is still redacted; the untouched tail is re-appended so no
    content is lost.
    """
    head_pii = "contact me at alice@example.com please. "
    bulk = "x" * (_MAX_PII_SCAN_LEN * 2)
    text = head_pii + bulk
    start = time.perf_counter()
    redacted, flags = _inspect(text)
    elapsed = time.perf_counter() - start
    assert elapsed < 0.1, f"oversized inspect took {elapsed:.3f}s"
    assert "alice@example.com" not in redacted
    assert "[REDACTED-EMAIL]" in redacted
    assert "pii:email" in flags
    # Tail content is preserved (length unchanged apart from the redaction substitution).
    assert redacted.endswith(bulk)


# ----------------------------------------- Residual loop-starvation (GIL-bound regex)


def test_scan_cap_is_small_enough_to_bound_loop_stall() -> None:
    """The scan cap must stay small (<= 128 KiB) so a sub-cap input cannot pin the loop.

    ``re`` holds the GIL while matching, so ``asyncio.to_thread`` does NOT let the event
    loop run concurrently with a regex scan — the only real lever is the input cap. A
    2 MiB cap left a ~900 ms loop stall; 128 KiB bounds it to ~tens of ms.
    """
    assert _MAX_PII_SCAN_LEN <= 128 * 1024


# The worst-case amplifier for the *bounded* rule set is a dotted run (``1.1.1.1...``):
# it keeps the ipv4 / phone / email patterns advancing-and-validating across the whole
# span instead of short-circuiting on a space. With every PII rule scanned over the input,
# this is what actually costs real CPU (and holds the GIL) per scanned byte — so the scan
# cost is ~LINEAR in scanned length. Measured: ~115 ms at 128 KiB, but ~1.8 s if a 2 MiB
# blob is scanned in full (the residual loop-stall the re-attack hit). The cap is the only
# thing that keeps the over-cap case cheap.
_HOSTILE_UNIT = "1."


def _hostile(length: int) -> str:
    """A dotted-digit run of exactly ``length`` chars — worst case for the bounded scan."""
    return (_HOSTILE_UNIT * (length // len(_HOSTILE_UNIT) + 1))[:length]


def test_hostile_worstcase_at_cap_completes_fast() -> None:
    """A hostile worst-case input exactly AT the cap inspects fast (loop-stall bounded).

    This reproduces the production amplifier — a full-cap dotted-digit run that forces the
    ipv4 / phone / email patterns to scan-and-validate the whole span — NOT a synthetic
    short string. At the new 128 KiB cap every scanned byte does work yet the whole inspect
    finishes well under the budget. (The same shape at the old 2 MiB cap took ~1.8 s,
    stalling the GIL-bound event loop — that is the bug this guards against.)
    """
    hostile = _hostile(_MAX_PII_SCAN_LEN)
    assert len(hostile) == _MAX_PII_SCAN_LEN  # exactly at the cap: no truncation path
    start = time.perf_counter()
    _redacted, _flags = _inspect(hostile)
    elapsed = time.perf_counter() - start
    # Budget is ~3x the measured cost: generous for CI jitter, still far below the ~1.8 s
    # a full 2 MiB scan of this shape costs, so a regression that re-raises the cap fails.
    assert elapsed < 0.35, f"at-cap inspect took {elapsed:.3f}s (loop-stall not bounded)"


def test_over_cap_input_is_truncated_not_scanned_in_full() -> None:
    """An over-cap hostile input is short-circuited: only the head (cap bytes) is scanned.

    The tail is re-appended verbatim, NOT scanned. Because the scan is ~linear in scanned
    length, scanning a 16x-over-cap blob in FULL would cost ~16x the at-cap time (~1.8 s of
    GIL-held stall here). The cap keeps the wall time near the at-cap budget regardless of
    how large the tail is, and the tail is preserved byte-for-byte.
    """
    head = _hostile(_MAX_PII_SCAN_LEN)
    tail = _hostile(_MAX_PII_SCAN_LEN * 16)  # 16x over cap, same hostile shape
    text = head + tail
    start = time.perf_counter()
    redacted, _flags = _inspect(text)
    elapsed = time.perf_counter() - start
    # If the tail were scanned this would take ~17x as long (~2 s). The 0.5 s ceiling sits
    # well above the truncated cost and far below a full scan, so a broken cap fails loudly.
    assert elapsed < 0.5, f"over-cap inspect took {elapsed:.3f}s (tail scanned in full?)"
    # The untouched tail is re-appended verbatim — no content lost, none of it scanned.
    assert redacted.endswith(tail)


def test_normal_email_still_redacts() -> None:
    """The bounded regex must still catch ordinary emails."""
    redacted, flags = _inspect("reach bob.smith+tag@sub.example.co.uk now")
    assert "bob.smith+tag@sub.example.co.uk" not in redacted
    assert "[REDACTED-EMAIL]" in redacted
    assert "pii:email" in flags


# ------------------------------------------------------ Finding #13: phone bypass


def test_dotted_phone_is_redacted() -> None:
    """A phone-shaped digit run with a decimal (``4155550.132``) must redact.

    A dot merely appearing inside a long digit run no longer exempts the span.
    """
    redacted, flags = _inspect("call 4155550.132 today")
    assert "4155550.132" not in redacted
    assert "[REDACTED-PHONE]" in redacted
    assert "pii:phone" in flags


def test_bare_decimal_is_not_treated_as_phone() -> None:
    """A clean whole-span decimal like ``3.14`` is data, never a phone."""
    redacted, flags = _inspect("the constant is 3.14159265 exactly")
    assert "3.14159265" in redacted
    assert "pii:phone" not in flags


def test_clean_phone_still_redacts() -> None:
    """A normal hyphen/space-grouped phone still redacts (no regression)."""
    redacted, flags = _inspect("ring 415-555-0132 anytime")
    assert "415-555-0132" not in redacted
    assert "[REDACTED-PHONE]" in redacted
    assert "pii:phone" in flags


def test_grouped_financial_number_is_not_a_phone() -> None:
    """A properly grouped financial figure (``1,234,567.89``) stays data."""
    redacted, flags = _inspect("revenue was 1,234,567.89 this year")
    assert "1,234,567.89" in redacted
    assert "pii:phone" not in flags
