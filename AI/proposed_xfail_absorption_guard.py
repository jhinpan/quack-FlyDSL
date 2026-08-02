"""PROPOSAL, NOT WIRED IN. Runtime enforcement of the invariant @Autotune's
AST meta-test states.

Deliberately not installed as a conftest hook: ``tests/test_autotune_cache_context.py``
is @Autotune's file and whether this lands in ``tests/conftest.py`` is his
call. Sitting here it does nothing. Copy it to ``tests/conftest_xfail_guard.py``
and re-export ``pytest_runtest_makereport`` from ``tests/conftest.py`` to arm it.

The invariant: when a test marked ``xfail(raises=E)`` records an XFAIL, the
exception must have come from that test's own code -- not from a shared helper
it happens to call. Absorption is exactly the case where the raise site is
somebody else's code.

Checked on the traceback rather than on the source text, so aliasing the class,
fetching it with ``getattr``, or moving the helper into another module all land
in the same place: the raising code object is not the test's.

"The test's own code" includes closures, comprehensions and generator
expressions defined lexically inside the test body -- they are compiled into
the test's ``co_consts``. An earlier version of this file compared only
``co_name`` and so called a nested closure absorption, which is the very
defect class it exists to catch: a check correct about a set other than the
one its label names.
"""

import pytest


def _own_code_objects(code, seen=None):
    """Every code object compiled lexically inside ``code``, including itself."""
    if seen is None:
        seen = set()
    if id(code) in seen:
        return seen
    seen.add(id(code))
    for const in code.co_consts:
        if hasattr(const, "co_code"):
            _own_code_objects(const, seen)
    return seen


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if call.when != "call" or call.excinfo is None:
        return
    marker = item.get_closest_marker("xfail")
    if marker is None:
        return
    expected = marker.kwargs.get("raises")
    if expected is None or not isinstance(call.excinfo.value, expected):
        return
    func = getattr(item, "function", None)
    if func is None:
        return

    own = _own_code_objects(func.__code__)
    tb = call.excinfo.tb
    while tb.tb_next is not None:
        tb = tb.tb_next
    frame = tb.tb_frame
    if id(frame.f_code) in own:
        return

    report.outcome = "failed"
    report.longrepr = (
        f"XFAIL ABSORPTION: {item.nodeid} expects {expected.__name__} from its "
        f"own body, but it was raised by {frame.f_code.co_name}() at "
        f"{frame.f_code.co_filename}:{frame.f_lineno}. Someone else's bug is "
        f"being recorded as this test's expected outcome."
    )
