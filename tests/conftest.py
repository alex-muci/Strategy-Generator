"""
pytest only (unittest never reads this file): the `slow` marker.

A test class with a truthy `slow` attribute is marked `slow`, and a slow test
is SKIPPED unless the run selects it with -m:

    python -m pytest -n auto --dist loadscope           # everything but the slow ones
    python -m pytest -m slow                             # only the slow ones
    python -m pytest -m "slow or not slow"               # all of them

The attribute, not `@pytest.mark.slow`, so that the test modules do not import
pytest: `python -m unittest discover -s tests -t .` runs every test, slow
ones included, on an install without the dev requirements.

The environment set-up every test process needs lives in tests/__init__.py.
"""

import pytest


@pytest.hookimpl(tryfirst=True)   # mark before pytest's own -m selection reads the markers
def pytest_collection_modifyitems(config, items):
    for item in items:
        if getattr(item.cls, "slow", False):
            item.add_marker(pytest.mark.slow)
    if "slow" in (config.getoption("markexpr") or ""):
        return
    skip = pytest.mark.skip(reason="slow: run with  python -m pytest -m slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _restore_hedge_share():
    """A research config read by pipeline.init_worker (in this process too)
    sets strategy.HEDGE_SHARE, and one written before the option existed
    sets 'discount': put the default back so no test leaks it into the next."""
    import strategy
    share = strategy.HEDGE_SHARE
    yield
    strategy.set_hedge_share(share)


@pytest.fixture(autouse=True, scope="class")
def _restore_annualization():
    """main.main / the dashboard's research set the process-wide
    annualization from the data they load (strategy.set_annualization:
    bars a year, session clock, calendar years): put back what a test class
    started with, so that a CSV run in one class does not leak its
    annualization into the next (class scope: it wraps the class's
    setUpClass / tearDownClass too)."""
    import strategy
    saved = strategy.current_annualization()
    yield
    strategy.set_annualization(saved)
