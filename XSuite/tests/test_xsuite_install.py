import math

import xpart as xp
import xtrack as xt


def _make_line():
    line = xt.Line(
        elements=[
            xt.Drift(length=1.0),
            xt.Quadrupole(length=0.2),
            xt.Drift(length=1.0),
            xt.Quadrupole(length=0.2),
            xt.Drift(length=1.0),
        ],
        element_names=["d1", "qf", "d2", "qd", "d3"],
    )
    line.particle_ref = xp.Particles(
        p0c=5e9,
        mass0=xp.PROTON_MASS_EV,
        q0=1,
    )
    line.vars["kqf"] = 0.15
    line.vars["kqd"] = -0.15
    line.element_refs["qf"].k1 = line.vars["kqf"]
    line.element_refs["qd"].k1 = line.vars["kqd"]
    line.build_tracker()
    return line


def test_xsuite_twiss_and_match_work():
    line = _make_line()

    twiss_before = line.twiss(
        method="4d",
        betx=2.0,
        alfx=0.0,
        bety=3.0,
        alfy=0.0,
    )

    assert math.isclose(twiss_before.s[-1], 3.4, rel_tol=0, abs_tol=1e-12)
    assert twiss_before.betx[-1] > 0
    assert twiss_before.bety[-1] > 0

    line.match(
        method="4d",
        betx=2.0,
        alfx=0.0,
        bety=3.0,
        alfy=0.0,
        vary=[
            xt.Vary("kqf", step=1e-5),
            xt.Vary("kqd", step=1e-5),
        ],
        targets=[
            xt.Target("betx", 6.0, at=xt.END, tol=1e-6),
            xt.Target("bety", 8.0, at=xt.END, tol=1e-6),
        ],
    )

    twiss_after = line.twiss(
        method="4d",
        betx=2.0,
        alfx=0.0,
        bety=3.0,
        alfy=0.0,
    )

    assert math.isclose(twiss_after.betx[-1], 6.0, rel_tol=0, abs_tol=1e-5)
    assert math.isclose(twiss_after.bety[-1], 8.0, rel_tol=0, abs_tol=1e-5)
