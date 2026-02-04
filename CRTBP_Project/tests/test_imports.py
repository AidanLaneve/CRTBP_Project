# tests

def test_imports_and_constant():
    import crtbp
    import crtbp.constants as c
    import crtbp.dynamics as d
    import crtbp.propagate as p
    
    assert hasattr(c, "MU_EARTH_MOON")
    assert isinstance(c.MU_EARTH_MOON, float)
    assert callable(d.crtbp_eom)
    assert callable(d.jacobi_constant)
    assert callable(p.propagate)
    