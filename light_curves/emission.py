from mlxoplanet.starry import __getattr__ as _missing_starry


def light_curve(*args, **kwargs):
    del args, kwargs
    _missing_starry("emission")
