import numpy as np
import pytest
from numpy import array, arange, linspace, random
from scipy import interpolate

from material import Material
from disk import Disk


@pytest.fixture(scope='session')
def material():
    return Material('10Х11Н20ТЗР',
                    {
                        "density": 8400,
                        "alpha": 18 * 10 ** -6,
                        "E": interpolate.interp1d(arange(400, 800 + 1, 100),
                                                  array([1.74, 1.66, 1.57, 1.47, 1.32]) * 10 ** 11,
                                                  kind=3, bounds_error=False, fill_value='extrapolate'),
                        "mu": interpolate.interp1d(arange(400, 800 + 1, 100),
                                                   [0.384, 0.379, 0.371, 0.361, 0.347],
                                                   kind=3, bounds_error=False, fill_value='extrapolate'),
                        "sigma_s": 900 * 10 ** 6
                    })


@pytest.fixture(scope='function')
def radius():
    return np.sort(random.random(10))


@pytest.fixture(scope='function')
def thickness():
    return random.random(10)

@pytest.fixture(scope='function')
def disk():
    return Disk(material(), radius(), thickness())
