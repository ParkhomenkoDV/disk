from numpy import array, linspace, sort, pi, random
from scipy import interpolate
import pytest

from material import Material
from disk import Disk


@pytest.fixture(scope='session')
def material():
    return Material('10Х11Н20ТЗР',
                    {
                        "density": random.randint(300, 13_600),
                        "alpha": random.randint(2, 50) * 10 ** -6,
                        "E": interpolate.interp1d(linspace(400, 800, 4 + 1, endpoint=True),
                                                  random.uniform(1.2, 2.5, 4 + 1) * 10 ** 11,
                                                  kind=3, bounds_error=False, fill_value='extrapolate'),
                        "mu": interpolate.interp1d(linspace(400, 800, 4 + 1, endpoint=True),
                                                   [0.384, 0.379, 0.371, 0.361, 0.347],
                                                   kind=3, bounds_error=False, fill_value='extrapolate'),
                        "sigma_s": random.randint(600, 1_200) * 10 ** 6
                    })


@pytest.fixture(scope='function')
def discreteness():
    return random.randint(10, 150)


@pytest.fixture(scope='function')
def radius():
    return sort(random.random(20))


@pytest.fixture(scope='function')
def thickness():
    return random.random(20)


@pytest.fixture(scope='function')
def nholes():
    return random.randint(1, 36, random.randint(0,15))


@pytest.fixture(scope='function')
def rholes(nholes, radius):
    return random.uniform(min(radius), max(radius), len(nholes))


@pytest.fixture(scope='function')
def dholes(nholes, radius):
    return [random.uniform(0.00001, 2*pi*radius[i] / nholes[i])  for i in range(len(nholes))]


@pytest.fixture(scope='function')
def disk(material, radius, thickness) -> Disk:
    return Disk(material, radius, thickness)


@pytest.fixture(scope='function')
def rotation_frequency():
    return random.uniform(-1600, 1600)


@pytest.fixture(scope='function')
def temperature0():
    return random.uniform(280, 900)


@pytest.fixture(scope='function')
def pressure() -> tuple:
    return random.uniform(-60, 60) * 10 ** 6, random.uniform(60, 300) * 10 ** 6


@pytest.fixture(scope='function')
def temperature() -> tuple:
    return random.uniform(280, 900), random.uniform(280, 900)
