from numpy import pi, random

import pytest

from disk import Disk

@pytest.mark.classes
def test_initialisation_disk(material, radius, thickness, nholes, rholes, dholes):
    assert Disk(material=material, radius=radius, thickness=thickness)
    assert Disk(material, radius=radius, thickness=thickness)
    assert Disk(material, radius=radius, thickness=thickness)
    assert Disk(material, radius, thickness)
    assert Disk(material, thickness, radius)
    #assert Disk(material, thickness, radius, nholes, rholes, dholes)


@pytest.mark.xfail
def test_fail_initialisation_disk(material, radius, thickness):
    assert Disk()
    assert Disk(material)
    assert Disk(radius)
    assert Disk(material, radius[::-1], thickness)


def test_disk_tensions(disk, rotation_frequency, temperature0, pressure, temperature, discreteness):
    assert disk.tensions(rotation_frequency=rotation_frequency, temperature0=temperature0,
                         pressure=pressure, temperature=temperature,
                         discreteness=discreteness, show=False)


@pytest.mark.skip('b < 0')
def test_disk_local_tension(disk, nholes, rholes, dholes):
    for i in range(len(nholes)):
        b = 2 * pi * rholes[i] / nholes[i] - dholes[i]
        if not 0 < b:
            print(b)
        assert disk.local_tension(nholes[i], rholes[i], dholes[i],
                                  random.uniform(-800, 800) * 10 ** 6, random.uniform(-800, 800) * 10 ** 6)
