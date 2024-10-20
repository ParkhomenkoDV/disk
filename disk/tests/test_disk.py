import pytest

from disk import Disk


def test_initialisation_disk(material, radius, thickness):
    assert Disk(material=material, radius=radius, thickness=thickness)
    assert Disk(material, radius=radius, thickness=thickness)
    assert Disk(material, radius=radius, thickness=thickness)
    assert Disk(material, radius, thickness)
    assert Disk(material, thickness, radius)


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
