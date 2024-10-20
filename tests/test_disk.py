from disk.disk import Disk


def test_initialisation_disk(material, radius, thickness):
    assert Disk(material=material, radius=radius, thickness=thickness)
    assert Disk(material, radius=radius, thickness=thickness)
    assert Disk(material, radius=radius, thickness=thickness)
    assert Disk(material, radius, thickness)
    assert Disk(material, thickness, radius)

def test_disk_tensions(disk):
    assert disk