import sys
sys.path.insert(0, './answers')
from answer import distance2

def test_distance():
    a = distance2("./data/plants.data", "qc", "on")
    assert(a == 1708)
    a = distance2("./data/plants.data", "ca", "az")
    assert(a == 10718)
