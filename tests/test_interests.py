import sys
sys.path.insert(0, './answers')
from answer import interests

def test_interests():
    a = interests("./data/plants.data", 15, 0.1, 0.3)
    assert(a==open("tests/interests.txt","r").read())
