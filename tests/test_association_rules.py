import sys
sys.path.insert(0, './answers')
from answer import association_rules

def test_association_rules():
    a = association_rules("./data/plants.data", 15, 0.1, 0.3)
    assert(a==open("tests/association_rules.txt","r").read())
