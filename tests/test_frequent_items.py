import sys
sys.path.insert(0, './answers')
from answer import frequent_itemsets

def test_frequent_items():
    a = frequent_itemsets("./data/plants.data", 15, 0.1, 0.3)
    assert(a==open("tests/frequent_items.txt","r").read())
