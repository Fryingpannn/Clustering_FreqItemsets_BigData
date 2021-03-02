import sys
sys.path.insert(0, './answers')
from answer import data_frame

def test_data_frame():
    a = data_frame("./data/plants.data", 11)
    assert(a==open("tests/data_frame.txt","r").read())
