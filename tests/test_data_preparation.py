import sys
sys.path.insert(0, './answers')
from answer import data_preparation

def test_data_preparation():
    a = data_preparation("./data/plants.data", "urtica", "qc")
    assert(a)
    a = data_preparation("./data/plants.data", "zinnia maritima", "hi")
    assert(a)
    a = data_preparation("./data/plants.data", "tephrosia candida", "az")
    assert(a == False)
