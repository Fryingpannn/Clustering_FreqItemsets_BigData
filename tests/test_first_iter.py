import sys
sys.path.insert(0, './answers')
from answer import first_iter

def test_first_iter():
    res = {'ct': ['al', 'ar', 'ct', 'dc', 'de', 'fl', 'ga', 'il', 'in', 'ky', 'la', 'ma', 'md', 'me', 'mi', 'mo', 'ms', 'nb', 'nc', 'nh', 'nj', 'ns', 'ny', 'oh', 'on', 'pa', 'qc', 'ri', 'sc', 'tn', 'va', 'vt', 'wi', 'wv'], 'nd': ['ab', 'ak', 'az', 'bc', 'ca', 'co', 'dengl', 'fraspm', 'ia', 'id', 'ks', 'lb', 'mb', 'mn', 'mt', 'nd', 'ne', 'nf', 'nm', 'nt', 'nu', 'nv', 'ok', 'or', 'sd', 'sk', 'tx', 'ut', 'wa', 'wy', 'yt'], 'hi': ['hi', 'pr', 'vi']}
    a = first_iter("./data/plants.data", 3, 123)
    assert(a == res)
