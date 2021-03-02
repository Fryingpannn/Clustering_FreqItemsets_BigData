import sys
sys.path.insert(0, './answers')
from answer import kmeans

def compare(r1, r2):
    if len(r1) != len(r2):
        print(str(len(r1)) + " - " + str(len(r2)))
    assert(len(r1) == len(r2))
    for i in range(len(r1)):
        list_found = False
        for j in range(len(r2)):
            res_list = r1[i]
            a_list = r2[j]
            if res_list == a_list:
                list_found = True
                break
        if not list_found:
            print(r2[j])
        assert(list_found)
def test_case1():
    res = [["ab", "bc", "mb", "mn", "qc", "sk" ],\
    ["ak", "dengl", "lb", "nt", "nu", "yt" ],\
    ["al", "fl", "ga", "ms", "nc", "sc" ],\
    ["ar", "ks", "ky", "la", "mo", "ok", "tn", "tx" ],\
    ["az", "ia", "nd", "ne", "sd" ],\
    ["ca", "co", "id", "mt", "nm", "nv", "or", "ut", "wa", "wy"],\
    ["ct", "dc", "de", "il", "in", "ma", "md", "me", "mi", "nh", "nj", "ny", "oh", "on", "pa", "ri", "va", "vt", "wi", "wv" ],\
    ["fraspm", "nb", "nf", "ns" ],\
    ["hi" ],\
    ["pr", "vi" ]]
    a = kmeans("./data/plants.data", 10, 123)
    compare(a,res)
def test_case2():
    res = [['ab', 'az', 'bc', 'ca', 'co', 'id', 'mt', 'nm', 'nv', 'or', 'ut', 'wa', 'wy'], ['ak', 'al', 'ar', 'ct', 'dc', 'de', 'dengl', 'fl', 'fraspm', 'ga', 'hi', 'ia', 'il', 'in', 'ks', 'ky', 'la', 'lb', 'ma', 'mb', 'md', 'me', 'mi', 'mn', 'mo', 'ms', 'nb', 'nc', 'nd', 'ne', 'nf', 'nh', 'nj', 'ns', 'nt', 'nu', 'ny', 'oh', 'ok', 'on', 'pa', 'pr', 'qc', 'ri', 'sc', 'sd', 'sk', 'tn', 'tx', 'va', 'vi', 'vt', 'wi', 'wv', 'yt']]
    a = kmeans("./data/plants.data", 2, 7070)
    compare(a,res)
def test_case3():
    res = [['ct', 'dc', 'de', 'ky', 'ma', 'md', 'nj', 'ny', 'pa', 'ri', 'va', 'wv'], ['me', 'nb', 'nf', 'nh', 'ns', 'qc', 'vt'], ['ak', 'dengl', 'fraspm', 'hi', 'lb', 'nt', 'nu', 'pr', 'vi', 'yt'], ['co', 'mt', 'wy'], ['nd', 'ne', 'nm', 'sd'], ['ia', 'il', 'in', 'ks', 'mo', 'ok'], ['bc'], ['mb', 'sk'], ['az', 'ca', 'nv', 'ut'], ['ab'], ['al', 'ar', 'fl', 'ga', 'la', 'ms', 'nc', 'sc', 'tn', 'tx'], ['mi', 'oh', 'on'], ['id', 'or', 'wa'], ['mn', 'wi']]
    a = kmeans("./data/plants.data", 14, 28447)
    compare(a,res)
