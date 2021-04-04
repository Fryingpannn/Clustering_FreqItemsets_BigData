import os
import sys
import copy
import time
import random
import pyspark
import math
from statistics import mean
from pyspark.rdd import RDD
from pyspark.sql import Row
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.ml.fpm import FPGrowth
from pyspark.sql.functions import desc, size, max, abs, col

'''
INTRODUCTION

With this assignment you will get a practical hands-on of frequent 
itemsets and clustering algorithms in Spark. Before starting, you may 
want to review the following definitions and algorithms:
* Frequent itemsets: Market-basket model, association rules, confidence, interest.
* Clustering: kmeans clustering algorithm and its Spark implementation.

DATASET

We will use the dataset at 
https://archive.ics.uci.edu/ml/datasets/Plants, extracted from the USDA 
plant dataset. This dataset lists the plants found in US and Canadian 
states.

The dataset is available in data/plants.data, in CSV format. Every line 
in this file contains a tuple where the first element is the name of a 
plant, and the remaining elements are the states in which the plant is 
found. State abbreviations are in data/stateabbr.txt for your 
information.
'''

'''
HELPER FUNCTIONS

These functions are here to help you. Instructions will tell you when
you should use them. Don't modify them!
'''

def init_spark():
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    return spark

def toCSVLineRDD(rdd):
    a = rdd.map(lambda row: ",".join([str(elt) for elt in row]))\
           .reduce(lambda x,y: "\n".join([x,y]))
    return a + "\n"

def toCSVLine(data):
    if isinstance(data, RDD):
        if data.count() > 0:
            return toCSVLineRDD(data)
        else:
            return ""
    elif isinstance(data, DataFrame):
        if data.count() > 0:
            return toCSVLineRDD(data.rdd)
        else:
            return ""
    return None


'''
PART 1: FREQUENT ITEMSETS

Here we will seek to identify association rules between states to 
associate them based on the plants that they contain. For instance, 
"[A, B] => C" will mean that "plants found in states A and B are likely 
to be found in state C". We adopt a market-basket model where the 
baskets are the plants and the items are the states. This example 
intentionally uses the market-basket model outside of its traditional 
scope to show how frequent itemset mining can be used in a variety of 
contexts.
'''

def data_frame(filename, n):
    '''
    Write a function that returns a CSV string representing the first 
    <n> rows of a DataFrame with the following columns,
    ordered by increasing values of <id>:
    1. <id>: the id of the basket in the data file, i.e., its line number - 1 (ids start at 0).
    2. <plant>: the name of the plant associated to basket.
    3. <items>: the items (states) in the basket, ordered as in the data file.

    Return value: a CSV string. Using function toCSVLine on the right 
                  DataFrame should return the correct answer.
    Test file: tests/test_data_frame.py
    '''
    spark = init_spark()

    # read data from file
    plants_df = spark.read.text(filename).na.drop()
    # convert into rdd and split single string into list, first element is the plant
    plants_rdd = plants_df.rdd.map(lambda row: row["value"].split(","))
    # add index ((name, states, states, etc.), index)
    plants_rdd = plants_rdd.map(list).zipWithIndex()
    # re-order: (id, plant, [states]) ---> problem: states are in the plant column...
    plants_rdd = plants_rdd.map(lambda row: Row(id=row[1], plant=row[0][0], items=[state for state in row[0][1:] if state]))
    return toCSVLine(spark.sparkContext.parallelize(plants_rdd.take(n)))

def frequent_itemsets(filename, n, s, c):
    '''
    Using the FP-Growth algorithm from the ML library (see 
    http://spark.apache.org/docs/latest/ml-frequent-pattern-mining.html), 
    write a function that returns the first <n> frequent itemsets 
    obtained using min support <s> and min confidence <c> (parameters 
    of the FP-Growth model), sorted by (1) descending itemset size, and 
    (2) descending frequency. The FP-Growth model should be applied to 
    the DataFrame computed in the previous task. 
    
    Return value: a CSV string. As before, using toCSVLine may help.
    Test: tests/test_frequent_items.py
    '''
    spark = init_spark()

    # read data from file
    plants_df = spark.read.text(filename).na.drop()
    # convert into rdd and split single string into list, first element is the plant
    plants_rdd = plants_df.rdd.map(lambda row: row["value"].split(","))
    # add index ((name, states, states, etc.), index)
    plants_rdd = plants_rdd.map(list).zipWithIndex()
    # re-order: (id, plant, [states]) ---> problem: states are in the plant column...
    plants_rdd = plants_rdd.map(
        lambda row: Row(id=row[1], plant=row[0][0], items=[state for state in row[0][1:] if state]))

    # convert to DF
    plants_df = spark.createDataFrame(plants_rdd)
    # create fp growth model
    fpGrowth = FPGrowth(itemsCol="items", minConfidence=c, minSupport=s)
    model = fpGrowth.fit(plants_df)
    # get frequent itemsets
    frequent_df = model.freqItemsets
    # add a column indicating size of item list
    frequent_df = frequent_df.withColumn('size', size(frequent_df.items))
    # sort
    frequent_df = frequent_df.orderBy(desc('size'), desc('freq'))

    return toCSVLine(frequent_df.select('items', 'freq').limit(n))

def association_rules(filename, n, s, c):
    '''
    Using the same FP-Growth algorithm, write a script that returns the 
    first <n> association rules obtained using min support <s> and min 
    confidence <c> (parameters of the FP-Growth model), sorted by (1) 
    descending antecedent size in association rule, and (2) descending 
    confidence.

    Return value: a CSV string.
    Test: tests/test_association_rules.py
    '''
    spark = init_spark()

    # read data from file
    plants_df = spark.read.text(filename).na.drop()
    # convert into rdd and split single string into list, first element is the plant
    plants_rdd = plants_df.rdd.map(lambda row: row["value"].split(","))
    # add index ((name, states, states, etc.), index)
    plants_rdd = plants_rdd.map(list).zipWithIndex()
    # re-order: (id, plant, [states]) ---> problem: states are in the plant column...
    plants_rdd = plants_rdd.map(
        lambda row: Row(id=row[1], plant=row[0][0], items=[state for state in row[0][1:] if state]))

    # convert to DF
    plants_df = spark.createDataFrame(plants_rdd)
    # create fp growth model
    fpGrowth = FPGrowth(itemsCol="items", minConfidence=c, minSupport=s)
    model = fpGrowth.fit(plants_df)

    # get association rules
    rules = model.associationRules
    # get/sortby size of antecedents
    rules = rules.withColumn('size', size('antecedent'))
    rules = rules.orderBy(desc('size'), desc('confidence')).select('antecedent', 'consequent', 'confidence')

    return toCSVLine(rules.limit(n))

def interests(filename, n, s, c):
    '''
    Using the same FP-Growth algorithm, write a script that computes 
    the interest of association rules (interest = |confidence - 
    frequency(consequent)|; note the absolute value)  obtained using 
    min support <s> and min confidence <c> (parameters of the FP-Growth 
    model), and prints the first <n> rules sorted by (1) descending 
    antecedent size in association rule, and (2) descending interest.

    Return value: a CSV string.
    Test: tests/test_interests.py
    '''
    spark = init_spark()
    # read data from file
    plants_df = spark.read.text(filename)
    # convert into rdd and split single string into list, first element is the plant
    plants_rdd = plants_df.rdd.map(lambda row: row["value"].split(","))
    # add index ((name, states, states, etc.), index)
    plants_rdd = plants_rdd.map(list).zipWithIndex()
    # re-order: (id, plant, [states])
    plants_rdd = plants_rdd.map(lambda row: Row(id=row[1], plant=row[0][0], items=[state for state in row[0][1:] if state]))

    # convert to DF
    plants_df = spark.createDataFrame(plants_rdd)
    # create fp growth model
    fpGrowth = FPGrowth(itemsCol="items", minConfidence=c, minSupport=s)
    model = fpGrowth.fit(plants_df)

    # frequent itemsets
    freq_items = model.freqItemsets

    # get association rules
    rules = model.associationRules

    # compute interest: confidence - (count of consequent / total baskets)
    # get total items
    total_baskets = plants_rdd.count()

    # frequent itemsets
    freq_items = model.freqItemsets

    # calculating interest
    # getting the count of each consequent in all baskets.
    consequent_counts = freq_items.join(rules.select('consequent'), col('consequent') == col('items')).select('freq', 'consequent')#.groupBy('consequent').count()
    # join the counts onto association rule DF
    rules = rules.join(consequent_counts, on='consequent')
    # add a column representing the frequency
    rules = rules.withColumn('freq(consequent)', col('freq') / total_baskets)
    # add column representing interest
    rules = rules.withColumn('interest', abs(col('confidence') - col('freq(consequent)')))
    # final DF without sorting
    rules = rules.select("antecedent", "consequent", "confidence", "consequent", "freq", "interest")

    # sorting final DF by antecedent size then interest
    sorted_rules = rules.withColumn('ant_size', size('antecedent')).distinct()
    sorted_rules = sorted_rules.orderBy(desc('ant_size'), desc('interest'))
    sorted_rules = sorted_rules.select("antecedent", "consequent", "confidence", "consequent", "freq", "interest")

    return toCSVLine(sorted_rules.limit(n))

'''
PART 2: CLUSTERING

We will now cluster the states based on the plants that they contain.
We will reimplement and use the kmeans algorithm. States will be 
represented by a vector of binary components (0/1) of dimension D, 
where D is the number of plants in the data file. Coordinate i in a 
state vector will be 1 if and only if the ith plant in the dataset was 
found in the state (plants are ordered alphabetically, as in the 
dataset). For simplicity, we will initialize the kmeans algorithm 
randomly.

An example of clustering result can be visualized in states.png in this 
repository. This image was obtained with R's 'maps' package (Canadian 
provinces, Alaska and Hawaii couldn't be represented and a different 
seed than used in the tests was used). The classes seem to make sense 
from a geographical point of view!
'''
def get_dict(plants_df):
    '''
    Get all the states and the plants they have.
    Returns RDD of items as (name of the state, {dictionary})
    '''
    # convert into rdd and split single string into list, first element is the plant
    plants_rdd = plants_df.rdd.map(lambda row: row["value"].split(","))
    # re-order: (id, plant, [states]) ---> problem: states are in the plant column...
    plants_rdd = plants_rdd.map(lambda row: Row(plant=row[0], items=row[1:]))

    # states & plants list
    states_plants = plants_rdd.map(lambda row: (row['plant'], row['items'])).collect()

    # use flat map to separate list returns into individual items. iterate list to create dictionary: (name of the state, {dictionary})
    states_plants_final = plants_rdd.flatMap(lambda row: row[1]).distinct().map(lambda _state: (_state, {row[0]: (1 if _state in row[1] else 0) for row in states_plants}))
    return states_plants_final

def data_preparation(filename, plant, state):
    '''
    This function creates an RDD in which every element is a tuple with 
    the state as first element and a dictionary representing a vector 
    of plant as a second element:
    (name of the state, {dictionary})

    The dictionary should contains the plant names as keys. The 
    corresponding values should be 1 if the plant occurs in the state 
    represented by the tuple.

    You are strongly encouraged to use the RDD created here in the 
    remainder of the assignment.

    Return value: True if the plant occurs in the state and False otherwise.
    Test: tests/test_data_preparation.py
    '''
    spark = init_spark()
    # read data from file
    plants_df = spark.read.text(filename).na.drop()
    states_plants_final = get_dict(plants_df)
    # filter the given state if it has the given plant
    is_plant_in_state = states_plants_final.filter(lambda row: row[0] == state and row[1][plant] == 1)

    return True if is_plant_in_state.count() > 0 else False

def distance2(filename, state1, state2):
    '''
    This function computes the squared Euclidean
    distance between two states.
    
    Return value: an integer.
    Test: tests/test_distance.py
    '''
    spark = init_spark()
    # read data from file
    plants_df = spark.read.text(filename).na.drop()
    # get all the dictionaries for each state
    states_plants = get_dict(plants_df)
    # filter to get only for the 2 states we need and return dictionary values as list
    state1_vector = list(states_plants.filter(lambda row: row[0] == state1).collect()[0][1].values())
    state2_vector = list(states_plants.filter(lambda row: row[0] == state2).collect()[0][1].values())
    # compute distance: sqrt(a^2 + b^2 + c^2)
    res = 0
    for i in range(0, len(state1_vector)):
        res += (state1_vector[i] - state2_vector[i]) ** 2

    return res

def init_centroids(k, seed):
    '''
    This function randomly picks <k> states from the array in answers/all_states.py (you
    may import or copy this array to your code) using the random seed passed as
    argument and Python's 'random.sample' function.

    In the remainder, the centroids of the kmeans algorithm must be
    initialized using the method implemented here, perhaps using a line
    such as: `centroids = rdd.filter(lambda x: x[0] in
    init_states).collect()`, where 'rdd' is the RDD created in the data
    preparation task.

    Note that if your array of states has all the states, but not in the same
    order as the array in 'answers/all_states.py' you may fail the test case or
    have issues in the next questions.

    Return value: a list of <k> states.
    Test: tests/test_init_centroid.py
    '''
    all_states = ["ab", "ak", "ar", "az", "ca", "co", "ct", "de", "dc", "fl",
                  "ga", "hi", "id", "il", "in", "ia", "ks", "ky", "la", "me", "md",
                  "ma", "mi", "mn", "ms", "mo", "mt", "ne", "nv", "nh", "nj", "nm",
                  "ny", "nc", "nd", "oh", "ok", "or", "pa", "pr", "ri", "sc", "sd",
                  "tn", "tx", "ut", "vt", "va", "vi", "wa", "wv", "wi", "wy", "al",
                  "bc", "mb", "nb", "lb", "nf", "nt", "ns", "nu", "on", "qc", "sk",
                  "yt", "dengl", "fraspm"]

    random.seed(seed)
    init = random.sample(all_states, 3)
    return init

def first_iter(filename, k, seed):
    '''
    This function assigns each state to its 'closest' class, where 'closest'
    means 'the class with the centroid closest to the tested state
    according to the distance defined in the distance function task'. Centroids
    must be initialized as in the previous task.

    Return value: a dictionary with <k> entries:
    - The key is a centroid.
    - The value is a list of states that are the closest to the centroid. The list should be alphabetically sorted.

    Test: tests/test_first_iter.py
    '''
    all_states = [ "ab", "ak", "ar", "az", "ca", "co", "ct", "de", "dc", "fl",
        "ga", "hi", "id", "il", "in", "ia", "ks", "ky", "la", "me", "md",
        "ma", "mi", "mn", "ms", "mo", "mt", "ne", "nv", "nh", "nj", "nm",
        "ny", "nc", "nd", "oh", "ok", "or", "pa", "pr", "ri", "sc", "sd",
        "tn", "tx", "ut", "vt", "va", "vi", "wa", "wv", "wi", "wy", "al",
        "bc", "mb", "nb", "lb", "nf", "nt", "ns", "nu", "on", "qc", "sk",
        "yt", "dengl", "fraspm" ]
    def closest_centroid(state, centroids):
        '''
        Helper function for first_iter(). Finds the closest centroid vector to 'state'.
        - state: dictionary form of a state.
        - centroids: the centroids from which to find the closest one. list of vectors.
        '''

        def distance(state_vector, centroid):
            '''
            Helper function for closest_centroid(): finds squared euclidian distance between two vectors.
            '''
            res = 0
            for i in range(0, len(state_vector)):
                res += (state_vector[i] - centroid[i]) ** 2
            return res

        '''
        ------------------------------------------
        start of closest_centroid
        '''
        # convert state to vector form
        state_vector = list(state.values())

        closest = None
        min_dist = float('inf')
        for centroid in centroids:
            temp_dist = distance(state_vector, centroid)
            if temp_dist < min_dist:
                min_dist = temp_dist
                closest = centroid
        # return as tuple so it's hashable (used for reduceByKey)
        return tuple(closest)

    def get_state(vector, first_vectors):
        '''
        Helper function for first_iter(): Converts vector back into state string.
        - returns the index for the state string.
        '''
        for i, centroid in enumerate(first_vectors):
            if centroid == vector:
                return i
        return -50

    '''
    --------------------------------------------
    start of first_iter()
    '''
    spark = init_spark()
    random.seed(seed)
    plants_df = spark.read.text(filename).na.drop()
    states_plants = get_dict(plants_df)
    # cache rdd
    states_plants.persist()

    # initialize centroids (states): states -> vectors
    # filter rows where the state are centroids, then get their vector -> returns list of vectors [[1,0,1][0,0,1]etc]
    first_centroids = sorted(init_centroids(k, seed))
    first_vectors = states_plants.filter(lambda row: row[0] in first_centroids).sortByKey().map(
        lambda row: list(row[1].values())).collect()

    # filter only the states asked
    first = states_plants.filter(lambda row: row[0] in all_states)
    # list of ((1,0,0,1), 'fl') -> returns all states with their closest centroid vector
    first = first.map(lambda row: (closest_centroid(row[1], first_vectors), row[0]))
    # -> list of 'k' ((1,0,1), 'fl, nc, kek') -> returns k centroids with their associated states as a string
    first = first.reduceByKey(lambda x, y: x + ',' + y)
    first = first.map(lambda row: [first_centroids[get_state(list(row[0]), first_vectors)], sorted(row[1].split(','))])

    return dict(first.collect())


def closest_centroid2(state_vector, centroids):
    '''
    Helper function for kmeans(). Finds the closest centroid vector to 'state'.
    - state: dictionary form of a state.
    - centroids: the centroids from which to find the closest one. list of vectors.
    - returns: (centroid, state_vector): closest centroid vector of the state_vector
    '''

    def distance2(state_vector, centroid):
        '''
        Helper function for closest_centroid(): finds squared euclidian distance between two vectors.
        '''
        res = 0
        for i in range(0, len(state_vector)):
            res += (state_vector[i] - centroid[i]) ** 2
        return res

    '''
    ------------------------------------------
    start of closest_centroid
    '''
    closest = None
    min_dist = float('inf')
    for i, centroid in enumerate(centroids):
        temp_dist = distance2(state_vector, centroid)
        if temp_dist < min_dist:
            min_dist = temp_dist
            closest = i
    # return index: only needed for reduceByKey. 'i' will be diferent for each centroids.
    return closest


def add_pairs(x, state_vector_tup):
    '''
    Both pairs' format: (state_vector, count, state_str)
    '''
    acc_vector = x[0]
    total_count = x[1]
    (state_vector, count, notusedstate_str) = state_vector_tup

    new_vector = []
    for i in range(len(acc_vector)):
        new_vector.append(acc_vector[i] + state_vector[i])
    total_count += count

    return (new_vector, total_count)


def kmeans(filename, k, seed):
    all_states = ["ab", "ak", "ar", "az", "ca", "co", "ct", "de", "dc", "fl",
                  "ga", "hi", "id", "il", "in", "ia", "ks", "ky", "la", "me", "md",
                  "ma", "mi", "mn", "ms", "mo", "mt", "ne", "nv", "nh", "nj", "nm",
                  "ny", "nc", "nd", "oh", "ok", "or", "pa", "pr", "ri", "sc", "sd",
                  "tn", "tx", "ut", "vt", "va", "vi", "wa", "wv", "wi", "wy", "al",
                  "bc", "mb", "nb", "lb", "nf", "nt", "ns", "nu", "on", "qc", "sk",
                  "yt", "dengl", "fraspm"]
    '''
    This function:
    1. Initializes <k> centroids.
    2. Assigns states to these centroids as in the previous task.
    3. Updates the centroids based on the assignments in 2.
    4. Goes to step 2 if the assignments have not changed since the previous iteration.
    5. Returns the <k> classes.

    Note: You should use the list of states provided in all_states.py to ensure that the same initialization is made.

    Return value: a list of lists where each sub-list contains all states (alphabetically sorted) of one class.
                  Example: [["qc", "on"], ["az", "ca"]] has two 
                  classes: the first one contains the states "qc" and 
                  "on", and the second one contains the states "az" 
                  and "ca".
    Test file: tests/test_kmeans.py
    '''
    spark = init_spark()
    plants_df = spark.read.text(filename).na.drop()
    # RDD of (name of the state, [vector rep of the state])
    states_plants = get_dict(plants_df).filter(lambda row: row[0] in all_states).map(
        lambda row: (row[0], [float(i) for i in list(row[1].values())]))
    states_plants.persist()

    # Initialization
    # -> returns list of vectors [[1,0,1][0,0,1]] representing the initial centroids
    random.seed(seed)
    first_centroids = random.sample(all_states, k)
    centroids = states_plants.filter(lambda row: row[0] in first_centroids).map(lambda row: row[1]).collect()

    # kmeans
    # stop_at = 3
    old_centroids = []
    while True:
        # -> returns: list of (index representing a closest centroid, (vector of the state, 1, string name of state))
        closest = states_plants.map(lambda row: (closest_centroid2(row[1], centroids), (row[1], 1, row[0])))
        # -> returns: list of k centroids and their accumulated vectors (centroid index, (accumulated vector, total_count))
        # (could return less than k if one or more centroid was not close to any item)
        centroid_pairs = closest.reduceByKey(lambda x, y: add_pairs(x, y))
        # now, get average for the accumulated vectors to obtain updated centroids
        centroids = []
        for pair in centroid_pairs.collect():
            acc_vector = pair[1][0]
            count = pair[1][1]
            updated_centroid = list(map(lambda i: i / count, acc_vector))
            centroids.append(updated_centroid)

        # end loop if converged
        if sorted(centroids) == sorted(old_centroids):
            break
        old_centroids = centroids

    # return as list of lists, each inner list represents a class
    return closest.map(lambda row: (row[0], row[1][2])).reduceByKey(lambda x, y: x + ',' + y).map(
        lambda row: sorted(row[1].split(','))).collect()
