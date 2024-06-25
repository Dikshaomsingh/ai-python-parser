import nltk
import sys
import re
from nltk import data
data.path += ['/tokenizers/punkt/PY3/english.pickle']

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

NONTERMINALS = """
S -> NP VP
NP -> N | Det N | Adj NP | Conj NP | P NP | NP Adv | Adv NP | NP Conj | NP VP
NP -> Det Adj NP | Det N Conj NP | Det Adj N Conj NP 
VP -> V | Adv V | VP NP
"""

# S -> NP VP
# NP -> N | Det N | Adj NP | Conj N | P N | NP Adv | NP Conj | NP VP
# NP -> Det Adj N | Det N Conj | Det Adj N Conj | Conj Det N | Conj Det Adj N
# NP -> P Det N | P Adj N | P Det Adj N | P N Conj | P Det N Conj | P Adj N Conj | P Det Adj N Conj
# NP -> Adv Det N | Adv Adj N | Adv Det Adj N | Adv N Conj | Adv Det N Conj | Adv Adj N Conj | Adv Det Adj N Conj
# VP -> V | Adv V | VP NP

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():

    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()

    # Otherwise, get sentence as input
    else:
        s = input("Sentence: ")

    # Convert input into list of words
    s = preprocess(s)

    # Attempt to parse sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


def preprocess(sentence):
    """
    Convert `sentence` to a list of its words.
    Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.
    """

    tokens = nltk.tokenize.word_tokenize(sentence.lower())
    res = []
    for word in tokens:
        if re.search("[A-Za-z]", word) is not None:
            res.append(word)
    return res
    # raise NotImplementedError


def np_chunk(tree):
    """
    Return a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is defined as any subtree of the sentence
    whose label is "NP" that does not itself contain any other
    noun phrases as subtrees.
    """

    sub_chunk = tree.subtrees()
    res = []
    remove_list = []
    for phrases in sub_chunk:
        if phrases.label() == 'NP' and phrases.label() != 'N':
            c_phrase = " ".join(phrases.flatten())
            for nodes in res:
                c_node = " ".join(nodes.flatten())
                if c_phrase in c_node:
                    if nodes not in remove_list:
                        remove_list.append(nodes)
            res.append(phrases)
    
    for item in remove_list:
        res.remove(item)

    return res

    # raise NotImplementedError


if __name__ == "__main__":
    main()
