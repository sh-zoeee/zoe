from scripts import create_strings, trees
from pqgrams.PQGram import Profile

#t = create_strings.generate_binaries(1, 3, 3)
#t = trees.string_to_pqtree(t[0])
t = trees.create_binary_tree(3, "_")
pqgram = Profile(t, p=2, q=3)

print(len(pqgram))
print(len(list(set(pqgram))))
