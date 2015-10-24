import numpy as np

class EmbeddingTable(object):
    """
    Reads in data path of word embeddings and
    constructs a matrix of these word embeddings.

    """
    # TODO: Add support for word2vec
    def __init__(self, dataPath, embeddingType="glove"):
        """
            :param dataPath: Path to file with precomputed vectors if relevant
            :param embeddingType: Shorthand for type of embedding
            being built. Will aim to support "glove", "word2vec", and "random"
            options.
        """
        self.type = embeddingType
        self.dataPath = dataPath

        vocab, embeddings, wordToIndex, indexToWord = self._readData(dataPath)

        # Vocabulary of all embeddings contained in table
        self.embeddingVocab = set(vocab) # TODO: maybe should make this a set?
        self.sizeVocab = embeddings.shape[0]
        self.embeddings = embeddings
        self.dimEmbeddings = embeddings.shape[1]

        # Mapping from word to embedding vector index
        self.wordToIndex = wordToIndex

        # Mapping from embedding vector index to word
        self.indexToWord = indexToWord


    # Note: There are word vectors for punctuation token -- how to handle those?
    def _readData(self, dataPath):
        """
        Reads in data and constructs a matrix of embeddings, vocabulary,
        and mappings from index:word and word:index
        """
        if dataPath is None:
            return (None, None, None, None)

        wordVocab = []
        wordVectors = []
        with open(dataPath, "r") as f:
            for word in f:
                wordContents = word.split()
                wordVocab.append(wordContents[0])
                wordVectors.append(wordContents[1:])

        wordVectors = np.array(wordVectors, dtype=np.float64)
        wordToIndex = dict(zip(wordVocab, range(len(wordVocab))))
        indexToWord = {idx: word for word, idx in wordToIndex.iteritems()}

        return wordVocab, wordVectors, wordToIndex, indexToWord


    def getWordEmbedding(self, word):
        """
        Return embedding vector for given word. If embedding not found,
        return a vector of random values.
        """
        # TODO: Is that proper way to handle unknown words?
        try:
            idx = self.wordToIndex[word]
            return self.embeddings[idx]
        except (KeyError, TypeError):
            print "Word not found in embedding matrix. Returning random vector..."
            return np.random.randn(self.dimEmbeddings)

