from math import log
from heapq import heappop, heappush, heapify


class PostingsList:

    def __init__(self, word):
        """
        Initializes the PostingList for a given word.
        The PostingList stores the word's frequencies in documents and other related information.
        """
        self.word = word  # The word for which the posting list is created
        self.freq_in_doc = {}  # Dictionary to store frequency of the word in each document
        self.freq_in_unique_doc = 0  # The number of documents containing the word
        self.freq_in_all = 0  # Total frequency of the word in all documents
        self.list = {}  # Dictionary to store positions of the word in each document
        self.champion_list = []  # A list of top documents with the highest frequency for this word

    def add(self, doc_id, position):
        """
        Adds the word's occurrence in a document.
        Updates the frequency of the word in the document and the overall posting list.
        """
        self.freq_in_all += 1  # Increment the word frequency in the entire corpus

        # If the word is not yet in this document, initialize its data
        if doc_id not in self.freq_in_doc:
            self.freq_in_doc[doc_id] = 0
            self.freq_in_unique_doc += 1  # Increment the count of unique documents containing the word
            self.list[doc_id] = []  # Initialize the list of positions for this document

        # Increment the frequency of the word in the document and add its position
        self.freq_in_doc[doc_id] += 1
        self.list[doc_id].append(position)

    def tf(self, doc_id):
        """
        Computes the Term Frequency (TF) for the word in the given document.
        TF is calculated as: 1 + log10(frequency in the document).
        """
        if doc_id not in self.freq_in_doc:
            return 0  # If the word is not in the document, return 0

        # Get the frequency of the word in the document
        freq = self.freq_in_doc[doc_id]
        return 1 + log(freq, 10)  # TF = 1 + log10(frequency)

    def idf(self, total_docs):
        """
        Computes the Inverse Document Frequency (IDF) for the word in the corpus.
        IDF is calculated as: log10(total_docs / number of documents containing the word).
        """
        return log(total_docs / self.freq_in_unique_doc, 10)

    def get_list_word(self):
        """
        Returns a list of document IDs that contain the word.
        """
        return list(self.list.keys())

    def create_champion_list(self, size):
        """
        Creates a champion list containing the top 'size' documents based on frequency of the word.
        The champion list helps in fast retrieval of the most relevant documents.
        """
        # Create a min-heap for the top 'size' documents based on frequency
        heap = []
        heapify(heap)

        for doc_id in self.list.keys():
            # Calculate the frequency of the word in the document
            score = self.freq_in_doc[doc_id]
            heappush(heap, (score, doc_id))

            # If the heap exceeds the specified size, remove the least frequent document
            if len(heap) > size:
                heappop(heap)

        # Extract the champion documents from the heap and reverse the order
        champion_docs = []
        while heap:
            champion_docs.append(heappop(heap))
        champion_docs.reverse()  # Reversing to get the highest frequency documents first

        return champion_docs

    def get_freq(self):
        """
        Returns the number of unique documents that contain the word.
        """
        return self.freq_in_unique_doc

    def __str__(self):
        """
        Provides a string representation of the PostingList object for debugging purposes.
        """
        return (f"Word: {self.word}, "
                f"Total Frequency: {self.freq_in_all}, "
                f"Unique Document Frequency: {self.freq_in_unique_doc}, "
                f"Document Frequencies: {self.freq_in_doc}, "
                f"Positions List: {self.list}")
