import sys
from math import sqrt, log
from heapq import heappop, heappush, heapify
from PostingsList import PostingsList


class PostingsLists:
    """
    This class manages an inverted index for a collection of documents. It stores and processes
    the posting lists for terms (words), computes TF-IDF scores, and supports retrieving
    the most relevant documents for search queries.
    """

    def __init__(self):
        """
        Initializes the PostingsList object. It sets up the necessary data structures for storing
        terms, their corresponding posting lists, document frequencies, and other metadata.
        """
        self.total_terms = 0  # The total number of unique terms in the index
        self.document_weights = dict()  # Stores the TF-IDF weights for each document
        self.term_to_postings = dict()  # Maps each term to its PostingList
        self.document_term_frequencies = dict()  # Tracks term frequency for each document
        self.term_champion_lists = dict()  # Stores the champion list for each term
        self.word_frequencies = []  # List of word frequencies across documents
        self.valid_documents = []  # List of valid documents to consider

    def set_valid_documents(self, document_ids):
        """
        Sets the list of valid documents. These are the documents considered for indexing
        and retrieval.

        Parameters:
        - document_ids (list): A list of valid document IDs.
        """
        self.valid_documents = document_ids

    def add_term_to_index(self, term, document_id, position):
        """
        Adds a term occurrence to the index. If the term or document doesn't exist, they are initialized.

        Parameters:
        - term (str): The word being indexed.
        - document_id (str): The document where the term appears.
        - position (int): The position of the term within the document.
        """
        # Initialize a new PostingList if the term is not in the index
        if self.term_to_postings.get(term) is None:
            posting = PostingsList(term)
            self.total_terms += 1  # Increment total terms
            self.term_to_postings[term] = posting  # Add the PostingList to the index

        # Initialize the document entry if the document is not in the term frequencies
        if self.document_term_frequencies.get(document_id) is None:
            self.document_term_frequencies[document_id] = dict()

        # Increment the frequency of the term in the document
        if self.document_term_frequencies[document_id].get(term) is None:
            self.document_term_frequencies[document_id][term] = 0
        self.document_term_frequencies[document_id][term] += 1

        # Add the term to the posting list with the position information
        self.term_to_postings[term].add(document_id, position)

    def __str__(self):
        """
        Returns a string representation of the PostingsList object.

        This method provides a visual representation of all terms and their corresponding
        posting lists.

        Returns:
        - str: A string representing the entire PostingsList.
        """
        result = "{"

        # Iterate through the terms and append each term and its posting list
        for idx, term in enumerate(self.term_to_postings.keys()):
            result += f'"{term}": {self.term_to_postings[term]}'
            if idx < len(self.term_to_postings) - 1:
                result += ', '  # Add a comma between key-value pairs

        result += "}"
        return result

    def calculate_term_relevance(self, document_id, term, total_documents):
        """
        Calculates the relevance score of a term in a given document based on TF and IDF.

        The formula used is: relevance_score = TF(term in doc) * IDF(term)

        Parameters:
        - document_id (str): The document ID.
        - term (str): The word to calculate the score for.
        - total_documents (int): The total number of documents in the collection (for IDF calculation).

        Returns:
        - float: The calculated relevance score for the term in the document.
        """
        relevance_score = self.calculate_term_frequency(document_id, term) * self.calculate_inverse_document_frequency(
            term, total_documents)
        return relevance_score

    def calculate_term_frequency(self, document_id, term):
        """
        Computes the Term Frequency (TF) for a given term in a document.

        Parameters:
        - document_id (str): The document ID.
        - term (str): The term to calculate the term frequency for.

        Returns:
        - float: The term frequency for the given term in the document.
        """
        if self.term_to_postings.get(term) is None:
            return 0
        return self.term_to_postings[term].tf(document_id)

    def calculate_inverse_document_frequency(self, term, total_documents):
        """
        Computes the Inverse Document Frequency (IDF) for a term in the corpus.

        IDF is calculated as: IDF(term) = log(total_documents / number of documents containing the term)

        Parameters:
        - term (str): The term to calculate the inverse document frequency for.
        - total_documents (int): The total number of documents in the collection.

        Returns:
        - float: The inverse document frequency of the term.
        """
        if self.term_to_postings.get(term) is None:
            return 0
        return self.term_to_postings[term].idf(total_documents)

    def compute_document_weights(self, total_documents):
        """
        Computes the TF-IDF weights for each term in each document.

        The weight for each term is normalized by dividing by the document size (magnitude).

        Parameters:
        - total_documents (int): The total number of documents, used for IDF calculation.
        """
        for document_id in self.valid_documents:
            self.document_weights[str(document_id)] = dict()
            magnitude = 0
            for term in self.document_term_frequencies[str(document_id)].keys():
                term_weight = self.calculate_inverse_document_frequency(term,
                                                                        total_documents) * self.calculate_term_frequency(
                    str(document_id), term)
                self.document_weights[str(document_id)][term] = term_weight
                magnitude += term_weight * term_weight  # Compute squared sum for normalization

            # Normalize the weight (magnitude is the size of the document vector)
            magnitude = sqrt(magnitude)
            for term in self.document_term_frequencies[str(document_id)].keys():
                self.document_weights[str(document_id)][term] /= magnitude

    def get_document_weight(self, document_id, term):
        """
        Retrieves the TF-IDF weight of a term in a given document.

        Parameters:
        - document_id (str): The document ID.
        - term (str): The term to retrieve the weight for.

        Returns:
        - float: The TF-IDF weight of the term in the document, or 0 if not found.
        """
        if self.document_weights[document_id].get(term) is None:
            return 0
        return self.document_weights[document_id][term]

    def get_documents_for_term(self, term):
        """
        Retrieves a list of document IDs that contain a specific term.

        Parameters:
        - term (str): The term to retrieve the list of documents for.

        Returns:
        - list: A list of document IDs where the term appears.
        """
        if self.term_to_postings.get(term) is None:
            return []
        return self.term_to_postings[term].get_list_word()

    def generate_champion_lists(self, top_k):
        """
        Generates a champion list for each term. The champion list contains the top 'k' documents
        based on the frequency of the term in those documents.

        Parameters:
        - top_k (int): The number of top documents to store in the champion list for each term.
        """
        for term in self.term_to_postings.keys():
            self.term_champion_lists[term] = self.term_to_postings[term].create_champion_list(top_k)

    def get_champion_list_for_term(self, term):
        """
        Retrieves the champion list for a given term, which contains the top documents for that term.

        Parameters:
        - term (str): The term to retrieve the champion list for.

        Returns:
        - list: A list of document IDs in the champion list for the term, or an empty list if not found.
        """
        if self.term_champion_lists.get(term) is None:
            return []
        return [item[1] for item in self.term_champion_lists[term]]

    def create_word_frequency_list(self):
        """
        Creates a list of words along with their frequencies across the entire collection of documents.
        """
        for term in self.term_to_postings.keys():
            self.word_frequencies.append([self.term_to_postings[term].get_freq(), term])

    def remove_most_frequent_terms(self):
        """
        Finds the 50 most frequent terms in the collection and removes them from the index.

        Returns:
        - list: A list of the 50 most frequent terms that were removed from the index.
        """
        self.create_word_frequency_list()
        self.word_frequencies.sort(reverse=True)  # Sort terms by frequency in descending order
        removed_terms = []

        # Remove the top 50 most frequent terms from the index
        for i in range(50):
            frequency, term = self.word_frequencies[i]
            self.term_to_postings.pop(term)
            removed_terms.append((term, frequency))

        return removed_terms
