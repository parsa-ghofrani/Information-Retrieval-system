import json
import pickle
import time
from heapq import heapify, heappush, heappop
from math import log, sqrt
from tokenization import *  # Assuming tokenization functions (tokenizer, normalizer) are imported
from PostingsLists import PostingsLists


# Global variables
document_number = 12202
map_doc = None
news_data = []


def read_data(path):
    """
    Reads and loads the data from a given JSON file.
    """
    try:
        with open(path) as file:
            data = json.load(file)
    except FileNotFoundError:
        print("Error: File not found.")
        exit()
    except Exception as e:
        print(f"Error: Failed to load news data from file. {e}")
        exit()

    return data


def gather_valid_indices(num, data):
    """
    Collects the indices of valid news articles from the data.
    """
    global news_data
    for i in range(num):
        if data.get(str(i)) is not None:
            news_data.append(i)
    return news_data


def preprocess(map_doc, datas):
    """
    Preprocesses the dataset by tokenizing the content of each news article,
    updating the inverted index, and saving the index to a binary file using pickle.
    """
    global news_data
    # Iterate over all valid news indices (news_in contains indices of valid news)
    for i in news_data:
        if i % 1000 == 0:
            print(f"{(i / document_number) * 100:.2f}% processed")

        # If the news article doesn't exist in the data, skip it
        if datas.get(str(i)) is None:
            continue

        # Get the content of the news article
        content = datas[str(i)]['content']

        # Tokenize the content using the tokenizer function
        content_tokens = tokenizer(content)

        # Initialize position counter for each word in the content
        position = 0

        # Add each word in the tokenized content to the inverted index
        for word in content_tokens:
            map_doc.add_term_to_index(word, str(i), position)
            position += 1

    # After processing all articles, save the inverted index using pickle
    save_inverted_index(map_doc.term_to_postings)

    return map_doc


def save_inverted_index(inverted_index):
    """
    Saves the inverted index to a binary file using pickle.
    """
    with open('postings', 'ab') as dbfile:
        pickle.dump(inverted_index, dbfile)


def search_query(query):
    """
    Processes a search query and ranks news articles based on relevance.
    """
    query = normalizer(query)
    token_query = tokenizer(query)

    word_count = len(token_query)
    word_frequencies = {word: token_query.count(word) for word in set(token_query)}

    word_count = sqrt(word_count)
    for word in word_frequencies:
        word_frequencies[word] /= word_count

    relevant_documents = []
    use_champion_list = False
    for word in word_frequencies:
        if use_champion_list:
            relevant_documents += map_doc.get_champion_list_for_term(word)
        else:
            relevant_documents += map_doc.get_documents_for_term(word)

    relevant_documents = list(dict.fromkeys(relevant_documents))

    heap = []
    heapify(heap)
    max_results = 3

    for doc_id in relevant_documents:
        score = 0
        for word in token_query:
            score += map_doc.get_document_weight(doc_id, word) * word_frequencies[word]
        heappush(heap, (score, doc_id))

        if len(heap) > max_results:
            heappop(heap)

    candidates = [heappop(heap) for _ in range(len(heap))]
    candidates.reverse()

    for candidate in candidates:
        doc_index = candidate[1]
        print(f"Title: {datas[doc_index]['title']}")
        print(f"URL: {datas[doc_index]['url']}")
        print(f"Content: {datas[doc_index]['content']}")




if __name__ == "__main__":
    # Step 1: Load data
    datas = read_data("IR_data_news_12k.json")

    # Step 2: Gather valid indices (collect document indices with data)
    news_data = gather_valid_indices(document_number, datas)

    # Step 3: Initialize the PostingsLists object
    map_doc = PostingsLists()

    # Step 4: Set valid documents in the PostingsLists object
    map_doc.set_valid_documents(news_data)

    # Step 5: Preprocess the data (tokenize and add to posting lists)
    map_doc = preprocess(map_doc, datas)

    # Step 6: Remove stop words from the posting list (most frequent terms)
    stop_words = map_doc.remove_most_frequent_terms()
    print("stop_words : ")
    print(stop_words)
    # Step 7: Compute document weights (e.g., TF-IDF)
    map_doc.compute_document_weights(document_number)

    # Step 8: Generate champion lists (for fast searching)
    map_doc.generate_champion_lists(20)

    # Step 9: Save the final posting lists to file
    save_inverted_index(map_doc.term_to_postings)

    print(f"Processed {document_number} documents, 100% done.")

    # Step 10: Interactive search loop
    while True:
        query = input("متن مورد نظر خود را برای جستجو وارد کنید:")
        search_query(query)
