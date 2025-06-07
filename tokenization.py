import json
from hazm import *
from collections import Counter
from parsivar import FindStems
import re


def transform_space_to_half_space(text):
    # 1. Handle verb prefixes: "می" and "نمی" at the start of the string
    prefix_pattern_start = r'^(می|نمی)( )'
    prefix_replacement_start = r'\1‌'  # Replace the space with a half-space
    result = re.sub(prefix_pattern_start, prefix_replacement_start, text)

    # 2. Handle verb prefixes: "می" and "نمی" in the middle of the string
    prefix_pattern_middle = r'( )(می|نمی)( )'
    prefix_replacement_middle = r'\1\2‌'  # Replace the space after the prefix with a half-space
    result = re.sub(prefix_pattern_middle, prefix_replacement_middle, result)

    # 3. Handle common Persian suffixes like "های", "تر", "ام", "ات", "اش"
    suffix_pattern = r'( )(های|ها|هایی|ی|ای|تر|تری|ترین|گر|گری|ام|ات|اش)( )'
    suffix_replacement = r'‌\2\3'  # Replace the space before the suffix with a half-space
    result = re.sub(suffix_pattern, suffix_replacement, result)

    # 4. Handle verbs like "شده" and "نشده"
    verb_pattern = r'( )(شده|نشده)( )'
    verb_replacement = r'‌\2‌'  # Replace both spaces around the verb with half-spaces
    result = re.sub(verb_pattern, verb_replacement, result)

    # 5. Handle complex verbs like "می‌خواهد" and "نمی‌خواهید"
    complex_verb_pattern = r'( )(می‌خواهند|نمی‌خواهند|می‌خواهید|نمی‌خواهید|می‌خواهیم|نمی‌خواهیم|می‌خواهی|نمی‌خواهی|می‌خواهد|نمی‌خواهد|می‌خواهم|نمی‌خواهم)( )'
    complex_verb_replacement = r'‌\2‌'  # Replace spaces around complex verbs with half-spaces
    result = re.sub(complex_verb_pattern, complex_verb_replacement, result)

    return result


def change_unicode(text):
    # Convert English numerals to Persian numerals
    num_map = {'0': '۰', '1': '۱', '2': '۲', '3': '۳', '4': '۴',
               '5': '۵', '6': '۶', '7': '۷', '8': '۸', '9': '۹'}
    for en_num, fa_num in num_map.items():
        text = re.sub(en_num, fa_num, text)

    # Normalize Persian characters
    char_map = {
        r'ﻲ|ﯾ|ﯿ|ي': 'ی',
        r'ﻚ|ﮏ|ﻛ|ﮑ|ﮐ|ك': 'ک',
        r'ٲ|ٱ|إ|ﺍ|أ|آ': 'ا',
        r'ﺆ|ۊ|ۇ|ۉ|ﻮ|ؤ': 'و',
        r'ّ': '',  # Remove tashdid
    }
    for pattern, replacement in char_map.items():
        text = re.sub(pattern, replacement, text)

    # Replace specific phrases and special symbols
    phrase_map = {
        r'﷽': 'بسم االله الرحمن الرحیم',
        r'طهران': 'تهران',
        r'گفت‌وگو|گفت و گو|گفت‌و‌گو': 'گفتگو',
        r'جست‌وجو|جست و جو|جست‌و‌جو': 'جستجو',
        r'دشک': 'تشک',
        r'طوس': 'توس',
        r'باطری': 'باتری',
        r'توفان': 'طوفان',
        r'بلیط': 'بلیت',
        r'FIFA': 'فیفا'
    }
    for pattern, replacement in phrase_map.items():
        text = re.sub(pattern, replacement, text)

    return text


def separate_numbers(text):
    """
    Adds spaces around numbers (both English and Persian) in the input string.

    Parameters:
    text (str): The input string containing text and numbers.

    Returns:
    str: The modified string with spaces around the numbers.
    """
    # Match sequences of English or Persian digits and add spaces around them
    pattern = r'([0-9۰-۹]+)'
    result = re.sub(pattern, r' \1 ', text)
    return result


stemmer = FindStems()


def stemming(text):

    #lemmatizer = Lemmatizer()
    #stemmer = Stemmer()
    words = text.split()
    stemmed_words = [stemmer.convert_to_stem(word) for word in words]
    #lemmatized_words = [lemmatizer.lemmatize(word) for word in stemmed_words]
    return " ".join(stemmed_words)


def delete_punctuation(text):
    # Define a pattern that matches any punctuation or special character
    pattern = r'[@#$%^&*\(\)\-_=+\[\]{};:,.?/<>،«»؛؟"\'!]'

    # Replace all matching characters with a space
    cleaned_text = re.sub(pattern, ' ', text)

    # Remove extra spaces (if any) by splitting and rejoining
    cleaned_text = " ".join(cleaned_text.split())

    return cleaned_text


def normalizer(string):
    string = delete_punctuation(string)
    string = change_unicode(string)
    string = transform_space_to_half_space(string)
    string = separate_numbers(string)
    string = stemming(string)
    return string


def tokenizer(text):
    # Normalize the input text
    normalized_text = normalizer(text)

    # Replace all types of whitespace (spaces, tabs, newlines) with a single space
    cleaned_text = re.sub(r'\s+', ' ', normalized_text)

    # Split the cleaned text into tokens and remove empty tokens
    tokens = [word for word in cleaned_text.split() if word]

    return tokens


def count_word_frequencies(texts):
    all_tokens = []

    # Tokenize each document
    for text in texts:
        tokens = tokenizer(text)
        all_tokens.extend(tokens)

    # Count word frequencies using Counter
    word_frequencies = Counter(all_tokens)

    return word_frequencies


def report_top_frequent_words(texts, top_n=50):
    """
    Report the top N most frequent words from the provided texts.

    Parameters:
    - texts: A list of documents (strings).
    - top_n: The number of most frequent words to report (default is 50).

    Returns:
    - top_frequent_words: A list of the top N most frequent words.
    """
    # Step 1: Count word frequencies
    word_frequencies = count_word_frequencies(texts)

    # Step 2: Get the top N most frequent words
    top_frequent_words = [word for word, _ in word_frequencies.most_common(top_n)]

    # Step 3: Report the top frequent words
    print(f"Top {top_n} Most Frequent Words:")
    for word in top_frequent_words:
        print(word)

    return top_frequent_words



