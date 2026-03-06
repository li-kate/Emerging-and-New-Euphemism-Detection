"""
First Pass: Identify Potential Euphemisms
This script scans a given text for potential euphemisms based on a predefined list of common euphemistic phrases.
The output is a list of identified euphemisms along with their context in the text.

Method: 
1. Define a list of common taboo topics.
2. Scan the input text for words that are potential euphemisms for these topics with embedding-based similarity.
3. Store the identified euphemisms along with their context and time stamps for the second pass that will go through 
   the entire text again and pick up all instances of the words identified in the first pass.

"""

taboo_words_path = "taboo_words.txt"