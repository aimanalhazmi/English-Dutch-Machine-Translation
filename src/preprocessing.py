from nltk.corpus import stopwords


en_stop_words = set(stopwords.words("english"))
nl_stop_words = set(stopwords.words("dutch"))

def remove_stop_words(text, stopWords) -> str:
    if not isinstance(text, str):
        return ""
    tokens = text.lower().split()
    tokens = [word for word in tokens if word not in stopWords]
    return " ".join(tokens)