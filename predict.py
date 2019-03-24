from joblib import load


def get_sentiment(str_list, model_path='model.joblib'):
    """
    get sentiment of a list of sentences

    :param str_list: The list of sentences
    :type str_list: list
    :param model_path: The relative or absolute path to model dump as *.joblib
    :type model_path: str
    :return: The list of sentences corresponding of sentence index
    :rtype: list
    """
    text_clf = load(model_path)
    sentiment_values = text_clf.predict_proba(str_list)
    return sentiment_values[:, 1].tolist()

print(get_sentiment(['Bonjour les amis comment allez vous ?', "j'ai toujours du travail et je m'en sors pas"]))