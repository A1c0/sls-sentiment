import json
from predict import get_sentiment

def main(event, context):
    sentences_array = eval(event["body"])
    response = {
        "statusCode": 200,
        "body": json.dumps(get_sentiment(sentences_array))
    }
    return response
