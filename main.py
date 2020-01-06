from flask_heroku import app
import logging
from nltk import download

if __name__ == "__main__":
    # logging.basicConfig(filename = 'app.log', level = logging.INFO)
    try:
        download('stopwords')
        download('wordnet')
        download('punkt')
        app.run(port=4444, debug=True)
        
    except Exception as e:
        print(str(e))

