from flask_heroku import app
import logging

if __name__ == "__main__":
    # logging.basicConfig(filename = 'app.log', level = logging.INFO)
    try:
        app.run(port=4444, debug=True)
    except Exception as e:
        logging.exception(str(e))

