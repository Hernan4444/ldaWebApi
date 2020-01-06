from constant import ALLOWED_EXTENSIONS
from hashlib import new, pbkdf2_hmac
from binascii import hexlify
from database import Database
from time import time
from random import randint


def encript_password(password):
    HASH = new('sha256')
    HASH.update(password.encode())
    halt = HASH.digest()
    hash_ = pbkdf2_hmac('sha256', halt, b'salt', 1000)
    password_encripty = hexlify(hash_).decode()
    return password_encripty


def generate_filename():
    random_string = str(Database.generate_secret_code(randint(10, 99)))
    return str(time()).replace(".", "") + random_string


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def is_xlsx(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == "xlsx"
