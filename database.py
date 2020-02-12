from pandas import read_csv
from os.path import join
from os import remove
from json import dump, load
from random import randint, sample
from string import ascii_uppercase


class Database:
    def __init__(self):
        self.database = None
        self._load_database()

    def _load_database(self):
        self.database = read_csv('_database.tsv', encoding="UTF-8", sep="\t", dtype=str)

    @staticmethod
    def generate_secret_code(number):
        first_number, second_number = randint(10, 99), randint(10, 99)
        id_ = number + first_number + second_number
        first_letters = "".join(sample(ascii_uppercase, 2))
        second_letters = "".join(sample(ascii_uppercase, 2))
        code = "{}{}{}{}{}".format(first_number, first_letters, id_, second_letters, second_number)
        return code

    def save(self, df, filename, password, exist_filename, indexs, is_teacher_pool):
        if df is None:
            return

        if exist_filename != "":
            new_filename = exist_filename
            indexs_filename = exist_filename + ".idx"
        else:
            index = self.database.shape[0]  # numbers of rows
            code = Database.generate_secret_code(index)
            new_filename = 'file_{}'.format(index)
            indexs_filename = 'file_{}.idx'.format(index)

        df.to_csv(join("data", new_filename), sep="\t")
        with open(join("data", indexs_filename), "w", encoding="UTF-8") as file:
            dump(indexs, file)

        if exist_filename == "":
            with open('_database.tsv', "a") as file:
                file.write("\n{}\t{}\t{}\t{}\t{}".format(
                    code, filename, new_filename, password, is_teacher_pool))
            self._load_database()

    def load_file(self, filename):
        df_data = self.database[self.database.file_name_client.astype(str) == filename].iloc[0]
        is_teacher_pool = df_data.is_teacher_pool == 'True'

        df = read_csv(join("data", df_data.file_name_backend),
                      encoding="UTF-8", sep="\t", index_col=0)
        with open(join("data", df_data.file_name_backend + ".idx"), encoding="UTF-8") as file:
            idx = load(file)

        return df, idx, is_teacher_pool

    def found(self, name_client, password):
        same_name_client = self.database.database_name_client.astype(str) == name_client
        same_password = self.database.password.astype(str) == password
        return self.database[(same_password) & (same_name_client)]

    def get_databases(self, password):
        df = self.database[self.database.password.astype(str) == password]
        return df[['file_name_client', 'database_name_client']]

    def remove(self, name_client, password):
        df = self.found(name_client, password)
        if df.shape[0] != 1:
            return False

        filename = join("data", df.iloc[0].file_name_backend)
        remove(filename)
        remove(filename+".idx")

        different_password = self.database.password != password
        different_name_client = self.database.database_name_client != name_client

        new_data = self.database[different_password | different_name_client]
        new_data.to_csv('_database.tsv', sep="\t", index=False)
        self._load_database()
        return True
