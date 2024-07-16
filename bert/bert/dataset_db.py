from typing import Iterable
import sqlite3


class BertDatasetDatabase:
    def __init__(self, path: str):
        self.path = path
        self.connection = None
        
    def connect(self):
        self.connection = sqlite3.connect(self.path, check_same_thread=False)
    
    def close(self):
        if self.connection is not None:
            self.connection.close()
        self.connection = None
        
    def init_table(self):
        if self.connection is None:
            raise RuntimeError('Database is not connected.')
        cursor = self.connection.cursor()
        statement = """
        CREATE TABLE IF NOT EXISTS dataset(
            id INTEGER PRIMARY KEY,
            sentence_1 STRING,
            sentence_2 STRING
        )
        """
        cursor.execute(statement)
        self.connection.commit()
    
    def add_sentence_pairs(self, sentence_pairs: Iterable[tuple[str, str]]) -> None:
        if self.connection is None:
            raise RuntimeError('Database is not connected.')
        cursor = self.connection.cursor()
        for pair in sentence_pairs:
            statement = """
            INSERT INTO dataset(
                sentence_1, sentence_2
            )
            VALUES(?, ?)
            """
            cursor.execute(statement, pair)
        self.connection.commit()
        
    def get(self, idx: int | Iterable[int]) -> tuple[str, str] | list[tuple[int, int]]:
        if self.connection is None:
            raise RuntimeError('Database is not connected.')
        
        cursor = self.connection.cursor()
        if isinstance(idx, (int, float)):
            statement = f"""
            SELECT sentence_1, sentence_2
            FROM dataset
            WHERE id == {idx + 1}
            """
            cursor.execute(statement)
            return cursor.fetchone()
        else:
            statement = f"""
                SELECT sentence_1, sentence_2
                FROM dataset
                WHERE id IN ({', '.join(str(index + 1) for index in idx)})
            """
            cursor.execute(statement)
            return cursor.fetchall()
    
    def __len__(self) -> int:
        if self.connection is None:
            raise RuntimeError('Database is not connected.')
        
        cursor = self.connection.cursor()
        statement = """
            SELECT COUNT(*)
            FROM dataset
        """
        cursor.execute(statement)
        return cursor.fetchone()[0]


if __name__ == '__main__':
    path = r'C:\Users\Ноутбук\Desktop\enviroment\sqlite_db\data.db'
    database = BertDatasetDatabase(path)
    database.connect()
    database.init_table()
    sentence_pairs = [('a', 'b'), ('b', 'c'), ('c', 'd')]
    database.add_sentence_pairs(sentence_pairs)
    print(database.get([0, 1, 2]))
    print(len(database))
    database.close()
