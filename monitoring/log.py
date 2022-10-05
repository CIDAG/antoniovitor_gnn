import abc
import json

# Abstract class
class Log(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def register(self, data):
        return

# Logs classes
class MongoLog(Log):
    def __init__(self, collection) -> None:
        super().__init__()
        # TODO

class FileLog(Log):
    def __init__(self, file_path) -> None:
        super().__init__()
        self.file_path = file_path
    
    def register(self, data):
        text = json.dump(data)
        with open(self.file_path, 'a') as file:
            file.write(text)

# Compose logs objects
class ComposeLogs(Log):
    def __init__(self, **logs_list) -> None:
        self.logs_list = logs_list

    def register(self, data):
        for log in self.logs_list:
            log.register(data)

        