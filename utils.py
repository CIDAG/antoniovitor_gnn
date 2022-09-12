import hashlib
import os


def generate_file_md5(path, blocksize=2**20):
    hash = hashlib.md5()
    with open(path, "rb") as file:
        while True:
            buffer = file.read(blocksize)
            if not buffer:
                break
            hash.update(buffer)
    
    return hash.hexdigest()