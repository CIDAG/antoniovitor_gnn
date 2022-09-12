class Trie():
    def __init__(self, list):
        self.root = dict()
        for item in list:
            word, data = item
            current_dict = self.root
            for char in word:
                current_dict = current_dict.setdefault(char, {})
            current_dict['data'] = data
    
    def get(self, word):
        current_dict = self.root
        for char in word:
            if char not in current_dict:
                return None
            
            current_dict = current_dict[char]
        
        return current_dict.get('data')