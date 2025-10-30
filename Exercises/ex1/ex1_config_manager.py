'''
Exercise 1: Config Manager (OOP + File Handling)

Create a class ConfigManager that:
Reads configurations from a JSON file.
Allows getting and setting keys dynamically.
When updated, automatically saves back to file.
Hint: Use __getitem__, __setitem__, and context manager (with).
'''

import json

class ConfigManager:
    # file called identies.json is there, now we need to read the configuration from that file
    # I/P file path
    # O/P Either read or update the files
    
    def __init__(self,file_path):
        self.file_path = file_path
        with open(file_path,'r') as f: 
            print("Reads configurations from a JSON file.")
            self.configs = json.load(f)
            print("In constructor",self.configs) # read the json file and load into dictionary, here f is file object
            
    """Allows getting and setting keys dynamically."""
    def __getitem__(self,key): #__getitem__ is used to get the value of a key, double underscore methods are called magic methods, means special methods means we can use them directly i.e, obj[key]
        return self.configs.get(key,None)
    
    # Read and update the key in file
    def __setitem__(self,key,value): # __setitem__ is used to set the value of a key
        self.configs[key] = value
        with open(self.file_path,'w') as f: # open the file in write mode
            json.dump(self.configs,f,indent=4) # dump the updated dictionary into the file
            print(f"Updated {key} to {value} in {self.file_path}")
            
# Example usage:
if __name__ == "__main__": # this is used to run the code only when this file is run directly, not when imported as a module
    config = ConfigManager('Exercises/ex1/identites.json')
    print(config)  # Get the values based all keys
    config['Age'] = 889     # Update the value of 'Age'
    print(config['Age'])   # Get the updated value of 'Age'
                
'''         
# Topics covered: 
1. Object-Oriented Programming (OOP)
2. File Handling
3. JSON Manipulation # using json module to read and write json files
4. Magic Methods (__getitem__, __setitem__)
5. Context Managers (with statement)
'''