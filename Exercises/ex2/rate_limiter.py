'''
Exercise 2: Rate Limiter (Decorator)
Create a decorator @rate_limit(calls_per_minute) that:
Limits how many times a function can be called per minute.
Raises an exception if limit exceeded.
Hint: Use time.time() and a simple list to track call timestamps.
'''

def fns():
    # simulating a function that we want to limit
    print("Function executed")
    
# Now we need to create a decorator that limits the number of calls to fns function
import time
def rate_limit(calls_per_minute):
    interval = 60 / calls_per_minute  # calculate the interval between calls i.e, if calls_per_minute is 5, interval will be 12 seconds