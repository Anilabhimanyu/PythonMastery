# Complete Guide to Async IO and Asynchronous Programming in Python

## Table of Contents
1. [Fundamentals](#fundamentals)
2. [Core Concepts](#core-concepts)
3. [Basic Async/Await Syntax](#basic-asyncawait-syntax)
4. [Event Loop Deep Dive](#event-loop-deep-dive)
5. [Working with Coroutines](#working-with-coroutines)
6. [Async Context Managers](#async-context-managers)
7. [Async Iterators and Generators](#async-iterators-and-generators)
8. [Concurrency Patterns](#concurrency-patterns)
9. [Error Handling](#error-handling)
10. [Performance and Best Practices](#performance-and-best-practices)
11. [Advanced Topics](#advanced-topics)
12. [Real-World Examples](#real-world-examples)

## Fundamentals

### What is Asynchronous Programming?

Asynchronous programming allows a program to handle multiple operations concurrently without blocking. Instead of waiting for one operation to complete before starting another, async programming lets you start multiple operations and handle them as they complete.

### Synchronous vs Asynchronous

```python
# Synchronous - blocking
import time
import requests

def fetch_sync(url):
    response = requests.get(url)
    return response.text

def main_sync():
    urls = ['http://httpbin.org/delay/1', 'http://httpbin.org/delay/2']
    start = time.time()
    
    for url in urls:
        result = fetch_sync(url)  # Blocks until complete
        print(f"Fetched {len(result)} characters")
    
    print(f"Total time: {time.time() - start:.2f}s")  # ~3 seconds

# Asynchronous - non-blocking
import asyncio
import aiohttp

async def fetch_async(session, url):
    async with session.get(url) as response:
        return await response.text()

async def main_async():
    urls = ['http://httpbin.org/delay/1', 'http://httpbin.org/delay/2']
    start = time.time()
    
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_async(session, url) for url in urls]
        results = await asyncio.gather(*tasks)  # Runs concurrently
        
        for result in results:
            print(f"Fetched {len(result)} characters")
    
    print(f"Total time: {time.time() - start:.2f}s")  # ~2 seconds

# Run async function
asyncio.run(main_async())
```

## Core Concepts

### Event Loop
The event loop is the heart of async programming in Python. It manages and executes coroutines, handles I/O operations, and coordinates concurrent tasks.

```python
import asyncio

# Getting the current event loop
loop = asyncio.get_event_loop()

# Creating a new event loop
new_loop = asyncio.new_event_loop()
asyncio.set_event_loop(new_loop)

# Running tasks in the event loop
async def hello():
    print("Hello")
    await asyncio.sleep(1)
    print("World")

# Different ways to run async code
asyncio.run(hello())  # Preferred method (Python 3.7+)

# Alternative methods
loop = asyncio.get_event_loop()
loop.run_until_complete(hello())
loop.close()
```

### Coroutines
Coroutines are functions that can be paused and resumed. They're defined with `async def` and called with `await`.

```python
import asyncio

async def coroutine_example():
    print("Start coroutine")
    await asyncio.sleep(1)  # Pause execution for 1 second
    print("End coroutine")
    return "Result"

# Coroutines are not executed until awaited
coro = coroutine_example()  # Creates coroutine object
print(type(coro))  # <class 'coroutine'>

# Execute the coroutine
result = asyncio.run(coro)
print(result)  # "Result"
```

### Tasks
Tasks are used to schedule coroutines concurrently. They wrap coroutines and allow them to run in the background.

```python
import asyncio

async def worker(name, delay):
    print(f"Worker {name} starting")
    await asyncio.sleep(delay)
    print(f"Worker {name} finished")
    return f"Result from {name}"

async def main():
    # Create tasks
    task1 = asyncio.create_task(worker("A", 2))
    task2 = asyncio.create_task(worker("B", 1))
    
    # Tasks start executing immediately
    print("Tasks created")
    
    # Wait for completion
    result1 = await task1
    result2 = await task2
    
    print(f"Results: {result1}, {result2}")

asyncio.run(main())
```

## Basic Async/Await Syntax

### Defining Async Functions

```python
import asyncio

# Basic async function
async def simple_async():
    await asyncio.sleep(1)
    return "Done"

# Async function with parameters
async def async_with_params(name, delay):
    print(f"Starting {name}")
    await asyncio.sleep(delay)
    print(f"Finished {name}")
    return name.upper()

# Calling async functions
async def main():
    result1 = await simple_async()
    result2 = await async_with_params("test", 0.5)
    print(f"Results: {result1}, {result2}")

asyncio.run(main())
```

### Await Rules

```python
import asyncio

async def correct_usage():
    # ✅ Correct - awaiting coroutine
    result = await asyncio.sleep(1)
    
    # ✅ Correct - awaiting task
    task = asyncio.create_task(asyncio.sleep(1))
    await task
    
    # ✅ Correct - awaiting future
    future = asyncio.Future()
    future.set_result("value")
    result = await future

async def common_mistakes():
    # ❌ Wrong - can't await regular function
    # result = await time.sleep(1)  # TypeError
    
    # ❌ Wrong - can't use await outside async function
    # result = await asyncio.sleep(1)  # SyntaxError
    
    # ❌ Wrong - forgetting await
    # asyncio.sleep(1)  # Returns coroutine object, doesn't execute
    pass
```

## Event Loop Deep Dive

### Event Loop Lifecycle

```python
import asyncio
import time

async def demonstrate_event_loop():
    print("Event loop demonstration")
    
    # Get current event loop
    loop = asyncio.get_running_loop()
    print(f"Loop: {loop}")
    
    # Schedule callbacks
    def callback(name):
        print(f"Callback {name} executed at {time.time():.2f}")
    
    # Schedule callbacks at different times
    loop.call_soon(callback, "soon")
    loop.call_later(1, callback, "later")
    loop.call_at(loop.time() + 2, callback, "at_time")
    
    # Wait for callbacks to execute
    await asyncio.sleep(3)

asyncio.run(demonstrate_event_loop())
```

### Custom Event Loop Integration

```python
import asyncio
import threading
import time

class CustomEventLoop:
    def __init__(self):
        self.loop = None
        self.thread = None
    
    def start(self):
        def run_loop():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_forever()
        
        self.thread = threading.Thread(target=run_loop)
        self.thread.start()
        
        # Wait for loop to be ready
        while self.loop is None:
            time.sleep(0.01)
    
    def stop(self):
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join()
    
    def run_coroutine(self, coro):
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future.result()

# Usage
async def async_task(name):
    print(f"Task {name} starting")
    await asyncio.sleep(1)
    print(f"Task {name} completed")
    return f"Result from {name}"

# Example usage (commented out as it requires threading)
# custom_loop = CustomEventLoop()
# custom_loop.start()
# result = custom_loop.run_coroutine(async_task("test"))
# custom_loop.stop()
```

## Working with Coroutines

### Creating and Managing Coroutines

```python
import asyncio
import inspect

async def example_coroutine(name, delay):
    print(f"Coroutine {name} started")
    await asyncio.sleep(delay)
    print(f"Coroutine {name} completed")
    return f"Result: {name}"

async def coroutine_management():
    # Creating coroutines
    coro1 = example_coroutine("A", 1)
    coro2 = example_coroutine("B", 2)
    
    # Check if it's a coroutine
    print(f"Is coroutine: {inspect.iscoroutine(coro1)}")
    
    # Running coroutines sequentially
    result1 = await coro1
    result2 = await coro2
    print(f"Sequential results: {result1}, {result2}")
    
    # Running coroutines concurrently
    coro3 = example_coroutine("C", 1)
    coro4 = example_coroutine("D", 2)
    
    results = await asyncio.gather(coro3, coro4)
    print(f"Concurrent results: {results}")

asyncio.run(coroutine_management())
```

### Coroutine States and Inspection

```python
import asyncio
import inspect

async def long_running_task():
    for i in range(5):
        print(f"Step {i}")
        await asyncio.sleep(0.5)
    return "Completed"

async def inspect_coroutines():
    # Create a task
    task = asyncio.create_task(long_running_task())
    
    # Inspect task state
    print(f"Task created: {task}")
    print(f"Task done: {task.done()}")
    print(f"Task cancelled: {task.cancelled()}")
    
    # Wait a bit and check again
    await asyncio.sleep(1)
    print(f"After 1s - Task done: {task.done()}")
    
    # Wait for completion
    result = await task
    print(f"Final result: {result}")
    print(f"Task done: {task.done()}")

asyncio.run(inspect_coroutines())
```

## Async Context Managers

### Basic Async Context Managers

```python
import asyncio
import aiofiles

class AsyncResource:
    def __init__(self, name):
        self.name = name
        self.is_open = False
    
    async def __aenter__(self):
        print(f"Opening resource {self.name}")
        await asyncio.sleep(0.1)  # Simulate async setup
        self.is_open = True
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        print(f"Closing resource {self.name}")
        await asyncio.sleep(0.1)  # Simulate async cleanup
        self.is_open = False
        if exc_type:
            print(f"Exception occurred: {exc_type.__name__}")

async def use_async_context_manager():
    async with AsyncResource("Database") as resource:
        print(f"Using resource: {resource.name}")
        print(f"Resource is open: {resource.is_open}")
        await asyncio.sleep(1)
    
    print(f"Resource is open after context: {resource.is_open}")

# File operations with async context managers
async def async_file_operations():
    # Writing to file
    async with aiofiles.open('example.txt', 'w') as f:
        await f.write("Hello, async world!")
    
    # Reading from file
    async with aiofiles.open('example.txt', 'r') as f:
        content = await f.read()
        print(f"File content: {content}")

asyncio.run(use_async_context_manager())
```

### Contextlib for Async Context Managers

```python
import asyncio
from contextlib import asynccontextmanager

@asynccontextmanager
async def database_connection(db_name):
    print(f"Connecting to {db_name}")
    await asyncio.sleep(0.5)  # Simulate connection time
    
    connection = f"Connection to {db_name}"
    try:
        yield connection
    finally:
        print(f"Disconnecting from {db_name}")
        await asyncio.sleep(0.2)  # Simulate cleanup

async def use_contextlib_manager():
    async with database_connection("PostgreSQL") as conn:
        print(f"Using {conn}")
        await asyncio.sleep(1)
        print("Work completed")

asyncio.run(use_contextlib_manager())
```

## Async Iterators and Generators

### Async Iterators

```python
import asyncio

class AsyncRange:
    def __init__(self, start, stop, delay=0.1):
        self.start = start
        self.stop = stop
        self.delay = delay
    
    def __aiter__(self):
        return self
    
    async def __anext__(self):
        if self.start >= self.stop:
            raise StopAsyncIteration
        
        current = self.start
        self.start += 1
        await asyncio.sleep(self.delay)
        return current

async def use_async_iterator():
    print("Using async iterator:")
    async for num in AsyncRange(1, 5):
        print(f"Number: {num}")

asyncio.run(use_async_iterator())
```

### Async Generators

```python
import asyncio
import random

async def async_number_generator(count):
    """Async generator that yields random numbers"""
    for i in range(count):
        await asyncio.sleep(0.1)  # Simulate async work
        yield random.randint(1, 100)

async def fetch_data_generator(urls):
    """Async generator for fetching data from URLs"""
    for url in urls:
        print(f"Fetching {url}")
        await asyncio.sleep(0.5)  # Simulate network delay
        yield f"Data from {url}"

async def use_async_generators():
    print("Async number generator:")
    async for num in async_number_generator(5):
        print(f"Generated: {num}")
    
    print("\nAsync data fetcher:")
    urls = ["api1.com", "api2.com", "api3.com"]
    async for data in fetch_data_generator(urls):
        print(data)

asyncio.run(use_async_generators())
```

### Async Comprehensions

```python
import asyncio

async def square_async(x):
    await asyncio.sleep(0.1)
    return x ** 2

async def async_comprehensions():
    # Async list comprehension
    numbers = [1, 2, 3, 4, 5]
    squared = [await square_async(x) for x in numbers]
    print(f"Squared (sequential): {squared}")
    
    # Async generator expression
    async def async_gen():
        for i in range(5):
            await asyncio.sleep(0.1)
            yield i * 2
    
    doubled = [x async for x in async_gen()]
    print(f"Doubled: {doubled}")
    
    # Conditional async comprehension
    filtered = [await square_async(x) for x in numbers if x % 2 == 0]
    print(f"Even squared: {filtered}")

asyncio.run(async_comprehensions())
```

## Concurrency Patterns

### Gather vs As_completed

```python
import asyncio
import random

async def worker(name, delay):
    await asyncio.sleep(delay)
    result = random.randint(1, 100)
    print(f"Worker {name} completed with result {result}")
    return result

async def gather_pattern():
    """Wait for all tasks to complete"""
    print("Using asyncio.gather:")
    tasks = [
        worker("A", 1),
        worker("B", 2),
        worker("C", 0.5)
    ]
    
    results = await asyncio.gather(*tasks)
    print(f"All results: {results}")

async def as_completed_pattern():
    """Process results as they become available"""
    print("\nUsing asyncio.as_completed:")
    tasks = [
        asyncio.create_task(worker("X", 1)),
        asyncio.create_task(worker("Y", 2)),
        asyncio.create_task(worker("Z", 0.5))
    ]
    
    for completed_task in asyncio.as_completed(tasks):
        result = await completed_task
        print(f"Got result: {result}")

async def main():
    await gather_pattern()
    await as_completed_pattern()

asyncio.run(main())
```

### Wait and Wait_for

```python
import asyncio

async def slow_operation():
    await asyncio.sleep(3)
    return "Slow result"

async def fast_operation():
    await asyncio.sleep(1)
    return "Fast result"

async def wait_patterns():
    tasks = [
        asyncio.create_task(slow_operation()),
        asyncio.create_task(fast_operation())
    ]
    
    # Wait for first completion
    done, pending = await asyncio.wait(
        tasks, 
        return_when=asyncio.FIRST_COMPLETED
    )
    
    print(f"First completed: {[task.result() for task in done]}")
    
    # Cancel remaining tasks
    for task in pending:
        task.cancel()
    
    # Wait with timeout
    try:
        result = await asyncio.wait_for(slow_operation(), timeout=2)
        print(f"Result: {result}")
    except asyncio.TimeoutError:
        print("Operation timed out!")

asyncio.run(wait_patterns())
```

### Semaphores and Locks

```python
import asyncio

# Semaphore for limiting concurrency
async def semaphore_example():
    semaphore = asyncio.Semaphore(2)  # Allow max 2 concurrent operations
    
    async def worker(name):
        async with semaphore:
            print(f"Worker {name} acquired semaphore")
            await asyncio.sleep(2)
            print(f"Worker {name} releasing semaphore")
    
    # Start 5 workers, but only 2 can run simultaneously
    tasks = [asyncio.create_task(worker(f"W{i}")) for i in range(5)]
    await asyncio.gather(*tasks)

# Lock for mutual exclusion
async def lock_example():
    lock = asyncio.Lock()
    shared_resource = 0
    
    async def increment_worker(name):
        nonlocal shared_resource
        for _ in range(5):
            async with lock:
                # Critical section
                current = shared_resource
                await asyncio.sleep(0.1)  # Simulate work
                shared_resource = current + 1
                print(f"{name}: {shared_resource}")
    
    tasks = [
        asyncio.create_task(increment_worker("Worker1")),
        asyncio.create_task(increment_worker("Worker2"))
    ]
    
    await asyncio.gather(*tasks)
    print(f"Final value: {shared_resource}")

async def main():
    await semaphore_example()
    print("\n" + "="*50 + "\n")
    await lock_example()

asyncio.run(main())
```

## Error Handling

### Exception Handling in Async Code

```python
import asyncio
import random

async def risky_operation(name):
    await asyncio.sleep(0.5)
    if random.random() < 0.5:
        raise ValueError(f"Error in {name}")
    return f"Success: {name}"

async def error_handling_patterns():
    # Basic try-catch
    try:
        result = await risky_operation("basic")
        print(result)
    except ValueError as e:
        print(f"Caught error: {e}")
    
    # Handling multiple coroutines
    tasks = [
        asyncio.create_task(risky_operation(f"task_{i}"))
        for i in range(5)
    ]
    
    # Method 1: gather with return_exceptions
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Task {i} failed: {result}")
        else:
            print(f"Task {i} succeeded: {result}")
    
    # Method 2: Handle exceptions individually
    new_tasks = [
        asyncio.create_task(risky_operation(f"individual_{i}"))
        for i in range(3)
    ]
    
    for task in asyncio.as_completed(new_tasks):
        try:
            result = await task
            print(f"Success: {result}")
        except ValueError as e:
            print(f"Failed: {e}")

asyncio.run(error_handling_patterns())
```

### Custom Exception Handling

```python
import asyncio
from typing import List, Union

class AsyncOperationError(Exception):
    def __init__(self, operation_name, original_error):
        self.operation_name = operation_name
        self.original_error = original_error
        super().__init__(f"Operation '{operation_name}' failed: {original_error}")

async def safe_async_operation(operation_name, should_fail=False):
    try:
        await asyncio.sleep(0.1)
        if should_fail:
            raise ConnectionError("Network timeout")
        return f"Success: {operation_name}"
    except Exception as e:
        raise AsyncOperationError(operation_name, e)

async def batch_with_error_recovery():
    operations = [
        ("op1", False),
        ("op2", True),   # This will fail
        ("op3", False),
        ("op4", True),   # This will fail
        ("op5", False)
    ]
    
    tasks = [
        asyncio.create_task(
            safe_async_operation(name, should_fail), 
            name=name
        )
        for name, should_fail in operations
    ]
    
    results = []
    for task in asyncio.as_completed(tasks):
        try:
            result = await task
            results.append(("success", result))
        except AsyncOperationError as e:
            results.append(("error", str(e)))
    
    # Print results
    for status, result in results:
        print(f"{status.upper()}: {result}")

asyncio.run(batch_with_error_recovery())
```

## Performance and Best Practices

### Efficient Async Patterns

```python
import asyncio
import time
from typing import List, Any

async def cpu_bound_task(n):
    """Simulate CPU-bound work"""
    await asyncio.sleep(0)  # Yield control
    return sum(i * i for i in range(n))

async def io_bound_task(delay):
    """Simulate I/O-bound work"""
    await asyncio.sleep(delay)
    return f"IO result after {delay}s"

async def performance_patterns():
    # ❌ Bad: Sequential execution of concurrent tasks
    start = time.time()
    results = []
    for i in range(3):
        result = await io_bound_task(0.5)
        results.append(result)
    print(f"Sequential: {time.time() - start:.2f}s")
    
    # ✅ Good: Concurrent execution
    start = time.time()
    tasks = [io_bound_task(0.5) for _ in range(3)]
    results = await asyncio.gather(*tasks)
    print(f"Concurrent: {time.time() - start:.2f}s")
    
    # ✅ Good: Batched processing
    async def process_batch(items, batch_size=10):
        results = []
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_tasks = [io_bound_task(0.1) for _ in batch]
            batch_results = await asyncio.gather(*batch_tasks)
            results.extend(batch_results)
        return results
    
    items = list(range(25))
    start = time.time()
    results = await process_batch(items, batch_size=5)
    print(f"Batched processing: {time.time() - start:.2f}s")

asyncio.run(performance_patterns())
```

### Resource Management

```python
import asyncio
from contextlib import asynccontextmanager

class ConnectionPool:
    def __init__(self, max_connections=10):
        self.max_connections = max_connections
        self.semaphore = asyncio.Semaphore(max_connections)
        self.active_connections = 0
    
    @asynccontextmanager
    async def get_connection(self):
        async with self.semaphore:
            self.active_connections += 1
            print(f"Acquired connection. Active: {self.active_connections}")
            
            # Simulate connection setup
            await asyncio.sleep(0.1)
            
            try:
                yield f"Connection-{self.active_connections}"
            finally:
                await asyncio.sleep(0.05)  # Cleanup
                self.active_connections -= 1
                print(f"Released connection. Active: {self.active_connections}")

async def use_connection_pool():
    pool = ConnectionPool(max_connections=3)
    
    async def worker(name):
        async with pool.get_connection() as conn:
            print(f"Worker {name} using {conn}")
            await asyncio.sleep(1)
            return f"Result from {name}"
    
    # Start 5 workers with pool of 3 connections
    tasks = [asyncio.create_task(worker(f"W{i}")) for i in range(5)]
    results = await asyncio.gather(*tasks)
    
    for result in results:
        print(result)

asyncio.run(use_connection_pool())
```

## Advanced Topics

### Async Queue Patterns

```python
import asyncio
import random

async def producer_consumer_pattern():
    # Create queue
    queue = asyncio.Queue(maxsize=5)
    
    async def producer(name, queue):
        for i in range(10):
            item = f"{name}-item-{i}"
            await queue.put(item)
            print(f"Producer {name} added {item}")
            await asyncio.sleep(random.uniform(0.1, 0.3))
        
        # Signal completion
        await queue.put(None)
    
    async def consumer(name, queue):
        while True:
            item = await queue.get()
            if item is None:
                # Shutdown signal
                await queue.put(None)  # Pass signal to other consumers
                break
            
            print(f"Consumer {name} processing {item}")
            await asyncio.sleep(random.uniform(0.2, 0.5))
            queue.task_done()
    
    # Start producer and consumers
    producer_task = asyncio.create_task(producer("P1", queue))
    consumer_tasks = [
        asyncio.create_task(consumer(f"C{i}", queue))
        for i in range(3)
    ]
    
    # Wait for producer to finish
    await producer_task
    
    # Wait for all items to be processed
    await queue.join()
    
    # Cancel consumers
    for task in consumer_tasks:
        task.cancel()

asyncio.run(producer_consumer_pattern())
```

### Task Groups (Python 3.11+)

```python
import asyncio

async def worker_with_potential_failure(name, should_fail=False):
    await asyncio.sleep(1)
    if should_fail:
        raise ValueError(f"Worker {name} failed")
    return f"Success: {name}"

async def task_groups_example():
    try:
        # TaskGroup ensures all tasks are cleaned up
        async with asyncio.TaskGroup() as tg:
            task1 = tg.create_task(worker_with_potential_failure("A"))
            task2 = tg.create_task(worker_with_potential_failure("B"))
            task3 = tg.create_task(worker_with_potential_failure("C", should_fail=True))
        
        # This line won't be reached due to exception
        print("All tasks completed successfully")
        
    except* ValueError as eg:
        print(f"Some tasks failed: {[str(e) for e in eg.exceptions]}")
        # All other tasks are automatically cancelled

# Note: TaskGroup is available in Python 3.11+
# For older versions, use asyncio.gather with return_exceptions=True
```

### Async Streaming

```python
import asyncio
import json

async def stream_processor():
    """Simulate streaming data processing"""
    
    async def data_stream():
        """Async generator simulating data stream"""
        for i in range(20):
            data = {"id": i, "value": random.randint(1, 100), "timestamp": time.time()}
            await asyncio.sleep(0.1)
            yield data
    
    async def process_item(item):
        """Process individual stream item"""
        await asyncio.sleep(0.05)  # Simulate processing
        return {**item, "processed": True, "doubled_value": item["value"] * 2}
    
    # Stream processing with concurrency control
    semaphore = asyncio.Semaphore(5)  # Limit concurrent processing
    
    async def bounded_process(item):
        async with semaphore:
            return await process_item(item)
    
    results = []
    async for item in data_stream():
        # Create task for concurrent processing
        task = asyncio.create_task(bounded_process(item))
        results.append(task)
        
        # Process in batches to avoid memory issues
        if len(results) >= 10:
            batch_results = await asyncio.gather(*results)
            for result in batch_results:
                print(f"Processed: {result}")
            results = []
    
    # Process remaining items
    if results:
        batch_results = await asyncio.gather(*results)
        for result in batch_results:
            print(f"Processed: {result}")

# Uncomment to run (requires imports)
# asyncio.run(stream_processor())
```

### Async Decorators

```python
import asyncio
import functools
import time

def async_retry(max_retries=3, delay=1):
    """Decorator for retrying async functions"""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        print(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                        await asyncio.sleep(delay)
                    else:
                        print(f"All {max_retries + 1} attempts failed")
            
            raise last_exception
        return wrapper
    return decorator

def async_timing(func):
    """Decorator to measure async function execution time"""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            duration = time.time() - start
            print(f"{func.__name__} took {duration:.2f}s")
    return wrapper

# Usage examples
@async_retry(max_retries=2, delay=0.5)
@async_timing
async def unreliable_operation():
    if random.random() < 0.7:
        raise ConnectionError("Random failure")
    return "Success!"

async def test_decorators():
    try:
        result = await unreliable_operation()
        print(f"Final result: {result}")
    except Exception as e:
        print(f"Operation ultimately failed: {e}")

# asyncio.run(test_decorators())
```

### Asyncio with Threading

```python
import asyncio
import threading
import time
import concurrent.futures

async def run_in_thread_example():
    """Running blocking operations in threads"""
    
    def blocking_operation(name, duration):
        """Simulate blocking I/O or CPU-bound work"""
        time.sleep(duration)
        return f"Blocking operation {name} completed"
    
    loop = asyncio.get_running_loop()
    
    # Method 1: run_in_executor with default thread pool
    result1 = await loop.run_in_executor(
        None, 
        blocking_operation, 
        "A", 
        1
    )
    print(result1)
    
    # Method 2: Custom thread pool executor
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        tasks = []
        for i in range(3):
            task = loop.run_in_executor(
                executor, 
                blocking_operation, 
                f"Worker-{i}", 
                0.5
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        for result in results:
            print(result)

asyncio.run(run_in_thread_example())
```

## Real-World Examples

### Web Scraping with Async

```python
import asyncio
import aiohttp
from urllib.parse import urljoin, urlparse
import re

class AsyncWebScraper:
    def __init__(self, max_concurrent=10, delay=0.1):
        self.max_concurrent = max_concurrent
        self.delay = delay
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def fetch_url(self, url):
        """Fetch a single URL with rate limiting"""
        async with self.semaphore:
            try:
                async with self.session.get(url) as response:
                    if response.status == 200:
                        content = await response.text()
                        await asyncio.sleep(self.delay)  # Rate limiting
                        return url, content
                    else:
                        return url, None
            except Exception as e:
                print(f"Error fetching {url}: {e}")
                return url, None
    
    async def scrape_multiple(self, urls):
        """Scrape multiple URLs concurrently"""
        tasks = [self.fetch_url(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful = []
        failed = []
        
        for result in results:
            if isinstance(result, Exception):
                failed.append(str(result))
            else:
                url, content = result
                if content:
                    successful.append((url, len(content)))
                else:
                    failed.append(url)
        
        return successful, failed

async def web_scraping_example():
    urls = [
        "http://httpbin.org/delay/1",
        "http://httpbin.org/delay/2", 
        "http://httpbin.org/json",
        "http://httpbin.org/html"
    ]
    
    async with AsyncWebScraper(max_concurrent=2) as scraper:
        successful, failed = await scraper.scrape_multiple(urls)
        
        print("Successful scrapes:")
        for url, content_length in successful:
            print(f"  {url}: {content_length} characters")
        
        print("Failed scrapes:")
        for failure in failed:
            print(f"  {failure}")

# asyncio.run(web_scraping_example())
```

### Async Database Operations

```python
import asyncio
import sqlite3
from contextlib import asynccontextmanager

class AsyncSQLiteWrapper:
    """Wrapper to make SQLite operations async-friendly"""
    
    def __init__(self, db_path):
        self.db_path = db_path
        self.loop = None
    
    async def execute(self, query, params=None):
        """Execute query in thread pool"""
        loop = asyncio.get_running_loop()
        
        def _execute():
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                
                if query.strip().upper().startswith('SELECT'):
                    return cursor.fetchall()
                else:
                    conn.commit()
                    return cursor.rowcount
        
        return await loop.run_in_executor(None, _execute)
    
    async def executemany(self, query, params_list):
        """Execute query with multiple parameter sets"""
        loop = asyncio.get_running_loop()
        
        def _executemany():
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.executemany(query, params_list)
                conn.commit()
                return cursor.rowcount
        
        return await loop.run_in_executor(None, _executemany)

async def database_example():
    db = AsyncSQLiteWrapper(':memory:')
    
    # Create table
    await db.execute('''
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE
        )
    ''')
    
    # Insert data concurrently
    users = [
        ("Alice", "alice@example.com"),
        ("Bob", "bob@example.com"),
        ("Charlie", "charlie@example.com")
    ]
    
    # Batch insert
    await db.executemany(
        "INSERT INTO users (name, email) VALUES (?, ?)",
        users
    )
    
    # Concurrent queries
    async def get_user_by_name(name):
        result = await db.execute(
            "SELECT * FROM users WHERE name = ?", 
            (name,)
        )
        return result[0] if result else None
    
    # Query multiple users concurrently
    names = ["Alice", "Bob", "Charlie"]
    tasks = [get_user_by_name(name) for name in names]
    results = await asyncio.gather(*tasks)
    
    for user in results:
        if user:
            print(f"User: {user}")

asyncio.run(database_example())
```

### Async API Server Pattern

```python
import asyncio
import json
from typing import Dict, Any

class AsyncAPIHandler:
    """Simulate async API request handling"""
    
    def __init__(self):
        self.request_count = 0
        self.cache = {}
    
    async def handle_request(self, request_id: str, endpoint: str, data: Dict[str, Any]):
        """Handle API request asynchronously"""
        self.request_count += 1
        
        print(f"[{request_id}] Handling {endpoint} (Request #{self.request_count})")
        
        # Simulate different endpoint behaviors
        if endpoint == "/fast":
            await asyncio.sleep(0.1)
            return {"status": "success", "data": "Fast response"}
        
        elif endpoint == "/slow":
            await asyncio.sleep(2)
            return {"status": "success", "data": "Slow response"}
        
        elif endpoint == "/cached":
            if endpoint in self.cache:
                print(f"[{request_id}] Cache hit")
                return self.cache[endpoint]
            
            await asyncio.sleep(1)
            result = {"status": "success", "data": "Cached response"}
            self.cache[endpoint] = result
            return result
        
        elif endpoint == "/error":
            await asyncio.sleep(0.5)
            raise ValueError("Simulated API error")
        
        else:
            return {"status": "error", "message": "Endpoint not found"}

async def simulate_api_server():
    """Simulate concurrent API requests"""
    handler = AsyncAPIHandler()
    
    # Simulate multiple concurrent requests
    requests = [
        ("req1", "/fast", {}),
        ("req2", "/slow", {}),
        ("req3", "/cached", {}),
        ("req4", "/cached", {}),  # Should hit cache
        ("req5", "/error", {}),
        ("req6", "/fast", {}),
    ]
    
    async def process_request(request_id, endpoint, data):
        try:
            result = await handler.handle_request(request_id, endpoint, data)
            print(f"[{request_id}] Success: {result}")
            return request_id, "success", result
        except Exception as e:
            print(f"[{request_id}] Error: {e}")
            return request_id, "error", str(e)
    
    # Process all requests concurrently
    tasks = [
        asyncio.create_task(process_request(req_id, endpoint, data))
        for req_id, endpoint, data in requests
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Summary
    success_count = sum(1 for _, status, _ in results if status == "success")
    print(f"\nProcessed {len(requests)} requests: {success_count} successful")

asyncio.run(simulate_api_server())
```

### Background Tasks and Services

```python
import asyncio
import signal
from datetime import datetime

class AsyncBackgroundService:
    """Background service that runs continuously"""
    
    def __init__(self, name, interval=1.0):
        self.name = name
        self.interval = interval
        self.running = False
        self.task = None
    
    async def start(self):
        """Start the background service"""
        if self.running:
            return
        
        self.running = True
        self.task = asyncio.create_task(self._run())
        print(f"Service {self.name} started")
    
    async def stop(self):
        """Stop the background service"""
        if not self.running:
            return
        
        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        
        print(f"Service {self.name} stopped")
    
    async def _run(self):
        """Main service loop"""
        try:
            while self.running:
                await self._work()
                await asyncio.sleep(self.interval)
        except asyncio.CancelledError:
            print(f"Service {self.name} cancelled")
            raise
    
    async def _work(self):
        """Override this method in subclasses"""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {self.name} working...")

class HealthCheckService(AsyncBackgroundService):
    """Example health check service"""
    
    def __init__(self):
        super().__init__("HealthCheck", interval=5.0)
        self.check_count = 0
    
    async def _work(self):
        self.check_count += 1
        # Simulate health check
        await asyncio.sleep(0.1)
        print(f"Health check #{self.check_count} completed")

class MetricsService(AsyncBackgroundService):
    """Example metrics collection service"""
    
    def __init__(self):
        super().__init__("Metrics", interval=2.0)
        self.metrics = {"requests": 0, "errors": 0}
    
    async def _work(self):
        # Simulate metrics collection
        self.metrics["requests"] += random.randint(1, 10)
        self.metrics["errors"] += random.randint(0, 2)
        print(f"Metrics: {self.metrics}")

async def background_services_example():
    """Run multiple background services"""
    
    # Create services
    health_service = HealthCheckService()
    metrics_service = MetricsService()
    
    # Start services
    await health_service.start()
    await metrics_service.start()
    
    try:
        # Main application logic
        print("Main application running...")
        await asyncio.sleep(10)  # Run for 10 seconds
        
    except KeyboardInterrupt:
        print("Received interrupt signal")
    finally:
        # Cleanup services
        await health_service.stop()
        await metrics_service.stop()

# Run with graceful shutdown
async def main_with_signals():
    """Main function with signal handling"""
    
    # Setup signal handlers for graceful shutdown
    shutdown_event = asyncio.Event()
    
    def signal_handler():
        print("\nReceived shutdown signal")
        shutdown_event.set()
    
    # Note: Signal handling works differently in different environments
    # This is a simplified example
    
    # Create background task
    task = asyncio.create_task(background_services_example())
    
    try:
        # Wait for either task completion or shutdown signal
        await asyncio.wait([task], return_when=asyncio.FIRST_COMPLETED)
    except KeyboardInterrupt:
        print("Keyboard interrupt")
        task.cancel()

# asyncio.run(main_with_signals())
```

## Testing Async Code

### Unit Testing Async Functions

```python
import asyncio
import unittest
from unittest.mock import AsyncMock, patch

class AsyncCalculator:
    async def add_async(self, a, b):
        await asyncio.sleep(0.1)  # Simulate async work
        return a + b
    
    async def fetch_and_calculate(self, url):
        # Simulate fetching data from URL
        await asyncio.sleep(0.2)
        # Mock response
        data = 42
        return await self.add_async(data, 10)

class TestAsyncCalculator(unittest.TestCase):
    def setUp(self):
        self.calculator = AsyncCalculator()
    
    async def async_test_add(self):
        result = await self.calculator.add_async(2, 3)
        self.assertEqual(result, 5)
    
    async def async_test_fetch_and_calculate(self):
        result = await self.calculator.fetch_and_calculate("http://example.com")
        self.assertEqual(result, 52)
    
    async def async_test_with_mock(self):
        with patch.object(self.calculator, 'add_async', new_callable=AsyncMock) as mock_add:
            mock_add.return_value = 100
            
            result = await self.calculator.fetch_and_calculate("http://example.com")
            
            mock_add.assert_called_once_with(42, 10)
            self.assertEqual(result, 100)
    
    # Wrapper methods to run async tests
    def test_add(self):
        asyncio.run(self.async_test_add())
    
    def test_fetch_and_calculate(self):
        asyncio.run(self.async_test_fetch_and_calculate())
    
    def test_with_mock(self):
        asyncio.run(self.async_test_with_mock())

# Alternative: Using pytest-asyncio
# pip install pytest-asyncio

"""
import pytest

class TestAsyncCalculatorPytest:
    @pytest.fixture
    def calculator(self):
        return AsyncCalculator()
    
    @pytest.mark.asyncio
    async def test_add_async(self, calculator):
        result = await calculator.add_async(2, 3)
        assert result == 5
    
    @pytest.mark.asyncio
    async def test_fetch_and_calculate(self, calculator):
        result = await calculator.fetch_and_calculate("http://example.com")
        assert result == 52
"""
```

### Debugging Async Code

```python
import asyncio
import logging
import warnings

# Enable asyncio debug mode
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def debug_async_operations():
    """Demonstrate debugging techniques"""
    
    # 1. Enable debug mode
    loop = asyncio.get_running_loop()
    loop.set_debug(True)
    
    # 2. Task naming for easier debugging
    async def named_task(name, delay):
        print(f"Task {name} starting")
        await asyncio.sleep(delay)
        print(f"Task {name} completed")
        return name
    
    # Create tasks with names
    tasks = [
        asyncio.create_task(named_task(f"Task-{i}", 0.5), name=f"Task-{i}")
        for i in range(3)
    ]
    
    # 3. Monitor task status
    for task in tasks:
        print(f"Created: {task.get_name()}")
    
    # 4. Use asyncio debug logging
    results = await asyncio.gather(*tasks)
    print(f"Results: {results}")
    
    # 5. Detect common issues
    async def problematic_function():
        # This will trigger a warning in debug mode
        await asyncio.sleep(0)
        return "done"
    
    # Missing await (will be caught in debug mode)
    coro = problematic_function()
    await coro  # Properly awaited
    
    # 6. Timeout debugging
    try:
        await asyncio.wait_for(asyncio.sleep(2), timeout=1)
    except asyncio.TimeoutError:
        print("Operation timed out (expected)")

# Run with debug information
asyncio.run(debug_async_operations(), debug=True)
```

## Common Patterns and Idioms

### Rate Limiting

```python
import asyncio
import time
from collections import deque

class AsyncRateLimiter:
    """Token bucket rate limiter for async operations"""
    
    def __init__(self, max_calls, time_window):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = deque()
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        async with self.lock:
            now = time.time()
            
            # Remove old calls outside time window
            while self.calls and self.calls[0] <= now - self.time_window:
                self.calls.popleft()
            
            # Check if we can make a call
            if len(self.calls) < self.max_calls:
                self.calls.append(now)
                return True
            
            # Calculate wait time
            oldest_call = self.calls[0]
            wait_time = oldest_call + self.time_window - now
            await asyncio.sleep(wait_time)
            
            # Try again
            return await self.acquire()

async def rate_limited_operations():
    # Allow 3 calls per 2 seconds
    limiter = AsyncRateLimiter(max_calls=3, time_window=2)
    
    async def api_call(name):
        await limiter.acquire()
        print(f"{time.time():.2f}: Making API call {name}")
        await asyncio.sleep(0.1)  # Simulate API call
        return f"Result from {name}"
    
    # Make 6 calls - should be rate limited
    tasks = [api_call(f"call-{i}") for i in range(6)]
    results = await asyncio.gather(*tasks)
    
    for result in results:
        print(result)

asyncio.run(rate_limited_operations())
```

### Circuit Breaker Pattern

```python
import asyncio
import random
from enum import Enum
from datetime import datetime, timedelta

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class AsyncCircuitBreaker:
    """Circuit breaker for async operations"""
    
    def __init__(self, failure_threshold=5, timeout=60, expected_exception=Exception):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
    
    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self):
        return (
            self.last_failure_time and
            datetime.now() - self.last_failure_time >= timedelta(seconds=self.timeout)
        )
    
    def _on_success(self):
        self.failure_count = 0
        self.state = CircuitState.CLOSED
    
    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN

async def unreliable_service():
    """Simulate unreliable external service"""
    await asyncio.sleep(0.1)
    if random.random() < 0.7:  # 70% failure rate
        raise ConnectionError("Service unavailable")
    return "Service response"

async def circuit_breaker_example():
    breaker = AsyncCircuitBreaker(failure_threshold=3, timeout=5)
    
    for i in range(10):
        try:
            result = await breaker.call(unreliable_service)
            print(f"Call {i}: Success - {result}")
        except Exception as e:
            print(f"Call {i}: Failed - {e} (State: {breaker.state.value})")
        
        await asyncio.sleep(0.5)

asyncio.run(circuit_breaker_example())
```

## Advanced Synchronization

### Async Condition Variables

```python
import asyncio
from collections import deque

class AsyncQueue:
    """Custom async queue implementation with condition variables"""
    
    def __init__(self, maxsize=0):
        self.maxsize = maxsize
        self.queue = deque()
        self.not_empty = asyncio.Condition()
        self.not_full = asyncio.Condition()
    
    async def put(self, item):
        async with self.not_full:
            while self.maxsize > 0 and len(self.queue) >= self.maxsize:
                await self.not_full.wait()
            
            self.queue.append(item)
            self.not_empty.notify()
    
    async def get(self):
        async with self.not_empty:
            while not self.queue:
                await self.not_empty.wait()
            
            item = self.queue.popleft()
            self.not_full.notify()
            return item

async def condition_variable_example():
    queue = AsyncQueue(maxsize=2)
    
    async def producer():
        for i in range(5):
            await queue.put(f"item-{i}")
            print(f"Produced item-{i}")
            await asyncio.sleep(0.5)
    
    async def consumer(name):
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=2)
                print(f"Consumer {name} got {item}")
                await asyncio.sleep(1)
            except asyncio.TimeoutError:
                print(f"Consumer {name} timed out")
                break
    
    # Run producer and consumers
    await asyncio.gather(
        producer(),
        consumer("A"),
        consumer("B")
    )

asyncio.run(condition_variable_example())
```

## Performance Optimization

### Profiling Async Code

```python
import asyncio
import time
import cProfile
import pstats
from functools import wraps

def async_profile(func):
    """Decorator to profile async functions"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        
        profiler.enable()
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            profiler.disable()
            
            # Print profiling results
            stats = pstats.Stats(profiler)
            stats.sort_stats('cumulative')
            stats.print_stats(10)  # Top 10 functions
    
    return wrapper

@async_profile
async def performance_test():
    """Function to profile"""
    
    async def cpu_intensive():
        # Simulate CPU work
        total = 0
        for i in range(100000):
            total += i * i
        await asyncio.sleep(0)  # Yield control
        return total
    
    async def io_intensive():
        await asyncio.sleep(0.1)
        return "IO result"
    
    # Mix of CPU and I/O operations
    tasks = []
    for i in range(10):
        if i % 2 == 0:
            tasks.append(asyncio.create_task(cpu_intensive()))
        else:
            tasks.append(asyncio.create_task(io_intensive()))
    
    results = await asyncio.gather(*tasks)
    return len(results)

# asyncio.run(performance_test())
```

### Memory Management

```python
import asyncio
import weakref
import gc

class AsyncResourceManager:
    """Manage async resources with proper cleanup"""
    
    def __init__(self):
        self.resources = weakref.WeakSet()
        self.cleanup_task = None
    
    async def create_resource(self, name):
        """Create a new async resource"""
        
        class AsyncResource:
            def __init__(self, name):
                self.name = name
                self.data = bytearray(1024 * 1024)  # 1MB of data
                self.closed = False
            
            async def close(self):
                if not self.closed:
                    print(f"Closing resource {self.name}")
                    await asyncio.sleep(0.01)  # Simulate cleanup
                    self.data = None
                    self.closed = True
            
            def __del__(self):
                if not self.closed:
                    print(f"Resource {self.name} was not properly closed!")
        
        resource = AsyncResource(name)
        self.resources.add(resource)
        return resource
    
    async def cleanup_resources(self):
        """Cleanup all managed resources"""
        for resource in list(self.resources):
            if hasattr(resource, 'close'):
                await resource.close()
    
    async def start_periodic_cleanup(self, interval=30):
        """Start periodic cleanup task"""
        async def cleanup_loop():
            while True:
                await asyncio.sleep(interval)
                gc.collect()  # Force garbage collection
                print(f"Cleanup: {len(self.resources)} resources active")
        
        self.cleanup_task = asyncio.create_task(cleanup_loop())
    
    async def stop_cleanup(self):
        """Stop periodic cleanup"""
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass

async def memory_management_example():
    manager = AsyncResourceManager()
    
    # Create some resources
    resources = []
    for i in range(5):
        resource = await manager.create_resource(f"Resource-{i}")
        resources.append(resource)
    
    print(f"Created {len(resources)} resources")
    
    # Properly close some resources
    for resource in resources[:3]:
        await resource.close()
    
    # Let some resources go out of scope without closing
    del resources[3:]
    
    # Force garbage collection
    gc.collect()
    
    # Cleanup remaining resources
    await manager.cleanup_resources()

asyncio.run(memory_management_example())
```

## Best Practices Summary

### Do's and Don'ts

```python
import asyncio

# ✅ DO: Use asyncio.run() for top-level async code
async def main():
    await some_async_operation()

asyncio.run(main())

# ✅ DO: Use asyncio.gather() for concurrent operations
async def concurrent_operations():
    tasks = [async_operation(i) for i in range(5)]
    results = await asyncio.gather(*tasks)
    return results

# ✅ DO: Use async context managers for resources
async def proper_resource_management():
    async with AsyncResource() as resource:
        await resource.do_work()

# ✅ DO: Handle exceptions properly
async def proper_exception_handling():
    try:
        result = await risky_operation()
    except SpecificException as e:
        # Handle specific error
        await handle_error(e)

# ❌ DON'T: Block the event loop with synchronous operations
async def bad_blocking():
    time.sleep(5)  # This blocks the entire event loop!

# ✅ DO: Use run_in_executor for blocking operations
async def good_blocking():
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, blocking_function)

# ❌ DON'T: Create tasks unnecessarily
async def unnecessary_tasks():
    task = asyncio.create_task(simple_async_func())
    return await task  # Just use: return await simple_async_func()

# ✅ DO: Use tasks when you need concurrent execution
async def proper_task_usage():
    task1 = asyncio.create_task(operation1())
    task2 = asyncio.create_task(operation2())
    
    # Do other work while tasks run
    await other_work()
    
    # Collect results
    result1 = await task1
    result2 = await task2
    return result1, result2
```

### Performance Tips

```python
import asyncio

async def performance_tips():
    """Demonstrate performance optimization techniques"""
    
    # 1. Batch operations when possible
    async def batch_database_operations(records):
        # ✅ Good: Batch insert
        await db.executemany("INSERT INTO table VALUES (?, ?)", records)
        
        # ❌ Bad: Individual inserts
        # for record in records:
        #     await db.execute("INSERT INTO table VALUES (?, ?)", record)
    
    # 2. Use connection pooling
    async def with_connection_pool():
        # Reuse connections instead of creating new ones
        async with aiohttp.ClientSession() as session:
            tasks = [fetch_url(session, url) for url in urls]
            return await asyncio.gather(*tasks)
    
    # 3. Limit concurrency with semaphores
    async def controlled_concurrency():
        semaphore = asyncio.Semaphore(10)  # Max 10 concurrent operations
        
        async def bounded_operation(item):
            async with semaphore:
                return await process_item(item)
        
        tasks = [bounded_operation(item) for item in large_dataset]
        return await asyncio.gather(*tasks)
    
    # 4. Use asyncio.create_task() for fire-and-forget operations
    async def background_processing():
        # Start background task
        background_task = asyncio.create_task(long_running_process())
        
        # Do immediate work
        immediate_result = await quick_operation()
        
        # Wait for background task if needed
        background_result = await background_task
        
        return immediate_result, background_result

# Placeholder functions for examples
async def some_async_operation(): return "result"
async def async_operation(i): return f"result-{i}"
async def risky_operation(): return "success"
async def handle_error(e): pass
def blocking_function(): return "blocking_result"
async def simple_async_func(): return "simple"
async def operation1(): return "op1"
async def operation2(): return "op2"
async def other_work(): pass
async def fetch_url(session, url): return f"data from {url}"
async def process_item(item): return f"processed {item}"
async def long_running_process(): return "background_result"
async def quick_operation(): return "quick_result"
```

## Integration with Popular Libraries

### FastAPI Integration

```python
import asyncio
from typing import List

# FastAPI example (conceptual - requires fastapi installation)
"""
from fastapi import FastAPI, BackgroundTasks
import uvicorn

app = FastAPI()

class AsyncDataProcessor:
    def __init__(self):
        self.processing_queue = asyncio.Queue()
        self.results = {}
    
    async def process_data(self, data_id: str, data: dict):
        # Simulate processing
        await asyncio.sleep(2)
        result = {"processed": True, "data_id": data_id, "result": data}
        self.results[data_id] = result
        return result

processor = AsyncDataProcessor()

@app.post("/process")
async def start_processing(data_id: str, data: dict, background_tasks: BackgroundTasks):
    # Start background processing
    background_tasks.add_task(processor.process_data, data_id, data)
    return {"message": "Processing started", "data_id": data_id}

@app.get("/result/{data_id}")
async def get_result(data_id: str):
    result = processor.results.get(data_id)
    if result:
        return result
    return {"status": "processing"}

@app.get("/health")
async def health_check():
    # Async health check
    await asyncio.sleep(0.1)
    return {"status": "healthy"}

# Run with: uvicorn main:app --reload
"""

# Alternative async web framework example
class SimpleAsyncServer:
    """Simple async HTTP-like server simulation"""
    
    def __init__(self):
        self.routes = {}
        self.middleware = []
    
    def route(self, path):
        def decorator(func):
            self.routes[path] = func
            return func
        return decorator
    
    async def handle_request(self, path, data=None):
        """Simulate handling HTTP request"""
        if path not in self.routes:
            return {"error": "Route not found", "status": 404}
        
        handler = self.routes[path]
        
        # Apply middleware
        for middleware in self.middleware:
            data = await middleware(data)
        
        try:
            result = await handler(data)
            return {"result": result, "status": 200}
        except Exception as e:
            return {"error": str(e), "status": 500}

# Usage example
server = SimpleAsyncServer()

@server.route("/users")
async def get_users(data):
    await asyncio.sleep(0.1)  # Simulate database query
    return [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]

@server.route("/process")
async def process_data(data):
    await asyncio.sleep(0.5)  # Simulate processing
    return {"processed": True, "input": data}

async def test_server():
    # Simulate concurrent requests
    requests = [
        server.handle_request("/users"),
        server.handle_request("/process", {"data": "test"}),
        server.handle_request("/nonexistent")
    ]
    
    responses = await asyncio.gather(*requests)
    for response in responses:
        print(response)

asyncio.run(test_server())
```

### Database Integration Patterns

```python
import asyncio
from contextlib import asynccontextmanager
from typing import List, Dict, Any

class AsyncDatabaseManager:
    """Database manager with connection pooling"""
    
    def __init__(self, max_connections=10):
        self.max_connections = max_connections
        self.connection_pool = asyncio.Queue(maxsize=max_connections)
        self.initialized = False
    
    async def initialize(self):
        """Initialize connection pool"""
        if self.initialized:
            return
        
        # Create connections (simulated)
        for i in range(self.max_connections):
            connection = f"connection-{i}"
            await self.connection_pool.put(connection)
        
        self.initialized = True
        print(f"Initialized pool with {self.max_connections} connections")
    
    @asynccontextmanager
    async def get_connection(self):
        """Get connection from pool"""
        if not self.initialized:
            await self.initialize()
        
        connection = await self.connection_pool.get()
        try:
            yield connection
        finally:
            await self.connection_pool.put(connection)
    
    async def execute_query(self, query: str, params: tuple = None):
        """Execute single query"""
        async with self.get_connection() as conn:
            # Simulate query execution
            await asyncio.sleep(0.1)
            return f"Query result for: {query}"
    
    async def execute_transaction(self, queries: List[tuple]):
        """Execute multiple queries in transaction"""
        async with self.get_connection() as conn:
            print(f"Starting transaction with {len(queries)} queries")
            
            try:
                results = []
                for query, params in queries:
                    await asyncio.sleep(0.05)  # Simulate query
                    results.append(f"Result: {query}")
                
                # Simulate commit
                await asyncio.sleep(0.02)
                print("Transaction committed")
                return results
                
            except Exception as e:
                print("Transaction rolled back")
                raise e

async def database_integration_example():
    db = AsyncDatabaseManager(max_connections=3)
    
    # Concurrent single queries
    async def worker(worker_id):
        queries = [
            f"SELECT * FROM users WHERE id = {i}"
            for i in range(worker_id * 3, (worker_id + 1) * 3)
        ]
        
        results = []
        for query in queries:
            result = await db.execute_query(query)
            results.append(result)
        
        return results
    
    # Start multiple workers
    workers = [asyncio.create_task(worker(i)) for i in range(4)]
    all_results = await asyncio.gather(*workers)
    
    print(f"Completed {len(all_results)} workers")
    
    # Transaction example
    transaction_queries = [
        ("INSERT INTO users (name) VALUES (?)", ("Alice",)),
        ("UPDATE users SET status = ? WHERE name = ?", ("active", "Alice")),
        ("INSERT INTO logs (action) VALUES (?)", ("user_created",))
    ]
    
    await db.execute_transaction(transaction_queries)

asyncio.run(database_integration_example())
```

## Common Pitfalls and Solutions

### Avoiding Common Mistakes

```python
import asyncio
import warnings

async def common_pitfalls_and_solutions():
    """Demonstrate common async pitfalls and their solutions"""
    
    # Pitfall 1: Forgetting to await
    async def forgot_await_bad():
        # ❌ This creates a coroutine but doesn't execute it
        result = asyncio.sleep(1)  # Missing await!
        return result
    
    async def forgot_await_good():
        # ✅ Properly awaited
        await asyncio.sleep(1)
        return "completed"
    
    # Pitfall 2: Using blocking operations
    async def blocking_operations_bad():
        # ❌ This blocks the entire event loop
        import time
        time.sleep(1)
        return "bad"
    
    async def blocking_operations_good():
        # ✅ Non-blocking equivalent
        await asyncio.sleep(1)
        return "good"
    
    # Pitfall 3: Not handling task cancellation
    async def cancellation_bad():
        try:
            await asyncio.sleep(10)
        except:
            pass  # ❌ Swallows CancelledError
    
    async def cancellation_good():
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            print("Task was cancelled")
            # ✅ Re-raise to properly handle cancellation
            raise
        except Exception as e:
            print(f"Other error: {e}")
    
    # Pitfall 4: Creating too many tasks
    async def too_many_tasks_bad():
        # ❌ Creates thousands of tasks at once
        tasks = [asyncio.create_task(small_operation()) for _ in range(10000)]
        return await asyncio.gather(*tasks)
    
    async def too_many_tasks_good():
        # ✅ Process in batches
        batch_size = 100
        items = list(range(10000))
        results = []
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_tasks = [asyncio.create_task(small_operation()) for _ in batch]
            batch_results = await asyncio.gather(*batch_tasks)
            results.extend(batch_results)
        
        return results
    
    # Pitfall 5: Sharing mutable state without synchronization
    class UnsafeCounter:
        def __init__(self):
            self.count = 0
        
        async def increment_bad(self):
            # ❌ Race condition
            current = self.count
            await asyncio.sleep(0.01)
            self.count = current + 1
        
        async def increment_good(self):
            # ✅ Protected with lock
            async with self.lock:
                current = self.count
                await asyncio.sleep(0.01)
                self.count = current + 1
    
    # Demonstrate the solutions
    print("Testing proper patterns...")
    
    result = await forgot_await_good()
    print(f"Await result: {result}")
    
    result = await blocking_operations_good()
    print(f"Non-blocking result: {result}")

async def small_operation():
    await asyncio.sleep(0.001)
    return "done"

asyncio.run(common_pitfalls_and_solutions())
```

## Conclusion

This guide covers the complete spectrum of asynchronous programming in Python, from basic concepts to advanced patterns. Key takeaways:

1. **Start Simple**: Begin with basic async/await patterns
2. **Understand the Event Loop**: It's the foundation of async programming
3. **Use Proper Error Handling**: Always consider exception scenarios
4. **Manage Resources**: Use async context managers and proper cleanup
5. **Control Concurrency**: Use semaphores and other synchronization primitives
6. **Profile and Optimize**: Measure performance and optimize bottlenecks
7. **Follow Best Practices**: Avoid common pitfalls and use established patterns

### Next Steps

1. Practice with real I/O operations (file handling, network requests)
2. Integrate with async-compatible libraries (aiohttp, asyncpg, motor)
3. Build complete applications using async frameworks (FastAPI, aiohttp)
4. Learn about async testing strategies
5. Explore advanced topics like custom event loop policies
6. Study async patterns in distributed systems

### Recommended Libraries

- **HTTP**: `aiohttp`, `httpx`
- **Database**: `asyncpg` (PostgreSQL), `aiomysql` (MySQL), `motor` (MongoDB)
- **Files**: `aiofiles`
- **Testing**: `pytest-asyncio`, `aioresponses`
- **Web Frameworks**: `FastAPI`, `aiohttp`, `Quart`
- **Message Queues**: `aio-pika` (RabbitMQ), `aiokafka` (Kafka)

Remember: Async programming shines with I/O-bound operations but isn't always the best choice for CPU-bound tasks. Choose the right tool for the job!