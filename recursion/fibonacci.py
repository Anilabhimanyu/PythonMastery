# Approach 1: Simple Recursion
def fibonacci_recursion(n):
    # time complexity: O(2^n) why means exponential time, because each function call generates two more calls
    # space complexity: O(n)
    if n<2:
        return n
    return fibonacci_recursion(n-1) + fibonacci_recursion(n-2)

# Approach 2: Recursion with Memoization
def fibonacci_memoization(n, memo={}):
    # time complexity: O(n)
    # space complexity: O(n)
    if n in memo:
        return memo[n]
    if n<2:
        return n
    memo[n] = fibonacci_memoization(n-1, memo) + fibonacci_memoization(n-2, memo)
    return memo[n]

# Approach 3: Iterative Method
def fibonacci_iterative(n):
    # time complexity: O(n)
    # space complexity: O(1)
    if n<2:
        return n
    a, b = 0, 1
    for _ in range(2, n+1):
        a, b = b, a + b
    return b

# Approach 4: Dynamic Programming
def fibonacci_dynamic_programming(n):
    # time complexity: O(n)
    # space complexity: O(n)
    if n<2:
        return n
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[n]

