# 242 Valid Anagram
# Easy
# Topics Hash Table, Sorting
# Given two strings s and t, return true if t is an anagram of s, and false otherwise.
# ex: s = "anagram", t = "nagaram" -> true
# ex: s = "rat", t = "car" -> false


class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        dict_s,dict_t={i:s.count(i) for i in s}, {i:t.count(i) for i in t}
        if dict_s == dict_t:
            return True
        else:
            return False
        