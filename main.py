def make_palindrome(s):
    return s + s[::-1]

def is_palindrome(s):
    return s == s[::-1]

def main():
    s = input("Enter a string: ")
    print(make_palindrome(s))

if __name__ == "__main__":
    main()