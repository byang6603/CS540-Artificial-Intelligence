import string
import sys
import math
import unicodedata


def get_parameter_vectors():
    '''
    This function parses e.txt and s.txt to get the  26-dimensional multinomial
    parameter vector (characters probabilities of English and Spanish) as
    descibed in section 1.2 of the writeup

    Returns: tuple of vectors e and s
    '''
    #Implementing vectors e,s as lists (arrays) of length 26
    #with p[0] being the probability of 'A' and so on
    e=[0]*26
    s=[0]*26

    with open('e.txt',encoding='utf-8') as f:
        for line in f:
            #strip: removes the newline character
            #split: split the string on space character
            char,prob=line.strip().split(" ")
            #ord('E') gives the ASCII (integer) value of character 'E'
            #we then subtract it from 'A' to give array index
            #This way 'A' gets index 0 and 'Z' gets index 25.
            e[ord(char)-ord('A')]=float(prob)
    f.close()

    with open('s.txt',encoding='utf-8') as f:
        for line in f:
            char,prob=line.strip().split(" ")
            s[ord(char)-ord('A')]=float(prob)
    f.close()

    return (e,s)

def shred(filename):
    #Using a dictionary here. You may change this to any data structure of
    #your choice such as lists (X=[]) etc. for the assignment
    X=dict()
    with open (filename,encoding='utf-8') as f:
        text = f.read()
        text = ''.join(c for c in text if not any(unicodedata.category(d) == 'Mn' for d in unicodedata.normalize('NFD', c)))
        for letter in string.ascii_uppercase:
            X[letter] = 0

        for letter in text:
            if letter.isalpha():
                letter = letter.upper()
                X[letter] += 1
        print("Q1")
        for key in sorted(X):
            print(key, X.get(key))
    return X



# TODO: add your code here for the assignment
# You are free to implement it as you wish!
# Happy Coding!
def convert_to_list(letter_dict):
    vector = [0] * 26
    for letter, n in sorted(letter_dict.items()):
        index = ord(letter)-ord('A')
        vector[index] = n
    return vector

def multinomial_probability(vector, total_letters, lang):
    total_letters_fact = math.factorial(total_letters)
    outcomes = 1
    for i in range(len(vector)):
        outcomes *= math.factorial(vector[i])
    probabilities = 1
    for i in range(len(vector)):
        probabilities *= lang[i] ** vector[i]
    result = total_letters_fact / outcomes * probabilities
    return result

def find_xloglang(vector, letter_dict):
    e, s = get_parameter_vectors()

    xloge = vector[0] * math.log(e[0])
    xlogs = vector[0] * math.log(s[0])

    print("Q2")
    print(f"{xloge:.4f}")
    print(f"{xlogs:.4f}")

def find_f_lang(vector, letter_dict):
    e_prior = float(sys.argv[2])
    s_prior = float(sys.argv[3])

    e, s = get_parameter_vectors()

    log_f_e = math.log(e_prior) + sum(vector[i] * math.log(e[i]) for i in range(len(vector)))
    log_f_s = math.log(s_prior) + sum(vector[i] * math.log(s[i]) for i in range(len(vector)))


    print("Q3")
    print(f"{log_f_e:.4f}")
    print(f"{log_f_s:.4f}")

    fdiff = log_f_s - log_f_e

    if fdiff >= 100:
        prob_english = 0
    elif fdiff <= -100:
        prob_english = 1
    else:
        prob_english = 1/(1 + math.exp(fdiff))

    print("Q4")
    print(f"{prob_english:.4f}")

letter_dict = shred(sys.argv[1])
vector = convert_to_list(letter_dict)
find_xloglang(vector, letter_dict)
find_f_lang(vector, letter_dict)