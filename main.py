import string
import numpy as np


def create_matrix(sentences, key_words):
    unique_words = sorted(set(word for sentence in sentences for word in sentence.split()))    
    result_matrix = []
    
    for sentence in sentences:
        words = set(sentence.split())
        row = [1 if word in words else 0 for word in unique_words]
        result_matrix.append(row)
    
    key_words_list = [1 if word in unique_words and word in key_words else 0 for word in unique_words]

    return result_matrix, key_words_list

def matrix_op(C, dims, q):
    C = C.T
    U, s, Vt = np.linalg.svd(C, full_matrices=False)

    k = dims
    sr = np.copy(s)
    sr[k:] = 0
    
    Sr = np.diag(sr)
    Ck = Sr.dot(Vt)

    sk = np.take(s, range(k), axis=0)
    Sk = np.diag(sk)

    Vk = np.take(Vt, range(k), axis=0)
    Ck = Sk.dot(Vk)

    q = q.T
    Sk_1 = np.linalg.inv(Sk)
    UkT = np.take(U.T, range(k), axis=0)

    qk = Sk_1.dot(UkT).dot(q)

    return Ck, qk

def cosinus(ck, qk):
    qk_mian = np.sqrt(np.sum(qk**2))
    ck_mian = [np.sqrt(np.sum(x**2)) for x in ck.T]
    mian = [ck_mian[i]*qk_mian for i in range(len(ck_mian))]

    licznik = []
    for column in ck.T:
        result = (column[0] * qk[0]) + (column[1] * qk[1])
        licznik.append(result) 


    wynik = [round(licznik[i]/mian[i], 2) for i in range(len(mian))]

    return wynik


def zadanie():
    docs_liczba = int(input())

    docsy = []

    for i in range(docs_liczba):
        docsy.append(input().translate(str.maketrans('', '', string.punctuation)).strip().lower())

    words = input().split(" ")

    k = int(input())

    slowa_dict, q = create_matrix(docsy, words)

    Ck, qk = matrix_op(np.array(slowa_dict), k, np.array(q))

    print(cosinus(Ck, qk))

zadanie()