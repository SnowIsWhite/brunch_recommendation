"""
train model,
get result
decide 100 docs to recommend
make final file
"""
import numpy as np
import os
import json

"""check if user reads doc again and again"""
def check_user_read_again():
    ratio_dict= {}
    with open('../data/train_data.txt', 'r') as f:
        for i, line in enumerate(f.readlines()):
            line = json.loads(line.strip())
            user = line['user']
            y = line['y']
            if user not in ratio_dict:
                ratio_dict[user] = {'total': 0, 'over_one': 0}
            ratio_dict[user]['total'] += y
            if y > 1:
                ratio_dict[user]['over_one'] += 1

    # ratio
    sum_ = 0
    for user in ratio_dict:
        ratio = ratio_dict[user]['over_one']/(ratio_dict[user]['total']*1.)
        ratio_dict[user]['ratio'] = ratio
        sum_ += ratio
    average = sum_/len(ratio_dict)
    user_over_avg = []
    for user in ratio_dict:
        if ratio_dict[user]['ratio'] > average:
            user_over_avg.append(user)
    return user_over_avg

def convert_index_to_doc(index_arr):
    jfile = open('../data/index2doc.json', 'r')
    index2doc = json.loads(jfile.readlines()[0])
    docs = [index2doc[idx] for idx in index_arr]
    return docs

def get_users_out_from_result(result_users, user_over_avg):
    return [user for user in result_users if user in user_over_avg]

def get_users_like_doc(users):
    user_doc_info = {}
    # {user: doc[], views[], age[]}
    with open('../data/train_data.txt', 'r') as f:
        for i, line in enumerate(f.readlines()):
            line = json.loads(line.strip())
            if line['user'] in users and line['views'] > 1:
                if line['user'] not in user_doc_info:
                    user_doc_info[line['user']] = {'doc':[line['doc']], 'views': [line['views']], 'age':[line['age']]}
                else:
                    user_doc_info[line['user']]['doc'].append(line['doc'])
                    user_doc_info[line['user']]['views'].append(line['views'])
                    user_doc_info[line['user']]['age'].append(line['age'])
    return user_doc_info

def get_docs_by_score(user_doc_info):
    user_doc = {}
    for user in user_doc_info:
        score = [val + user_doc_info[user]['age'][i] for i, val in enumerate(user_doc_info[user]['views'])]
        tuples = list(zip(user_doc_info[user]['doc'], score))
        res = sorted(tuples, key=lambda x: x[1], reversed=True)
        res = [tup[0] for tup in res]
        user_doc[user] = res
    return user_doc

def final_recommendation(result, ratio=20):
    result_users = result.keys()
    user_over_avg = check_user_read_again()
    target_users = get_users_out_from_result(result_users, user_over_avg)
    user_doc_info = get_users_like_doc(target_users)
    user_doc = get_docs_by_score(user_doc_info)

    final = {}
    for user in result:
        if not user in user_doc_info:
            final[user] = result[user]
        else:
            final[user] = result[user][:80] + user_doc[user][:20]

    for user in final:
        final[user] = convert_index_to_doc(final[user])
    return final

def write_recommendation_file(user_order, final):
    with open('./result/recommendation.txt', 'w') as f:
        for i, user in enumerate(user_order):
            f.write(str(i+1))
            for doc in final[user]:
                f.write(doc)
                f.write(' ')
            f.write('\n')
