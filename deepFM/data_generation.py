data_dir = '../data'
import os
import json
import pandas as pd
import itertools

def get_valid_documents(thresh=100):
    doc_read = {}
    path = "{}/read/".format(data_dir)
    for dirpath, subdirs, files in os.walk(path):
        for f in files:
            filename = dirpath+f
            file = open(filename, 'r')
            for line in file.readlines():
                words = line.strip().split(' ')
                user = words[0]
                for doc in words[1:]:
                    if doc not in doc_read:
                        doc_read[doc] = {}
                        doc_read[doc]['num']  = 1
                        doc_read[doc]['reader'] = [user]
                    else:
                        doc_read[doc]['num'] += 1
                        doc_read[doc]['reader'].append(user)

    doc_read_thresh = {key:{'num':doc_read[key]['num'], \
    'reader':doc_read[key]['reader']} for key in doc_read if doc_read[key]['num'] > thresh}
    """
    total doc: 505840
    doc over thresh=100: 36340
    """
    return doc_read_thresh

def get_user_list(valid_doc, thresh=100, etc_user_num=200):
    user_read_num = {}
    user_read_doc = {}
    for doc in valid_doc:
        readers = valid_doc[doc]['reader']
        for reader in readers:
            if reader not in user_read_doc:
                user_read_doc[reader] = [doc]
                user_read_num[reader] = 1
            else:
                user_read_doc[reader].append(doc)
                user_read_num[reader] += 1

    user_read_num1 = {key:user_read_num[key] for key in user_read_num if user_read_num[key] >= thresh}
    user_read_doc1 = {key:user_read_doc[key] for key in user_read_num1}

    user_read_num2 = {key:user_read_num[key] for key in user_read_num if user_read_num[key] < thresh}
    user_read_num2 = {key:user_read_num2[key] for i, key in enumerate(user_read_num2) if i < etc_user_num}
    user_read_doc2 = {key:user_read_doc[key] for key in user_read_num2}
    return user_read_doc1, user_read_num1, user_read_doc2, user_read_num2

def categorize_value(target, cat_num=100):
    new_target = {}
    target_list = list(target.values())
    max_ = max(target_list)
    min_ = min(target_list)
    division = int((max_ - min_ +1) / cat_num)
    for key in target:
        for i in range(cat_num):
            if target[key] >= (min_ + division*i) and target[key] < (min_+division*(i+1)):
                new_target[key] = i+1
        if target[key] >= min_ + division*cat_num:
            new_target[key] = cat_num
    return new_target

def doc_id2author_id(doc_id):
    return doc_id.split('_')[0]

def get_y_views(user_id, doc_id, user_read_doc):
    cnt = 0
    docs_read = user_read_doc[user_id]
    for doc in docs_read:
        if doc == doc_id:
            cnt += 1
    return cnt

def get_doc_meta_and_age(valid_doc, date_cat_num):
    """
    {document(id): { tags(keyword_list),age(unix timestamp) ,magazine_id,}}
    """
    data = open('../data/metadata.json', 'r')
    meta={}
    age = {}
    for line in data.readlines():
        line = json.loads(line)
        if line['id'] in valid_doc:
            tmp_dict={}
            tmp_age={}
            tmp_dict[line['id']]= {'keyword_list':line['keyword_list'],
            'mag_id':line['magazine_id']
            }
            tmp_age[line['id']] = line['reg_ts']
            meta.update(tmp_dict)
            age.update(tmp_age)
    age = categorize_value(age, date_cat_num)
    return meta, age

def get_following_list(user_read_num1, user_read_num2):
    user_list = {key:user_read_num1[key] for key in user_read_num1}
    user_list.update(user_read_num2)
    result = {}
    data = open('../data/users.json', 'r')
    for line in data.readlines():
        line = json.loads(line)
        if line['id'] in user_list:
            result[line['id']] = line['following_list']
    return result

def check_is_followed(user_id, author_id, following_data):
    if author_id in following_data[user_id]:
        return 1
    return 0

def prepare_data(param):
    print("preparing valid documents...")
    valid_doc = get_valid_documents(thresh=param['doc_thresh'])
    print("preparing user information...")
    user_read_doc1, user_read_num1, user_read_doc2, user_read_num2 \
    = get_user_list(valid_doc, etc_user_num=param['etc_user_num'])
    pop = {key:valid_doc[key]['num'] for key in valid_doc}
    print("preparing popularity...")
    popularity = categorize_value(pop, cat_num=param['pop_cat_num'])
    print("preparing meta and age...")
    meta, age = get_doc_meta_and_age(valid_doc, param['date_cat_num'])
    following = get_following_list(user_read_num1, user_read_num2)
    return valid_doc, user_read_doc1, user_read_num1, user_read_doc2, user_read_num2,\
    popularity, meta, age, following

def __index(dict):
    ind={}
    if dict==user_read_num2:
        ind.update({key:1000000 for key in dict.keys()}) ##1000000은 수정 가능
    else:
        for (index, entry) in enumerate(dict):
            ind.update({entry:index})
    return ind
#for tags & author - dataframe화 이후
##tag_a=["여행","자유"]/tag_b=["a","b"]/tag_c=["c","d"]꼴일때 - 각각의 tag가 input!
#
# def idx_for_col(x):
#     t = [doc.split(" ") for doc in x]
#     all_values = itertools.chain.from_iterable(t)
#     f_dict = {token: idx for idx, token in enumerate(set(all_values))}
#     size = len(f_dict) # the number of unique field value(feature) = p_i
#     id_vec=[]
#     for i in range(len(x)):
#         id_vec.append(f_dict[x[i]])
#     return id_vec
# tag_a_idx = idx_for_col(tag_a)
# tag_b_idx = idx_for_col(tag_b)
# tag_c_idx = idx_for_col(tag_c)
# author_idx = idx_for_col(author)

def get_index_data(valid_doc, user_read_num1, user_read_num2, meta):
    doc_indexed = __index(valid_doc)
    valid_user_idx = __index(user_read_num1)
    etc_user_idx = __index(user_read_num2)
    valid_user_idx.update(etc_user_idx)
    user_idx=valid_user_idx

    # author_idx
    authors = list(set([doc_id2author_id(doc_id) for doc_id in valid_doc]))
    author_indexed = {author:idx for idx, author in enumerate(authors)}

    # tags indx
    tags = []
    for doc_id in meta:
        tags += meta[doc_id]['keyword_list']
        tags = list(set(tags))
    tag_indexed = {tag:idx for idx, tag in enumerate(tags)}
    return doc_indexed, user_idx, author_indexed, tag_indexed

def data_to_index(doc_indexed, user_idx, author_indexed, tag_indexed, target_file='train'):
    writefile = open('../data/{}_data.txt'.format(target_file), 'w')
    writefile.write('')
    writefile = open('../data/{}_data.txt'.format(target_file), 'a')

    with open('../data/{}_data_raw.txt'.format(target_file), 'r') as f:
        for line in f.readlines():
            line = json.loads(line.strip())
            if line['user'] in user_idx:
                line['user'] = user_idx[line['user']]
            else:
                line['user'] = 1000000
            line['doc'] = doc_indexed[line['doc']]
            line['author'] = author_indexed[line['author']]
            line['tagA'] = tag_indexed[line['tagA']]
            line['tagB'] = tag_indexed[line['tagB']]
            line['tagC'] = tag_indexed[line['tagC']]
            writefile.write(json.dumps(line))
            writefile.write('\n')


def generate_data(user_id, doc_id, valid_doc, user_read_doc, popularity, \
meta, age, following, dataframe):
    """
    author: from doc_id2author_id <- doc_id
    tags : meta -> keyword_list
    magazine_id: meta -> magazine_id
    age: age  <- doc_id
    is_followed: check_is_followed <- following, user_id, author_id
    pop: popularity
    y: get_y_views <- user_id, doc_id, user_read_doc
    """
    author_id = doc_id2author_id(doc_id)
    dataframe['user'] = user_id
    dataframe['doc'] = doc_id
    dataframe['author'] = author_id
    dataframe['tagA'] = meta[doc_id]['keyword_list'][0]
    dataframe['tagB'] = meta[doc_id]['keyword_list'][1]
    dataframe['tagC'] = meta[doc_id]['keyword_list'][2]
    dataframe['magazine_id'] = meta[doc_id]['mag_id']
    dataframe['age'] = age[doc_id]
    dataframe['is_followed'] = check_is_followed(user_id, author_id, following)
    dataframe['popularity'] = popularity[doc_id]
    dataframe['y'] = get_y_views(user_id, doc_id, user_read_doc)
    return dataframe

def generate_train_data(valid_doc, user_read_doc1, user_read_doc2,\
popularity, meta, age, following, dataframe):
    """create data to train"""
    writefile = open('../data/train_data_raw.txt', 'w')
    writefile.write('')
    writefile = open('../data/train_data_raw.txt', 'a')
    for user_id in user_read_doc1:
        for doc_id in user_read_doc1[user_id]:
            d = generate_data(user_id, doc_id, valid_doc, user_read_doc1, \
            popularity, meta, age, following, dataframe)
            writefile.write(json.dumps(d))
            writefile.write('\n')
    for user_id in user_read_doc2:
        for doc_id in user_read_doc2[user_id]:
            d = generate_data(user_id, doc_id, valid_doc, user_read_doc2, \
            popularity, meta, age, following, dataframe)
            print(d)
            writefile.write(json.dumps(d))
            writefile.write('\n')
    return

def generate_test_data(valid_doc, user_read_doc1, user_read_doc2,\
popularity, meta, age, following, dataframe):
    test_file = open('../data/predict/dev.users', 'r')
    writefile = open('../data/test_data_raw.txt', 'w')
    writefile.write('')
    writefile = open('../data/test_data_raw.txt', 'a')
    for line in test_file.readlines():
        user_id = line.strip()
        for doc_id in valid_doc:
            d = generate_data(user_id, doc_id, valid_doc, user_read_doc1, \
            popularity, meta, age, following, dataframe)


if __name__ == "__main__":
    param = {
        'user_thresh': 100,
        'doc_thresh': 100,
        'pop_cat_num': 100,
        'date_cat_num': 100,
        'etc_user_num': 200
    }
    dataframe = {'user': '', 'doc':'', 'author': '', 'tagA': '', 'tagB': '', 'tagC':'',\
    'is_followed': 0, 'magazine_id': '', 'popularity': 0, 'age': 0, 'y': 0}
    valid_doc, user_read_doc1, user_read_num1, user_read_doc2, user_read_num2,\
    popularity, meta, age, following = prepare_data(param)

    #doc_indexed, user_idx, author_indexed, tag_indexed = \
    #get_index_data(valid_doc, user_read_num1, user_read_num2, meta)

    generate_train_data(valid_doc, user_read_doc1, user_read_doc2,\
    popularity, meta, age, following, dataframe)

    # data_to_index(doc_indexed, user_idx, author_indexed, tag_indexed, target_file='train')
    #
    # generate_test_data(valid_doc, user_read_doc1, user_read_doc2,\
    # popularity, meta, age, following, dataframe)
