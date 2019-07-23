data_dir = '../data'
import os
import json
import pandas as pd

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
    tmp = {key:user_read_num1[key] for key in user_read_num1}
    user_list = tmp.update(user_read_num2)
    result = {}
    data = open('../data/users.json', 'r')
    for line in data.readlines():
        line = json.loads(line)
        if line['id'] in user_list:
            result[line['id']] = line['following_list']
    return result

def check_is_followed(user_id, author_id, following_data):
    return author_id in following_data[user_id]

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

def generate_data(user_id, doc_id, valid_doc, user_read_doc1, user_read_num1, \
user_read_doc2, user_read_num2, popularity, meta, age, following, data_frame):
    pass

def generate_train_data(valid_doc, user_read_doc1, user_read_num1, user_read_doc2,\
user_read_num2, popularity, meta, age, following, data_frame):
    """create data to train"""

    pass

def generate_test_data():
    pass

def data_to_index():
    pass

def data_to_dataframe():
    pass

if __name__ == "__main__":
    param = {
        'user_thresh': 100,
        'doc_thresh': 100,
        'pop_cat_num': 100,
        'date_cat_num': 100,
        'etc_user_num': 200
    }
    data_frame = {'user': '', 'doc':'', 'author': '', }
    valid_doc, user_read_doc1, user_read_num1, user_read_doc2, user_read_num2,\
    popularity, meta, age, following = prepare_data(param)
    generate_train_data(valid_doc, user_read_doc1, user_read_num1, user_read_doc2,\
    user_read_num2, popularity, meta, age, following, data_frame)
