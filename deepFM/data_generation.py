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
                new_target[key] = i
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
        line = json.loads(line.strip())
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
    if user_id in following_data:
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
    print("preparing user read num categories...")
    tmp = {key:user_read_num1[key] for key in user_read_num1}
    tmp.update(user_read_num2)
    user_read = categorize_value(tmp, cat_num=param['user_read_cat_num'])
    return valid_doc, user_read_doc1, user_read_num1, user_read_doc2, user_read_num2,\
    popularity, meta, age, following, user_read

def __index(dict, name=1):
    ind={}
    if name==2:
        ind.update({key:1000000 for key in dict.keys()}) ##1000000은 수정 가능
    else:
        for (index, entry) in enumerate(dict):
            ind.update({entry:index})
    return ind

def get_index_data(valid_doc, user_read_num1, user_read_num2, meta, dummy):
    doc_indexed = __index(valid_doc)
    valid_user_idx = __index(user_read_num1)
    etc_user_idx = __index(user_read_num2, name=2)
    valid_user_idx.update(etc_user_idx)
    user_idx=valid_user_idx

    # author_idx
    authors = list(set([doc_id2author_id(doc_id) for doc_id in valid_doc]))
    author_indexed = {author:idx for idx, author in enumerate(authors)}

    # tags indx
    tags = []
    mags = []
    for doc_id in meta:
        tags += meta[doc_id]['keyword_list']
        tags = list(set(tags))
        mags.append(meta[doc_id]['mag_id'])
        mags = list(set(mags))
    # add DUMMY_TAG and empty tag
    tags += ['', dummy['DUMMY_TAG']]
    mags.append(dummy['DUMMY_MAG_ID'])
    tag_indexed = {tag:idx for idx, tag in enumerate(tags)}
    mag_indexed = {mag:idx for idx, mag in enumerate(mags)}
    return doc_indexed, user_idx, author_indexed, tag_indexed, mag_indexed

def data_to_index(doc_indexed, user_idx, author_indexed, tag_indexed, mag_indexed, target_file='train'):
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
            line['magazine_id'] = mag_indexed[line['magazine_id']]
            writefile.write(json.dumps(line))
            writefile.write('\n')

def generate_data(user_id, doc_id, valid_doc, user_read_doc, user_read_num, popularity, \
meta, age, following, dataframe, dummy, state='train'):
    """
    author: from doc_id2author_id <- doc_id
    tags : meta -> keyword_list
    magazine_id: meta -> magazine_id
    age: age  <- doc_id
    is_followed: check_is_followed <- following, user_id, author_id
    pop: popularity
    y: get_y_views <- user_id, doc_id, user_read_doc

    is_followed, y, views changes for test data
    """
    author_id = doc_id2author_id(doc_id)
    dataframe['user'] = user_id
    dataframe['doc'] = doc_id
    dataframe['author'] = author_id
    # add dummy data
    tag_list = ['tagA', 'tagB', 'tagC']
    if doc_id in meta:
        for i, keyword in enumerate(meta[doc_id]['keyword_list']):
            if i >= 3: break
            dataframe[tag_list[i]] = keyword
        dataframe['magazine_id'] = meta[doc_id]['mag_id']
        dataframe['age'] = age[doc_id]
    else:
        dataframe['tagA'] = dummy['DUMMY_TAG']
        dataframe['tagB'] = dummy['DUMMY_TAG']
        dataframe['tagC'] = dummy['DUMMY_TAG']
        dataframe['magazine_id'] = dummy['DUMMY_MAG_ID']
        dataframe['age'] = dummy['DUMMY_DATE']
    dataframe['popularity'] = popularity[doc_id]
    if state == 'train':
        dataframe['y'] = get_y_views(user_id, doc_id, user_read_doc)
        dataframe['views'] = user_read_num[user_id]
        dataframe['is_followed'] = check_is_followed(user_id, author_id, following)
    else:
        dataframe['y'] = 0
        dataframe['views'] = 0
        dataframe['is_followed'] = 0
    return dataframe

def generate_train_data(valid_doc, user_read_doc1, user_read_doc2,\
user_read, popularity, meta, age, following, dataframe, dummy):
    """create data to train"""
    writefile = open('../data/train_data_raw.txt', 'w')
    writefile.write('')
    writefile = open('../data/train_data_raw.txt', 'a')
    for user_id in user_read_doc1:
        for doc_id in list(set(user_read_doc1[user_id])):
            d = generate_data(user_id, doc_id, valid_doc, user_read_doc1, \
            user_read, popularity, meta, age, following, dataframe, dummy, \
            state='train')
            writefile.write(json.dumps(d))
            writefile.write('\n')
    for user_id in user_read_doc2:
        for doc_id in list(set(user_read_doc2[user_id])):
            d = generate_data(user_id, doc_id, valid_doc, user_read_doc2, \
            user_read, popularity, meta, age, following, dataframe, dummy, \
            state='train')
            writefile.write(json.dumps(d))
            writefile.write('\n')
    return

def generate_test_data(valid_doc, user_read_doc1, user_read_doc2,\
user_read, popularity, meta, age, following, dataframe):
    test_file = open('../data/predict/dev.users', 'r')
    writefile = open('../data/test_data_raw.txt', 'w')
    writefile.write('')
    writefile = open('../data/test_data_raw.txt', 'a')
    tmp_read_doc = {key:user_read_doc1[key] for key in user_read_doc1}
    tmp_read_doc.update(user_read_doc2)
    for line in test_file.readlines():
        user_id = line.strip()
        for doc_id in valid_doc:
            if user_id in user_read_doc1:
                if doc_id in user_read_doc1[doc_id]: continue
            if user_id in user_read_doc2:
                if doc_id in user_read_doc2[doc_id]: continue
            d = generate_data(user_id, doc_id, valid_doc, tmp_read_doc, \
            user_read, popularity, meta, age, following, dataframe, \
            state='test')
            writefile.write(json.dumps(d))
            writefile.write('\n')
    return

def load_data(target='train', data_num=-1):
    df = pd.DataFrame()
    with open('../data/{}_data.txt'.format(target), 'r') as f:
        for i, line in enumerate(f.readlines()):
            if data_num != -1 and i % data_num == 0 and i != 0:
                print("Progress: {}".format(str(i)))
                return df
            line = json.loads(line.strip())
            dict = {i: line}
            tmp_df = pd.DataFrame(dict).transpose()
            df = pd.concat([df, tmp_df], axis=0)
    return df

def save_field_vocab_size(doc_indexed, user_idx, author_indexed, tag_indexed, \
mag_indexed, param, dataframe):
    # user, doc, author, tagA, tagB, tagC, is_followed, views, magazine_id,
    # popularity, age

    dataframe['user'] = len(user_idx)
    dataframe['doc'] = len(doc_indexed)
    dataframe['author'] = len(author_indexed)
    dataframe['tagA'] = len(tag_indexed)
    dataframe['tagB'] = len(tag_indexed)
    dataframe['tagC'] = len(tag_indexed)
    dataframe['magazine_id'] = len(mag_indexed)
    dataframe['is_followed'] = 2
    dataframe['views'] = param['user_read_cat_num']
    dataframe['popularity'] = param['pop_cat_num']
    dataframe['age'] = param['date_cat_num'] + 1
    with open('../data/field_vocab_size.txt', 'w') as f:
        f.write(json.dumps(dataframe))
    return

def get_field_vocab_size():
    with open('../data/field_vocab_size.txt', 'r') as f:
        data = json.loads(f.readlines()[0].strip())
    return data

def make_data(state='train'):
    param = {
        'user_thresh': 100,
        'doc_thresh': 100,
        'pop_cat_num': 100,
        'date_cat_num': 100,
        'user_read_cat_num': 20,
        'etc_user_num': 200
    }
    dummy = {
        'DUMMY_TAG' : 'JAEICK',
        'DUMMY_DATE' : param['date_cat_num']+1,
        'DUMMY_MAG_ID' : -1
    }
    dataframe = {'user': '', 'doc':'', 'author': '', 'tagA': '', 'tagB': '', 'tagC':'',\
    'is_followed': 0, 'views': 0, 'magazine_id': '', 'popularity': 0, 'age': 0, 'y': 0}
    valid_doc, user_read_doc1, user_read_num1, user_read_doc2, user_read_num2,\
    popularity, meta, age, following, user_read = prepare_data(param)

    print("Indexing...")
    doc_indexed, user_idx, author_indexed, tag_indexed, mag_indexed = \
    get_index_data(valid_doc, user_read_num1, user_read_num2, meta, dummy)

    if state == 'train':
        print("Generating train data...")
        generate_train_data(valid_doc, user_read_doc1, user_read_doc2,\
        user_read, popularity, meta, age, following, dataframe, dummy)

        print("Converting string to index...")
        data_to_index(doc_indexed, user_idx, author_indexed, tag_indexed, \
        mag_indexed, target_file=state)

        print("Saving field vocab size...")
        save_field_vocab_size(doc_indexed, user_idx, author_indexed, tag_indexed, \
        mag_indexed, param, dataframe)

    else:
        print("Generating test data...")
        generate_test_data(valid_doc, user_read_doc1, user_read_doc2,\
        user_read, popularity, meta, age, following, dataframe, dummy)

        print("Converting string to index...")
        data_to_index(doc_indexed, user_idx, author_indexed, tag_indexed, \
        mag_indexed, target_file=state)

if __name__ == "__main__":
    make_data('train')
