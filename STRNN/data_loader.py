import numpy as np
from datetime import datetime
import pandas as pd


def load_data(source_file):
    f = open(source_file, 'r')
    lines = f.readlines()

    user2id, poi2id = {}, {}
    train_user, train_time, train_lat, train_lon, train_loc = [], [], [], [], []
    valid_user, valid_time, valid_lat, valid_lon, valid_loc = [], [], [], [], []
    test_user, test_time, test_lat, test_lon, test_loc = [], [], [], [], []
    user_time, user_lat, user_lon, user_loc = [], [], [], []
    attack_threshold = 30

    # The next line is to obtain the user id.
    prev_user = int(lines[0].split('\t')[0])
    attack_cnt = 0

    for i, line in enumerate(lines):
        # The next line is to convert the splited line into a list.
        tokens = line.strip().split('\t')
        # The next line obtains the original user id of the current line.
        user = int(tokens[0])
        if user == prev_user:
            attack_cnt += 1
        # This branch is effective when the line represents the next user.
        else:
            # This is to create a map from original id to new id.
            # Only considers users having more records than the threshold
            if attack_cnt >= attack_threshold:
                user2id[prev_user] = len(user2id)
            # This is to re-initiate the prev_user and attack_cnt
            prev_user = user
            attack_cnt = 1

    prev_user = int(lines[0].split('\t')[0])
    for i, line in enumerate(lines):
        tokens = line.strip().split('\t')
        user = user2id.get(int(tokens[0]))
        # The next line is to get rid of the users
        # who have less than 30 records
        if user is None:
            continue

        # Now, we will only deal with users with more than 30 records.
        time = (datetime.strptime(tokens[1], "%Y-%m-%d") - datetime(1970, 1,
                                                                    1)).days
        lat, lon, location = float(tokens[2]), float(tokens[3]), float(tokens[4])

        # The next line creates the poi2id dictionary.
        # It maps the existing location id to a new id.
        # The new id is defined upon the order of the appearance.
        # You can view it as simply rename location id.
        if poi2id.get(location) is None:
            poi2id[location] = len(poi2id)
        loc = poi2id.get(location)

        # When the user is the previous one,
        # Just add his attributes into lists accordingly.
        # Note that our file is ordered by user id.
        if user == prev_user:
            user_time.insert(0, time)
            user_lat.insert(0, lat)
            user_lon.insert(0, lon)
            user_loc.insert(0, loc)
        # We will update train / valid / test lists nnce new user appears.
        # To illustrate, each element in train_time is a list containing
        # the first 70% time record of a user. Others are similarly defined.
        # Note that (i == len(lines) - 1) is the corner case for the last one
        # Or it won't be included in the train/test/valid lists.
        if (user != prev_user) or (i == len(lines) - 1):
            train_threshold = int(len(user_time) * 0.7)
            valid_threshold = int(len(user_time) * 0.8)

            train_user.append(user)
            train_time.append(user_time[:train_threshold])
            train_lat.append(user_lat[:train_threshold])
            train_lon.append(user_lon[:train_threshold])
            train_loc.append(user_loc[:train_threshold])

            valid_user.append(user)
            valid_time.append(user_time[train_threshold:valid_threshold])
            valid_lat.append(user_lat[train_threshold:valid_threshold])
            valid_lon.append(user_lon[train_threshold:valid_threshold])
            valid_loc.append(user_loc[train_threshold:valid_threshold])

            test_user.append(user)
            test_time.append(user_time[valid_threshold:])
            test_lat.append(user_lat[valid_threshold:])
            test_lon.append(user_lon[valid_threshold:])
            test_loc.append(user_loc[valid_threshold:])

            prev_user = user
            user_time = [time]
            user_lat = [lat]
            user_lon = [lon]
            user_loc = [loc]

    f.close()

    return len(user2id), poi2id, \
           train_user, train_time, train_lat, train_lon, train_loc, \
           valid_user, valid_time, valid_lat, valid_lon, valid_loc, \
           test_user, test_time, test_lat, test_lon, test_loc


def treat_prepro(train, step):
    train_f = open(train, 'r')
    # Need to change depending on threshold
    if step==1:
        lines = train_f.readlines()#[:86445] #659 #[:309931]
    elif step==2:
        lines = train_f.readlines()#[:13505]#[:309931]
    elif step==3:
        lines = train_f.readlines()#[:30622]#[:309931]

    train_user = []
    train_td = []
    train_ld = []
    train_loc = []
    train_dst = []

    user = 1
    user_td = []
    user_ld = []
    user_loc = []
    user_dst = []

    for i, line in enumerate(lines):
        tokens = line.strip().split('\t')
        if len(tokens) < 3:
            if user_td: 
                train_user.append(user)
                train_td.append(user_td)
                train_ld.append(user_ld)
                train_loc.append(user_loc)
                train_dst.append(user_dst)
            user = int(tokens[0])
            user_td = []
            user_ld = []
            user_loc = []
            user_dst = []
            continue
        td = np.array([float(t) for t in tokens[0].split(',')])
        ld = np.array([float(t) for t in tokens[1].split(',')])
        loc = np.array([int(t) for t in tokens[2].split(',')])
        dst = int(tokens[3])
        user_td.append(td)
        user_ld.append(ld)
        user_loc.append(loc)
        user_dst.append(dst)

    if user_td: 
        train_user.append(user)
        train_td.append(user_td)
        train_ld.append(user_ld)
        train_loc.append(user_loc)
        train_dst.append(user_dst)

    return train_user, train_td, train_ld, train_loc, train_dst


def inner_iter(data, batch_size):
    data_size = len(data)
    num_batches = int(len(data)/batch_size)
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield data[start_index:end_index]
