import numpy as np
from multiprocessing import Process, Queue


def sample_property(User, user_train_set, user_history, Item, property_dict, usernum, itemnum, subreddits, batch_size, result_queue, SEED,
                    is_test=False, User_test=[]):
    num_sub = len(subreddits)

    def sample_ui():
        user = np.random.randint(0, usernum)
        num_item_train = len(User[user])

        if not is_test:
            item_is = user_history[user]
            item_j = np.random.randint(0,num_item_train)
            item_j = User[user][item_j]
        else:
            num_item_test = len(User_test[user])
            item_is = user_history[user]
            item_j = np.random.randint(0, num_item_test)
            item_j = User_test[user][item_j]
        # the subreddit item j belongs to
        uj_s = Item[item_j]['sub']

        # the subreddit item j doesn't belongs to
        # or the subreddit item jp belongs to

        s = user_train_set[user]
        item_jp = np.random.randint(0,itemnum)
        # item_jp in User_test[user]?
        while item_jp in s or item_jp == item_j or item_jp in item_is: item_jp = np.random.randint(0, itemnum)

        uj_sp = Item[item_jp]['sub']

        item_is = item_is[-15:]

        # add property
        user_property = []
        for pd in property_dict:
            u_property = pd[user]
            user_property.append(u_property)

        return  user,item_is,uj_s,uj_sp,item_j,item_jp,user_property

    def sample_ii():
        while True:
            i = np.random.randint(0, itemnum)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            batch1 = sample_ui()
            one_batch.append(batch1)
        result_queue.put(zip(*one_batch))

class Sampler(object):

    def __init__(self,User, user_train_set, user_history, Item, property_dict, usernum, itemnum, subreddits, batch_size=10000, n_workers=2,
                 is_test=False, User_test=[]):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_property,args=(User,
                                                     user_train_set,
                                                     user_history,
                                                     Item,
                                                     property_dict,
                                                     usernum,
                                                     itemnum,
                                                     subreddits,
                                                     batch_size,
                                                     self.result_queue,
                                                     np.random.randint(2e9),
                                                     is_test,
                                                     User_test)))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


def sample_topk(user_train_set, user_history, Item, property_dict, usernum, itemnum, subreddits, batch_size, candidates_num, result_queue, SEED,
                User_test):
    num_sub = len(subreddits)

    def sample_ui():
        user = np.random.randint(0, usernum)
        num_item_test = len(User_test[user])
        item_is = user_history[user]
        item_j = np.random.randint(0, num_item_test)
        item_j = User_test[user][item_j]

        uj_s = Item[item_j]['sub']

        item_jps = []
        s = user_train_set[user]
        for iter in range(candidates_num):
            while True:
                uj_sp = np.random.randint(0, num_sub)
                if not uj_sp == uj_s: break
            item_jp = np.random.randint(0, itemnum)

            while item_jp in s or item_jp == item_j or item_jp in item_is: item_jp = np.random.randint(0, itemnum)
            item_jps.append(item_jp)

        item_is = item_is[-15:]

        # add property
        user_property = []
        for pd in property_dict:
            u_property = pd[user]
            user_property.append(u_property)

        return user, item_is, uj_s, uj_sp, item_j, item_jps, user_property

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            batch1 = sample_ui()
            # batch2 = sample_ii()

            # one_batch.append(batch1 + batch2)
            one_batch.append(batch1)
        result_queue.put(zip(*one_batch))

class Sampler_topk(object):

    def __init__(self,user_train_set, User_test, user_history, Item, property_dict, usernum, itemnum, subreddits, candidates_num, batch_size=1, n_workers=2):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_topk(),args=(user_train_set,
                                                   user_history,
                                                   Item,
                                                   property_dict,
                                                   usernum,
                                                   itemnum,
                                                   subreddits,
                                                   batch_size,
                                                   candidates_num,
                                                   self.result_queue,
                                                   np.random.randint(2e9),
                                                   User_test)))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()
