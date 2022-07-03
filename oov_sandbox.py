import re
from statistics import mean

import pandas as pd
import numpy as np

dataset = "kaggle"  # toggle: "kaggle" (kaggle large) or "kaggle_toxic" (kaggle original)
sample_type = ["random", "topic", "wordbank"]  # remove "topic" if using original kaggle set
subset = ["all", "non", "abusive"]
i = 3  # number of sets per sample type
k = 5  # k-fold cross validation

# get list of words from lexicon
lexicon = open("lexicon.manual.all.CSV").read().splitlines()

# these for-loops will generate all the results you need
for sample in sample_type:  # sample type loop (e.g. the `random` in random1)
    for i in range(1, i + 1):  # set number loop (e.g. the `1` in random1)
        print(f"{dataset}.{sample}{i} Set =======")

        for sub in subset:  # subset loop (e.g. "all", "non-abusive", "abusive")
            list_of_oovs = []  # will hold OOV for each k. then the list will be averaged

            for k in range(1, k + 1):  # fold loop (k is the total number of folds)
                train = pd.read_csv(f"{dataset}/{sample}{i}/oov.{sample}{i}.fold{k}.train.CSV")
                test = pd.read_csv(f"{dataset}/{sample}{i}/oov.{sample}{i}.fold{k}.test.CSV")

                # filter data if necessary
                if sub == "non":
                    train = train[train["class"] == 0]  # get only non-abusive
                    test = test[test["class"] == 0]
                elif sub == "abusive":
                    train = train[train["class"] == 1]  # get only abusive
                    test = test[test["class"] == 1]

                # get used words in train
                regex = re.compile("[^A-Za-z0-9]+", re.IGNORECASE)
                train_used = []
                for comment in train["comment_text"].tolist():
                    for word in comment.split():
                        w_re = re.sub(regex, '', word)  # filter special characters
                        train_used.append(w_re.lower())

                # get used words in test
                test_used = []
                for comment in test["comment_text"].tolist():
                    for word in comment.split():
                        w_re = re.sub(regex, '', word)  # filter special characters
                        test_used.append(w_re.lower())

                # get words that appear in sets BUT NOT lexicon
                # train_unused = set(train_used) - set(lexicon)
                # test_unused = set(test_used) - set(lexicon)

                # get OOV
                oov_words = set(test_used) - set(train_used)  # used words that appear in test but not in train
                oov = (len(oov_words) / len(test_used)) * 100  # percentage of words in `test` that don't appear in train
                list_of_oovs.append(oov)

            # get average + print
            oov_avg = mean(list_of_oovs)
            str1 = f'Average OOV ({sub}): '
            print(f"{str1:<23}{oov_avg:>16}")

        print("")  # delineate

print(":)")  # celebrate
