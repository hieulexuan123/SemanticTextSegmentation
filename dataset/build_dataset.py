import os
import random
import json

folder_path = '../crawl/final_data/small'
train_size = 0.7
val_size = 0.15

file_names = os.listdir(folder_path)
total_size = len(file_names)
indices = list(range(total_size))
random.shuffle(indices)

train_set = [file_names[i] for i in indices[:int(train_size * total_size)]]
val_set = [file_names[i] for i in indices[int(train_size * total_size): int((train_size + val_size) * total_size)]]
test_set = [file_names[i] for i in indices[int((train_size + val_size) * total_size):]]

assert total_size == len(train_set) + len(val_set) + len(test_set)


def save_dataset(dataset, save_path):
    articles = []
    for file_name in dataset:
        with open(os.path.join(folder_path, file_name), 'r', encoding='utf-8') as f:
            article = json.loads(f.read())
            articles.append(article)
    with open(save_path, 'w', encoding='utf-8') as f:
         f.write(json.dumps(articles, ensure_ascii=False, indent=4))


save_dataset(train_set, './train.txt')
save_dataset(val_set, './val.txt')
save_dataset(test_set, './test.txt')
