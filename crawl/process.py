import os
import json
import re
from callapi import segment
from format import createFinal

file_list = []
folder_path = 'raw_data/small'
for file_name in os.listdir(folder_path):
    # Check if the file name matches the format 'n.txt', where n is a number
    if file_name.endswith('.txt') and file_name[:-4].isdigit():
        file_list.append(file_name)

for file_name in file_list:
    print(file_name)
    file_path = os.path.join(folder_path, file_name)

    with open(file_path, 'r', encoding='utf-8') as f:
        article = json.loads(json.loads(f.read()))
    content = article['content']
    content = re.sub(r'[“”]', '"', content)  # Preprocessing
    print(f"Raw data: {content}\n")

    # Process ......
    segmented_content = segment(content, "gpt-3.5")

    if len(segmented_content) != 0:
        file_path = os.path.join('segmented_data/small', file_name)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(segmented_content)

        final_json = createFinal(segmented_content)
        file_path = os.path.join('final_data/small', file_name)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(final_json)

        file_path = os.path.join(folder_path, file_name)
        os.remove(file_path)  # delete raw data that is already processed
