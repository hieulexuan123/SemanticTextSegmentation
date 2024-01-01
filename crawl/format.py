import json


def createFinal(segmented_content):
    sections = json.loads(segmented_content)  # convert json to dict

    sentences = []
    labels = []

    for section in sections:
        section_sentences = section['sentences']
        if len(section_sentences) != 0:
            sentences.append(section_sentences[0])
            labels.append('b')
            for i in range(1, len(section_sentences)):
                sentences.append(section_sentences[i])
                labels.append('i')

    data = {'sentences': sentences, 'labels': labels}
    final_json = json.dumps(data, ensure_ascii=False, indent=4)
    return final_json
