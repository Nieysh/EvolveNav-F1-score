import os
import spacy
import json
from tqdm import tqdm
nlp = spacy.load("en_core_web_lg")

method = "t2t"
blip2_caption_dir = "/data1/nieyunshuang/nys_new/zhaolx/NaviLLM/data/captions"
llava_vicuna_13b_4bit_caption_dir = "/data1/nieyunshuang/nys_new/llava2vln_data/llava-v1.6-vicuna-13b-4bit_captions"
with open('/data1/nieyunshuang/nys_new/llava2vln_data/vln_undealwith_data/singleview_tag2text.json', 'r') as f1:
    t2t_caption_data = json.load(f1)

def load_json(file):
    with open(file,'r') as f:
        data = json.load(f)
    return data

def save_json(file, data):
    with open(file,'w') as f:
        json.dump(data, f)

def checkandmake(path):
    if not os.path.exists(path):
        os.makedirs(path)



def remove_unwanted_landmarks(landmarks):
    ### TODO: landmark需要统计分析一下, room/hallway等。另外，全部处理完以后可视化出来看看
    unwanted_landmarks = ["floor", "inside", "wall", "ceiling", "house", "apartment", "home", "that", "it", "this"]
    for i, item in enumerate(landmarks):
        for delete_landmark in unwanted_landmarks:
            if delete_landmark in item:
                del landmarks[i]
                break
    return landmarks

def getlandmark(caption):
    doc = nlp(caption)
    nouns = []

    for noun in doc.noun_chunks:
        nouns.append(noun.lemma_)

    nouns = remove_unwanted_landmarks(nouns)
    return nouns

res_dict = {}
if method == 't2t':
    landmark_save_dir = "/data1/nieyunshuang/nys_new/zhaolx/NaviLLM/data/t2t_landmarks"

    scans = os.listdir(landmark_save_dir)

    for scan in tqdm(scans):
        vpids = os.listdir(os.path.join(landmark_save_dir, scan))
        for vpid in vpids:
            savepath = os.path.join(landmark_save_dir, scan, vpid)
            savefile = savepath + f"/{scan}_{vpid}.json"
            data = load_json(savefile)
            keys = list(data.keys())
            for k in keys:
                data[k] = remove_unwanted_landmarks(data[k])
            save_json(savefile, data)

# get_caption_for_each_cand(candidates)