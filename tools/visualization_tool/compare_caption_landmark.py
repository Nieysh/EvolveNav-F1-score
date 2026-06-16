import os

import spacy
import json
nlp = spacy.load("en_core_web_lg")

candidates = ["zsNo4HB9uLZ_11bdfcf0a0984901a1d6f60fdd8ee4a6_14", "zsNo4HB9uLZ_11bdfcf0a0984901a1d6f60fdd8ee4a6_16", "zsNo4HB9uLZ_11bdfcf0a0984901a1d6f60fdd8ee4a6_18", "zsNo4HB9uLZ_11bdfcf0a0984901a1d6f60fdd8ee4a6_22", "zsNo4HB9uLZ_11bdfcf0a0984901a1d6f60fdd8ee4a6_22", "zsNo4HB9uLZ_11bdfcf0a0984901a1d6f60fdd8ee4a6_22"]

blip2_caption_dir = "/data1/nieyunshuang/nys_new/zhaolx/NaviLLM/data/captions"
llava_vicuna_13b_4bit_caption_dir = "/data1/nieyunshuang/nys_new/llava2vln_data/llava-v1.6-vicuna-13b-4bit_captions"
with open('/data1/nieyunshuang/nys_new/llava2vln_data/vln_undealwith_data/singleview_tag2text.json', 'r') as f1:
    t2t_caption_data = json.load(f1)

def load_json(file):
    with open(file,'r') as f:
        data = json.load(f)
    return data

# def get_caption_for_each_cand(candidates):
#     # blip2_caption_all_cands = []
#     # llava_vicuna_13b_4bit_caption_all_cands = []
#     # t2t_caption_all_cands = []
#     # blip2_landmark_all_cands = []
#     # llava_vicuna_13b_4bit_landmark_all_cands = []
#     # t2t_landmark_all_cands = []
#     all_cands = []
#     for cand in candidates:                #zzz
#         per_cand = []
#         scan = cand.split('_')[0]
#         vpid = cand.split('_')[1]
#         img_idx = cand.split('_')[2]

#         blip2_caption_file = os.path.join(blip2_caption_dir,scan,vpid,f"{scan}_{vpid}.json")
#         blip2_caption = load_json(blip2_caption_file)[cand]
#         blip2_landmark = getlandmark(blip2_caption)
#         per_cand.append(blip2_caption)
#         per_cand.append(blip2_landmark)

#         llava_vicuna_13b_4bit_caption_file = os.path.join(llava_vicuna_13b_4bit_caption_dir,scan,vpid,f"{scan}_{vpid}.json")
#         llava_vicuna_13b_4bit_caption = load_json(llava_vicuna_13b_4bit_caption_file)[cand]
#         llava_vicuna_13b_4bit_landmark = getlandmark(llava_vicuna_13b_4bit_caption)
#         per_cand.append(llava_vicuna_13b_4bit_caption)
#         per_cand.append(llava_vicuna_13b_4bit_landmark)

#         t2t_caption = t2t_caption_data[cand]
#         t2t_landmark = getlandmark(t2t_caption)
#         per_cand.append(t2t_caption)
#         per_cand.append(t2t_landmark)
#         all_cands.append(per_cand)

#     for item in all_cands:
#         print(item)
    # print(f"blip caption:\n{blip2_caption_all_cands}\nblip landmark:\n{blip2_landmark_all_cands}\n\nllava-caption:\n{llava_vicuna_13b_4bit_caption_all_cands}\nllava-landmark:\n{llava_vicuna_13b_4bit_landmark_all_cands}\n\nt2t-caption:\n{t2t_caption_all_cands}\nt2t-landmark:\n{t2t_landmark_all_cands}")

def get_caption_for_each_cand(candidates):
    blip2_caption_all_cands = []
    llava_vicuna_13b_4bit_caption_all_cands = []
    t2t_caption_all_cands = []
    blip2_landmark_all_cands = []
    llava_vicuna_13b_4bit_landmark_all_cands = []
    t2t_landmark_all_cands = []
    all_cands = []
    
    for cand in candidates:                
        per_cand = []
        scan = cand.split('_')[0]
        vpid = cand.split('_')[1]
        img_idx = cand.split('_')[2]

        # Blip2 caption and landmark
        blip2_caption_file = os.path.join(blip2_caption_dir, scan, vpid, f"{scan}_{vpid}.json")
        blip2_caption = load_json(blip2_caption_file).get(cand, None)
        if blip2_caption is not None:
            blip2_landmark = getlandmark(blip2_caption)
            per_cand.append(blip2_caption)
            per_cand.append(blip2_landmark)
            blip2_caption_all_cands.append(blip2_caption)
            blip2_landmark_all_cands.append(blip2_landmark)

        # Llava caption and landmark
        llava_vicuna_13b_4bit_caption_file = os.path.join(llava_vicuna_13b_4bit_caption_dir, scan, vpid, f"{scan}_{vpid}.json")
        llava_vicuna_13b_4bit_caption = load_json(llava_vicuna_13b_4bit_caption_file).get(cand, None)
        if llava_vicuna_13b_4bit_caption is not None:
            llava_vicuna_13b_4bit_landmark = getlandmark(llava_vicuna_13b_4bit_caption)
            per_cand.append(llava_vicuna_13b_4bit_caption)
            per_cand.append(llava_vicuna_13b_4bit_landmark)
            llava_vicuna_13b_4bit_caption_all_cands.append(llava_vicuna_13b_4bit_caption)
            llava_vicuna_13b_4bit_landmark_all_cands.append(llava_vicuna_13b_4bit_landmark)

        # T2T caption and landmark
        t2t_caption = t2t_caption_data.get(cand, None)
        if t2t_caption is not None:
            t2t_landmark = getlandmark(t2t_caption)
            per_cand.append(t2t_caption)
            per_cand.append(t2t_landmark)
            t2t_caption_all_cands.append(t2t_caption)
            t2t_landmark_all_cands.append(t2t_landmark)
        
        # Collect all candidates' data
        if per_cand:
            all_cands.append(per_cand)

    # Finally print all results
    print(f"blip caption:\n{blip2_caption_all_cands}")
    print(f"blip landmark:\n{blip2_landmark_all_cands}")
    print(f"\nllava caption:\n{llava_vicuna_13b_4bit_caption_all_cands}")
    print(f"llava landmark:\n{llava_vicuna_13b_4bit_landmark_all_cands}")
    print(f"\nt2t caption:\n{t2t_caption_all_cands}")
    print(f"t2t landmark:\n{t2t_landmark_all_cands}")

def remove_unwanted_landmarks(landmarks):
    ### TODO: landmark需要统计分析一下, room/hallway等。另外，全部处理完以后可视化出来看看
    unwanted_landmarks = ["floor", "inside", "wall", "ceiling", "house"]
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

get_caption_for_each_cand(candidates)