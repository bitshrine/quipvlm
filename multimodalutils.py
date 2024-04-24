from os import path
import pandas as pd
from PIL import Image

"""
# Multimodal GPTQ
## Requirements:
- Be able to specify which model blocks to quantize, and the bit-width
- Choose whether the last layer is quantized (lm_head)
    -> lm_head shares weights with the embedding layer of the language model

## Issues:
The implementation of GPTQ assumes:
- we are quantizing a unimodal model
- all the layers we wish to quantize are continuous and of the same type
- the dataset can melted into a single sequence of tokens and be sampled at random


### Base GPTQ dataflow
A tensor is created with one row per input. This tensor
stores the input corresponding to each sample from the dataset.
The inputs are collected by instrumenting the model to capture
the input arguments to a layer during a forward pass (and halt the pass
once the capture has been performed). The other arguments to `layer.forward()`
such as `attention_mask` and `position_ids` are stored **once** at the first
layer of the block, and then reused for all samples and all layers of the block.

Note that the stored inputs are modified as the algorithm moves through the layers, since
they are passed through the layer to obtain outputs, which then become
inputs for the next layer -> once the algorithm has moved on, the base inputs
are no longer in memory.
"""

FILE_URLS = {
    'coco': 'http://images.cocodataset.org/zips/train2017.zip',
    'gqa': 'https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip',
    'textvqa': 'https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip'
}

def get_file(local_path: str, url: str) -> str:
    from tqdm import tqdm
    import requests
    from pathlib import Path

    if (not path.exists(local_path)):
        Path('/'.join(local_path.split('/')[:-1])).mkdir(parents=True, exist_ok=True)
        with open(local_path, 'wb+') as f:
            for data in tqdm(requests.get(url, stream=True).iter_content(chunk_size=2048), desc=f'Fetching {local_path} from remote'):
                f.write(data)

    return local_path

def get_llava_instruct_img(img: pd.Series) -> pd.Series:
    """
    Returns a Series of PIL.Image objects based on the
    image paths in the supplied arguments.
    If the images are not present locally, they are downloaded.
    This function batches the downloads to make the process faster.
    """
    from remotezip import RemoteZip

    to_download = img[img.map(lambda f: not path.exists(f))]

    splits = {
        'coco': to_download[to_download.str.contains('coco/')],
        'gqa': to_download[to_download.str.contains('gqa/')],
        'textvqa': to_download[to_download.str.contains('textvqa/')]
    }
    for name, series in splits.items():
        if (len(series) > 0):
            print(f'Downloading images from {name} dataset...')
            split_path = series.str.split('/')
            base_path = split_path.iloc[0][0]
            img_paths = split_path.map(lambda l: '/'.join(l[1:])).values
            with RemoteZip(FILE_URLS[name]) as rzip:
                rzip.extractall(path=base_path, members=img_paths)
            print("Done.")

    return img.map(lambda f: Image.open(f))

def get_llava_instruct_150k_data(nsamples: int = 128, seed: int = 0) -> pd.DataFrame:
    import numpy as np

    print("Loading instruction file...")
    data = pd.read_json(get_file('multimodal/llava/llava_v1_5_mix665k.json', 'https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/llava_v1_5_mix665k.json'))
    print('Done.')

    # Ensure we have enough data to sample from
    # even if we take fewer samples for the actual calibration
    # (this is just the number of samples to fetch)
    fetch_samples = max(nsamples, 128)

    data = data[data['image'].notna()]
    # Exclude ocr_vqa and vg samples for now (they require extra steps to download)
    data = data[~data['image'].str.contains('ocr_vqa/', na=False) & ~data['image'].str.contains('vg/', na=False)]

    # Convert text data
    convert_convo = lambda d: ('USER: ' if d['from'] == 'human' else 'ASSISTANT: ') + d['value'].replace('<image>', '') + ' ' 
    concat = data['conversations'].sample(n=fetch_samples, random_state = np.random.seed(seed)) \
                    .map(lambda l: ''.join(map(convert_convo, l))) \
                    .aggregate('sum')
    
    # Get a sample of the images
    samples = data.sample(n=fetch_samples, random_state=np.random.seed(seed))
    samples['image_data'] = get_llava_instruct_img(samples['image'])

    return concat, list(samples['image_data'])

def eval_vqav2(model, processor_identifier: str, questions_path: str, template='USER:<image>\n{question} Answer with a single word or phrase.\nASSISTANT:'):
    from tqdm import tqdm
    from transformers import AutoProcessor
    import json
    processor = AutoProcessor.from_pretrained(processor_identifier)

    base_path = '/'.join(questions_path.split('/')[:-1])
    json_questions = json.loads(open(questions_path).read())
    version = 'v' + json_questions['info']['version'].split('.')[0]
    task_type = ''.join(json_questions['task_type'].split('-'))
    data_type = json_questions['data_type']
    data_subtype = json_questions['data_subtype']

    img_dir = base_path + f'/{data_subtype}'
    response_file = base_path + f'/{version}_{task_type}_{data_type}_{data_subtype}_responses.jsonl'

    questions_df = pd.DataFrame(json_questions['questions'])
    questions_df['image_path'] = questions_df['image_id'] \
                                .map(lambda id: f"{img_dir}/COCO_{data_subtype}_{id:012d}.jpg")

    with open(response_file, 'w+', 1) as f_out:
        for img_path, sub_df in tqdm(questions_df.groupby(by='image_path'), desc="Evaluating VQAV2..."):
            qimage = Image.open(img_path)

            for _, row in sub_df.iterrows():
                
                question = template.format(question=row['question'])
                inputs = processor(text=question, images=[qimage])
                generate_ids = model.generate(**inputs, max_new_tokens=15)
                answer = processor.batch_decode(generate_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces = False)[0]

                res = {
                    "question_id": str(row['question_id']),
                    "answer": answer,
                }

                f_out.write(json.dumps(res) + '\n')

