from os import path
import pandas as pd
from PIL import Image
import torch

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

def eval_vqav2(args, model, questions_file: str, template='USER:<image>\n{question} Answer with a single word or phrase.\nASSISTANT:'):
    from tqdm import tqdm
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    import json
    from vqav2.PythonEvaluationTools.vqaEvalDemo import eval
    processor = AutoProcessor.from_pretrained(args.model)

    generate_returns_prompt = type(model) in [LlavaForConditionalGeneration] # LLaVA returns the prompt with the answer, BLIP-2 only returns the answer

    base_path = 'vqav2/'


    #base_path = '/'.join(questions_path.split('/')[:-1])
    json_questions = json.loads(open(base_path + 'Questions/' + questions_file).read())
    version = 'v' + json_questions['info']['version'].split('.')[0]
    task_type = ''.join(json_questions['task_type'].split('-'))
    data_type = json_questions['data_type']
    data_subtype = json_questions['data_subtype']
    result_type = type(model).__name__ + '-v{}p{}l{}'.format(*args.wbits)

    img_dir = base_path + f'Images/{data_type}/{data_subtype}'
    # Store all answers as a .jsonl file first, for progressive evaluation (write every answer
    # as the model generates them, and on subsequent executions skip a question if the answer
    # is already present in the file)
    response_file = base_path + f'Results/{version}_{task_type}_{data_type}_{data_subtype}_{result_type}_unformatted_responses.jsonl'
    # vqav2 requires a .json file of a single array of json objects, which
    # we create from the unformatted responses once all the questions have been answered.
    results_file = base_path + f'Results/{version}_{task_type}_{data_type}_{data_subtype}_{result_type}_results.json'

    questions_df = pd.DataFrame(json_questions['questions'])
    questions_df['image_path'] = questions_df['image_id'] \
                                .map(lambda id: f"{img_dir}/COCO_{data_subtype}_{id:012d}.jpg")
    
    try:
        answered_questions = pd.read_json(response_file, lines=True)['question_id'].unique()
    except:
        answered_questions = pd.Series()

    remaining_questions = questions_df.loc[~questions_df['question_id'].isin(answered_questions)]

    with open(response_file, 'a+', 1) as f_out:
        for img_path, sub_df in tqdm(remaining_questions.groupby(by='image_path'), desc="Evaluating VQAV2..."):
            qimage = Image.open(img_path)

            for _, row in sub_df.iterrows():
                
                question = template.format(question=row['question'])
                inputs = processor(text=question, images=[qimage], return_tensors="pt").to(dtype=torch.float16)
                generate_ids = model.generate(**inputs, max_length=inputs.input_ids.shape[1] + 32)
                answer_start_idx = inputs.input_ids.shape[1] if generate_returns_prompt else None
                answer = processor.batch_decode(generate_ids[:, answer_start_idx:], skip_special_tokens=True, clean_up_tokenization_spaces = False)[0]

                res = {
                    "question_id": str(row['question_id']),
                    "answer": answer.strip(),
                }

                f_out.write(json.dumps(res) + '\n')

    print('All questions answered.')
    answer_df = pd.read_json(response_file, lines=True)
    answer_df.to_json(results_file, mode='w', orient='records')
    print(f'Formatted results written to {results_file}\n Launching evaluation...')
    
    eval(version + '_', task_type, data_type, data_subtype, result_type)


def eval_seed1(args, model, question_file: str = 'SEED-Bench.json', template: str = 'USER:<image>\n{question} Answer with a single word or phrase.\nASSISTANT: '):
    import json
    from tqdm import tqdm
    import numpy as np
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model)

    base_path = 'seed1/'
    question_file = 'SEED-Bench.json'

    seed_json = json.loads(open(base_path + question_file).read())
    full_df = pd.DataFrame(seed_json['questions'])
    # Only keep image tasks
    image_task_df = full_df.loc[full_df['question_type_id'] <= 9]
    image_task_df['image_path'] = image_task_df['data_id'].map(lambda x: base_path + 'SEED-Bench-image/' + x)

    response_file = base_path + type(model).__name__ + '-v{}p{}l{}'.format(*args.wbits) + '.jsonl'
    results_file = base_path + type(model).__name__ + '-v{}p{}l{}'.format(*args.wbits) + '_results.txt'

    try:
        answered_questions = pd.read_json(response_file, lines=True)['question_id'].unique()
    except:
        answered_questions = pd.Series()

    remaining_questions = image_task_df.loc[~image_task_df['question_id'].isin(answered_questions)]

    with open(response_file, 'a+', 1) as f_out:
        for img_path, sub_df in tqdm(remaining_questions.groupby(by='image_path'), desc="Evaluating SEED-1..."):
            qimage = Image.open(img_path)

            for _, row in sub_df.iterrows():
                
                question = template.format(question=row['question'])
                all_losses = []
                for choice in ['a', 'b', 'c', 'd']:
                    cand = row['choice_' + choice]
                    answer_input_ids = processor(text=cand, return_tensors="pt").to(dtype=torch.float16).input_ids.unsqueeze(0)

                    prompt = question + cand
                    inputs = processor(text=prompt, images=[qimage], return_tensors="pt").to(dtype=torch.float16)
                    
                    num_mask = answer_input_ids.shape[1]
                    labels = inputs.input_ids.clone()
                    labels[:, :-1 * (num_mask)] = -100
                    output = model(**inputs, labels=labels)
                    all_losses.append(output.loss.item())
                
                class_ranks = np.argsort(all_losses)
                pred_id = ['A','B','C','D'][class_ranks[0]]


                res = {
                    "question_id": str(row['question_id']),
                    "prediction": pred_id,
                }

                f_out.write(json.dumps(res) + '\n')

    responses_df = pd.read_json(response_file, lines=True)
    responses_df['question_id'] = responses_df['question_id'].astype(str)
    image_task_df['question_id'] = image_task_df['question_id'].astype(str)

    joined_df = responses_df.merge(image_task_df)

    acc = len(joined_df.loc[joined_df['prediction'] == joined_df['answer']]) / len(joined_df)
    
    results = f"""
    ======== SEED-1 RESULTS ========
    Total # of questions:   {len(image_task_df)}
    Accuracy:               {acc}
"""
    
    print(results)
    with open(results_file, 'w+') as out:
        out.write(results)
