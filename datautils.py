import numpy as np
import torch


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

def get_blip2_calibration(nsamples, seed, seqlen, model):
    from multimodalutils import get_llava_instruct_150k_data

    # Following the same idea as text-only datasets,
    # concatenate all conversations as `USER: <prompt> ASSISTANT: <answer>`
    # strings and then randomly select max-length token sequences
    # to use for the forward passes.
    # Each batch receives one random image from the dataset. Due to this,
    # at least one token should be replaced with 32000, the image token_id

    text_data, image_data = get_llava_instruct_150k_data(nsamples=nsamples, seed=seed)
    from transformers import AutoProcessor, BatchFeature
    processor = AutoProcessor.from_pretrained(model)

    trainenc = processor(text=text_data, images=image_data, return_tensors="pt")

    import random
    random.seed(seed)
    trainloader = []
    for img_id in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen

        input_ids = trainenc.input_ids[:, i:j]
        attention_mask = trainenc.attention_mask[:, i:j]
        pixel_values = trainenc.pixel_values[img_id].unsqueeze(0)
        #input_ids[:, 0] = 32000

        inp = BatchFeature(data={"input_ids": input_ids,
                                 "attention_mask":attention_mask,
                                 "pixel_values": pixel_values})
        tar = inp.input_ids.clone()
        tar[:-1] = -100
        trainloader.append((inp, tar))
    
    return trainloader, None

def get_llava_instruct_150k(nsamples, seed, seqlen, model):
    from multimodalutils import get_llava_instruct_150k_data

    # Following the same idea as text-only datasets,
    # concatenate all conversations as `USER: <prompt> ASSISTANT: <answer>`
    # strings and then randomly select max-length token sequences
    # to use for the forward passes.
    # Each batch receives one random image from the dataset. Due to this,
    # at least one token should be replaced with 32000, the image token_id

    text_data, image_data = get_llava_instruct_150k_data(nsamples=nsamples, seed=seed)
    from transformers import AutoProcessor, BatchFeature
    processor = AutoProcessor.from_pretrained(model)

    trainenc = processor(text=text_data, images=image_data)

    import random
    random.seed(seed)
    trainloader = []
    for img_id in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen

        input_ids = trainenc.input_ids[:, i:j]
        input_ids[:, 0] = 32000

        inp = BatchFeature(data={"input_ids": input_ids,
                                 "attention_mask":trainenc.attention_mask[:, i:j],
                                 "pixel_values": trainenc.pixel_values[img_id].unsqueeze(0)})
        tar = inp.input_ids.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    
    return trainloader, None

def get_wikitext2(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_ptb(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    valdata = load_dataset('ptb_text_only',
                           'penn_treebank',
                           split='validation')

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer("\n\n".join(traindata['sentence']),
                         return_tensors='pt')
    testenc = tokenizer("\n\n".join(valdata['sentence']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_c4(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset(
        'allenai/c4',
        #'allenai--c4',
        data_files={'train': 'en/c4-train.00000-of-01024.json.gz'},
        split='train')
    valdata = load_dataset(
        'allenai/c4',
        #'allenai--c4',
        data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'},
        split='validation')

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            #if trainenc.input_ids.shape[1] >= seqlen:
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    import random
    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
            if tmp.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)

    class TokenizerWrapper:

        def __init__(self, input_ids):
            self.input_ids = input_ids

    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc


def get_ptb_new(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_c4_new(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset(
        'allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    valdata = load_dataset(
        'allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    )

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc




def get_loaders(name, nsamples=128, seed=0, seqlen=2048, model=''):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, model)
    if 'ptb' in name:
        if 'ptb-new' in name:
                return get_ptb_new(nsamples, seed, seqlen, model)
        return get_ptb(nsamples, seed, seqlen, model)
    if 'c4' in name:
        if 'c4-new' in name:
            return get_c4_new(nsamples, seed, seqlen, model)
        return get_c4(nsamples, seed, seqlen, model)
    if 'seed' in name:
        return get_seed_data(nsamples, seed, seqlen, model)
    if 'vqa' in name:
        return get_vqa_data(nsamples, seed, seqlen, model)
    if 'gqa' in name:
        return get_gqa_data(nsmaples, seed, seqlen, model)
    if 'llava_instruct_150k' in name:
        return get_llava_instruct_150k(nsamples, seed, seqlen, model)
    if 'blip2-calibration' in name:
        return get_blip2_calibration(nsamples, seed, seqlen, model)