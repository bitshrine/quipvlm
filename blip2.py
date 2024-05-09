import argparse
import time
import numpy as np
import torch
import torch.nn as nn
# import quant

from gptq import GPTQ
from texttable import Texttable

# added for integration with QuIP
from datautils import get_loaders
from modelutils import * # includes DEV
from quant import Quantizer
from bal import Balance
from near import Nearest

from tqdm import tqdm

from dataclasses import dataclass, field
import typing
import inspect

@dataclass
class CatchableLayer():
    """
    A single quantizable layer
    """
    name: str
    layer: nn.Module

@dataclass
class CatchConfig():
    """
    Configuration object used to quantize
    a block of layers within a model.
    """
    layers: list[CatchableLayer] # Block of layers to quantize. Only the first layer will be instrumented
    input_size: tuple[int] # Size of the input to each of the layers in the block (same size for all layers)
    cache: dict[str, typing.Any] = field(default_factory=dict) # Dict indicating which kwargs to store when layer.forward is called
    siblings: list[CatchableLayer] = field(default_factory=list) # Layers that are not quantized (e.g. activation layers) but should still be moved to device

def transform_layer(model: nn.Module, name: str, transform: typing.Callable[[nn.Module], nn.Module] = lambda x: x) -> nn.Module:
    """
    Perform a transformation on a layer in the model.
    Catch-all method that can be used to replace a layer with another one,
    or be used to retrieve the layer by name by omitting the transform parameter.
    """
    parent_layer = model
    name_parts = name.split('.')
    for subname in name_parts[:-1]:
        parent_layer = parent_layer._modules[subname]
    parent_layer._modules[name_parts[-1]] = transform(parent_layer._modules[name_parts[-1]])
    return parent_layer._modules[name_parts[-1]]

def get_blip2(model):

    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import Blip2ForConditionalGeneration
    model = Blip2ForConditionalGeneration.from_pretrained(model, torch_dtype='auto')
    model.seqlen = 1024
    model.config.max_length = model.seqlen
    return model


@torch.no_grad()
def quant_sequential(full_model: nn.Module, model: nn.Module, catch_config: CatchConfig, dataloader,
                     wbits: int = 16, dev: torch.device = torch.device('cpu'), sequential: list[list[str]] = []):
    
    layers = list(map(lambda l: CatchableLayer(*l), catch_config.layers))
    cache: dict = catch_config.cache
    cache.update({'i': 0})

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, *catch_config.input_size), dtype=dtype, device=dev
    )

    class Catcher(nn.Module):
        def __init__(self, module: nn.Module):
            super().__init__()
            self.module = module
        def __getattr__(self, name):
            if name in ['module']:  # Exclude 'module' from interception
                return super().__getattr__(name)
            return getattr(self.module, name)
        def forward(self, *args, **kwargs):
            cachable_args = inspect.getcallargs(self.module.forward, *args, **kwargs)
            inps[cache['i']] = args[0]
            cache['i'] += 1
            for k in cache.keys() - 'i':
                if (k in cachable_args.keys()):
                    cache[k] = cachable_args[k]

            raise ValueError
        
    transform_layer(model, layers[0].name, lambda layer: Catcher(layer).to(dev))

    # Put sibling layers on device
    for sib in catch_config.siblings:
        transform_layer(model, sib.name, lambda layer: layer.to(dev))

    for batch_inp, _ in dataloader:
        try:
            # model(batch[0].to(dev))
            full_model(**(batch_inp.to(dev)))
        except ValueError:
            pass

    transform_layer(model, layers[0].name, lambda layer: layer.module.cpu())

    # Remove sibling layers from device
    for sib in catch_config.siblings:
        transform_layer(model, sib.name, lambda layer: layer.cpu())

    del cache['i']
    
    EMPTY_CACHE()

    # Prepare for quantization
    outs = torch.zeros_like(inps) if len(layers) > 1 else [None] * inps.shape[0]

    #print('Ready.')

    quantizers = {}
    pbar = tqdm(range(len(layers)))
    for i in pbar:
        pbar.set_description(f"Quantizing {layers[i].name}")
        layer = transform_layer(model, layers[i].name, lambda l: l.to(dev))
        subset = find_layers(layer)

        # Initialize Quant Method and Çompute H
        quant_method = {}
        for name in subset:
            if args.quant == 'gptq':
                quant_method[name] = GPTQ(subset[name])
                quant_method[name].quantizer = Quantizer()
                quant_method[name].quantizer.configure(wbits,
                                               perchannel=True,
                                               sym=False,
                                               qfn=args.qfn,
                                               mse=False)
            elif args.quant == 'nearest':
                quant_method[name] = Nearest(subset[name])
                quant_method[name].quantizer = Quantizer()
                quant_method[name].quantizer.configure(wbits,
                                               perchannel=True,
                                               sym=False,
                                               qfn=args.qfn,
                                               mse=False)
            elif args.quant in ['bitbal','parbal','allbal','allbal_block','allbal_clipevery','allbal_stochinit',
            'ldlq','ldlqRG','ldlqRG_block','ldlbal_admm']:
                quant_method[name] = Balance(subset[name])
                quant_method[name].configure(
                                    args.quant,
                                    wbits,
                                    args.npasses,
                                    unbiased=args.unbiased)
                quant_method[name].quantizer = Quantizer()
                quant_method[name].quantizer.configure(wbits,
                                               perchannel=True,
                                               sym=False,
                                               qfn=args.qfn,
                                               mse=False)
        def add_batch(name):

            def tmp(_, inp, out):
                quant_method[name].add_batch(inp[0].data, out.data)

            return tmp
        
        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), **cache)[0]
        for h in handles:
            h.remove()
        # added for QuIP integration
        for name in subset:
            quant_method[name].post_batch()

        for name in subset:
            quant_method[name].preproc(
                                preproc_gptqH=args.pre_gptqH, percdamp=args.percdamp,
                                preproc_rescale=args.pre_rescale, 
                                preproc_proj=args.pre_proj, preproc_proj_extra=args.pre_proj_extra)
            if args.quant == 'gptq':
                quant_method[name].fasterquant(groupsize=args.groupsize)
            elif args.quant in ['bitbal','parbal','allbal','allbal_block','allbal_clipevery','allbal_stochinit',
            'ldlq','ldlqRG','ldlqRG_block','ldlbal_admm']:
                quant_method[name].fasterquant()
            elif args.quant == 'nearest':
                quant_method[name].fasterquant()

            quantizers[f'{layers[i].name}{"." + name if len(name) > 0 else ""}'] = quant_method[name].quantizer
            quant_method[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), **cache)[0]
        
        transform_layer(model, layers[i].name, lambda l: l.cpu())
        del layer
        del quant_method 
        
        EMPTY_CACHE()

        inps, outs = outs, inps
  
    return quantizers


@torch.no_grad()
def blip2_sequential(model, dataloader, args, dev):

    #from transformers.models.clip.modeling_clip import CLIPEncoderLayer
    from transformers.models.blip_2.modeling_blip_2 import Blip2EncoderLayer, Blip2QFormerLayer
    from transformers.models.opt.modeling_opt import OPTDecoderLayer
    #from transformers.models.llama.modeling_llama import LlamaDecoderLayer
    print('Starting ...')

    quantizers = {}

    image_size = model.vision_model.config.image_size
    viz_num_patches = (image_size // model.vision_model.config.patch_size) ** 2
    viz_num_pos_embeds = viz_num_patches + 1

    # Quantize vision
    vision_bits = args.wbits[0]
    if (vision_bits < 16):
        print(f'Quantizing vision model to {vision_bits} bits')
        # The first layer is a Conv2D layer, should only be included
        # when not --exclude-conv
        vision_blocks = []
        if (not args.exclude_conv):
            vision_blocks.append(CatchConfig([('vision_model.embeddings.patch_embedding', model.vision_tower.vision_model.embeddings.patch_embedding)], (3, image_size, image_size)))
        
        # Last layer is part of a block of layers
        block_end_idx = None
        if (args.skip_last_vision):
            block_end_idx = -1
        vision_blocks.append(
            CatchConfig([*find_layers(model.vision_model, layers=[Blip2EncoderLayer]).items()][:block_end_idx],
                        (viz_num_pos_embeds, model.vision_model.config.hidden_size),
                        cache={'attention_mask': None})
        )
        for layer_block in vision_blocks:
            vision_quantizers = quant_sequential(model, model.vision_model, layer_block, dataloader, wbits=vision_bits, dev=dev)
            quantizers.update({f'vision_model.{n}': q for n, q in vision_quantizers.items()})

    # Quantize multimodal projection block in BLIP-2
    # This consists of a 'qformer' block followed by a language projection layer
    projector_bits = args.wbits[1]
    if (projector_bits < 16):
        print(f'Quantizing projector to {projector_bits} bits')
        projector_blocks = [
            #CatchConfig([('linear_1', model.multi_modal_projector.linear_1)],
            #            (1, viz_num_patches, model.vision_tower.config.hidden_size)),
            CatchConfig([*find_layers(model.qformer, layers=[Blip2QFormerLayer]).items()],
                        (model.config.num_query_tokens, model.qformer.config.hidden_size),
                        cache={'attention_mask': None, 'encoder_hidden_states': None, 'query_length': 0})
        ]
        for layer_block in projector_blocks:
            projector_quantizers = quant_sequential(model, model.qformer, layer_block, dataloader, wbits=projector_bits, dev=dev)
            quantizers.update({f'qformer.{n}': q for n, q in projector_quantizers.items()})
    
        # Skip language projection
        if (not args.skip_last_proj):
            lm_proj_block = CatchConfig([
                ('language_projection', model.language_projection)],
                (model.config.num_query_tokens, model.qformer.config.hidden_size)
            )
            language_proj_quantizer = quant_sequential(model, model, lm_proj_block, dataloader, wbits=projector_bits, dev=dev)
            quantizers.update({n: q for n, q in language_proj_quantizer.items()})

    # Quantize language model
    language_bits = args.wbits[2]
    if (language_bits < 16):
        print(f'Quantizing language model to {language_bits} bits')
        use_cache = model.language_model.config.use_cache
        model.language_model.config.use_cache = False
        language_blocks = [
            CatchConfig([*find_layers(model.language_model, layers=[OPTDecoderLayer]).items()],
                        (model.seqlen + model.config.num_query_tokens, model.language_model.config.hidden_size),
                        cache={'attention_mask': None})
        ]
        # Last layer is a block of its own
        if (not args.skip_last_language):
            language_blocks.append(
                CatchConfig([('lm_head', model.language_model.lm_head)],
                            (model.seqlen + model.config.num_query_tokens, model.language_model.config.hidden_size))
            )
        
        for layer_block in language_blocks:
            language_quantizers = quant_sequential(model, model.language_model, layer_block, dataloader, wbits=language_bits, dev=dev)
            quantizers.update({f'language_model.{n}': q for n, q in language_quantizers.items()})
        
        model.language_model.config.use_cache = use_cache

    return quantizers


@torch.no_grad()
def blip2_eval(model, testenc, dev):
    pass

def blip2_pack(model, quantizers, wbits, groupsize):
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    quant.make_quant_linear(model, quantizers, wbits, groupsize)
    qlayers = find_layers(model, [quant.QuantLinear])
    print('Packing ...')
    for name in qlayers:
        print(name)
        quantizers[name], scale, zero, g_idx, _, _ = quantizers[name]
        qlayers[name].pack(layers[name], scale, zero, g_idx)
    print('Done.')
    return model


# original includes fast inference
def load_quant_original(model, checkpoint, wbits, groupsize=-1, fused_mlp=True, eval=True, warmup_autotune=True):
    from transformers import Blip2Config, Blip2ForConditionalGeneration, modeling_utils
    config = Blip2Config.from_pretrained(model)

    def noop(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = noop
    torch.nn.init.uniform_ = noop
    torch.nn.init.normal_ = noop

    torch.set_default_dtype(torch.half)
    modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = Blip2ForConditionalGeneration(config)
    torch.set_default_dtype(torch.float)
    if eval:
        model = model.eval()
    layers = find_layers(model)
    for name in ['lm_head']:
        if name in layers:
            del layers[name]
    quant.make_quant_linear(model, layers, wbits, groupsize)

    del layers

    print('Loading model ...')
    if checkpoint.endswith('.safetensors'):
        from safetensors.torch import load_file as safe_load
        model.load_state_dict(safe_load(checkpoint))
    else:
        model.load_state_dict(torch.load(checkpoint))

    if eval:
        quant.make_quant_attn(model)
        quant.make_quant_norm(model)
        if fused_mlp:
            quant.make_fused_mlp(model)

    if warmup_autotune:
        quant.autotune_warmup_linear(model, transpose=not (eval))
        if eval and fused_mlp:
            quant.autotune_warmup_fused(model)
    #model.seqlen = 2048
    model.seqlen = 4096
    print('Done.')

    return model


# slow inference, just to get perplexity/accuracy
def load_quant(model, checkpoint, wbits, groupsize=-1, eval=True):
    from transformers import LlavaConfig, LlavaForConditionalGeneration, modeling_utils
    config = LlavaConfig.from_pretrained(model)

    def noop(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = noop
    torch.nn.init.uniform_ = noop
    torch.nn.init.normal_ = noop

    torch.set_default_dtype(torch.half)
    modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = LlavaForConditionalGeneration(config)
    torch.set_default_dtype(torch.float)
    if eval:
        model = model.eval()
    layers = find_layers(model)
    for name in ['lm_head']:
        if name in layers:
            del layers[name]

    del layers

    print('Loading model ...')
    if checkpoint.endswith('.safetensors'):
        from safetensors.torch import load_file as safe_load
        model.load_state_dict(safe_load(checkpoint))
    else:
        model.load_state_dict(torch.load(checkpoint), strict=False) # weird version issue with rotary_emb

    #model.seqlen = 2048
    model.seqlen = 4096
    print('Done.')

    return model


def blip2_multigpu(model, gpus, gpu_dist):
    pass


def benchmark(model, input_ids, check=False):
    input_ids = input_ids.to(model.gpus[0] if hasattr(model, 'gpus') else DEV)
    SYNCHRONIZE()

    cache = {'past': None}

    def clear_past(i):

        def tmp(layer, inp, out):
            if cache['past']:
                cache['past'][i] = None

        return tmp

    for i, layer in enumerate(model.model.layers):
        layer.register_forward_hook(clear_past(i))

    print('Benchmarking ...')

    if check:
        loss = nn.CrossEntropyLoss()
        tot = 0.

    def sync():
        if hasattr(model, 'gpus'):
            for gpu in model.gpus:
                torch.cuda.synchronize(gpu)
        else:
            SYNCHRONIZE()

    max_memory = 0
    with torch.no_grad():
        attention_mask = torch.ones((1, input_ids.numel()), device=DEV)
        times = []
        for i in range(input_ids.numel()):
            tick = time.time()
            out = model(input_ids[:, i:i + 1], past_key_values=cache['past'], attention_mask=attention_mask[:, :(i + 1)].reshape((1, -1)))
            sync()
            times.append(time.time() - tick)
            print(i, times[-1])
            if hasattr(model, 'gpus'):
                mem_allocated = sum(torch.cuda.memory_allocated(gpu) for gpu in model.gpus) / 1024 / 1024
            else:
                mem_allocated = torch.cuda.memory_allocated() / 1024 / 1024
            max_memory = max(max_memory, mem_allocated)
            if check and i != input_ids.numel() - 1:
                tot += loss(out.logits[0].to(DEV), input_ids[:, (i + 1)].to(DEV)).float()
            cache['past'] = list(out.past_key_values)
            del out
        sync()
        print('Median:', np.median(times))
        if check:
            print('PPL:', torch.exp(tot / (input_ids.numel() - 1)).item())
            print('max memory(MiB):', max_memory)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Added arguments for integration with QuIP
    parser.add_argument('--quant',
                        choices=['bitbal', 'parbal', 'allbal', 'allbal_block', 'allbal_clipevery', 'allbal_stochinit', 
                        'ldlq', 'ldlqRG', 'ldlqRG_block', 'ldlbal_admm', 'nearest', 'gptq', 'gptq_updown'],
                        default='nearest',
                        help='Which quantization method to use.')
    parser.add_argument('--pre_gptqH', action='store_true',help='preprocessing')
    parser.add_argument('--pre_rescale', action='store_true', help='preprocessing')
    parser.add_argument('--pre_proj', action='store_true', help='preprocessing')
    parser.add_argument( '--pre_proj_extra', type=int, default=0, choices=[0, 1, 2], help='Extra options to control pre_proj step.')
    parser.add_argument('--qfn', type=str, default='a', help='qfn: a is default, b is sym incoherent based')
    parser.add_argument('--npasses', type=int, default=1, help='number passes to repeat balance loop over 1-d.')

    parser.add_argument('model', type=str, help='llama model to load')
    parser.add_argument('dataset', type=str, choices=['wikitext2', 'ptb', 'c4', 'llava_instruct_150k'], help='Where to extract calibration data from.')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration data samples.')
    parser.add_argument('--percdamp', type=float, default=.01, help='Percent of the average Hessian diagonal to use for dampening.')
    parser.add_argument('--wbits', type=str, default="16", help="#bits to use for quantization; use 16 for evaluating base model. Specify the number of bits per block by passing 'vision_wbits,projector_wbits,language_wbits', ex. --wbits 16,4,16 to only quantize the projector to 4 bits")
    parser.add_argument('--groupsize', type=int, default=-1, help='Groupsize to use for quantization; default uses full row.')
    #parser.add_argument('--eval', action='store_true', help='evaluate quantized model.')
    parser.add_argument('--eval', type=str, default='', choices=['vqav2', 'seed1'], nargs='?', const=['vqav2', 'seed1'], help='Benchmark to evaluate the model on. If none, all benchmarks are run. Available choices: [vqav2, seed1]')
    #parser.add_argument('--test-generation', action='store_true', help='test generation.')
    parser.add_argument('--save', type=str, default='', help='Save quantized checkpoint under this name.')
    parser.add_argument('--load', type=str, default='', help='Load quantized model.')
    parser.add_argument('--benchmark', type=int, default=0, help='Number of tokens to use for benchmarking.')
    parser.add_argument('--check', action='store_true', help='Whether to compute perplexity during benchmarking for verification.')
    parser.add_argument('--new-eval', action='store_true', help='Whether to use the new PTB and C4 eval')
    parser.add_argument('--layers-dist', type=str, default='', help='Distribution of layers across GPUs. e.g. 2:1:1 for 2 layers on GPU 0, 1 layer on GPU 1, and 1 layer on GPU 2. Any remaining layers will be assigned to your last GPU.')
    
    parser.add_argument('--skip-last-language', action='store_true', help='Whether to skip the last layer of the language model during quantization')
    parser.add_argument('--skip-last-proj', action="store_true", help='Whether to skip the last layer of the projector model during quantization')
    parser.add_argument('--skip-last-vision', action="store_true", help='Whether to skip the last layer of the vision model during quantization')
    parser.add_argument('--exclude-conv', action="store_false", help='Whether to include Conv2D and Conv1D layers in quantization')

    parser.add_argument(
        '--incoh_processing',
        action='store_true',
        help='incoherence processing')
    parser.add_argument(
        '--unbiased',
        action='store_true',
        help='unbiased')

    args = parser.parse_args()

    # defaults to incoherence processing
    if args.incoh_processing:
        args.pre_gptqH   = True
        args.pre_rescale = True
        args.pre_proj    = True
        args.proj_extra  = 1
        args.qfn         = 'b'

    wbits = list(map(int, args.wbits.split(',')))
    if (len(wbits) not in [1, 3]):
        raise ValueError("Incorrect number of values supplied to --wbits. Please only provide 1 or 3 values.")
    if any(map(lambda v: v not in [2, 3, 4, 8, 16], wbits)):
        raise ValueError("Incorrect value for bit-width. Please select one of [2, 3, 4, 8, 16]")
    if (len(wbits) == 1):
        wbits = [wbits[0]] * 3
    args.wbits = wbits

    if (args.exclude_conv):
        DEFAULT_Q_LAYERS.remove(nn.Conv2d)

    if args.layers_dist:
        gpu_dist = [int(x) for x in args.layers_dist.split(':')]
    else:
        gpu_dist = []

    if type(args.load) is not str:
        args.load = args.load.as_posix()

    if args.load:
        model = load_quant(args.model, args.load, args.wbits, args.groupsize)
    else:
        model = get_blip2(args.model)
        model.half()
        model.eval()

    args.dataset = 'blip2-calibration'
    dataloader, testloader = get_loaders(args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen)

    if not args.load and any(map(lambda x: x < 16, args.wbits)):
        tick = time.time()
        quantizers = blip2_sequential(model, dataloader, args, DEV)
        print(time.time() - tick)

    # if args.benchmark:
    #     gpus = [torch.device('cuda:%d' % i) for i in range(torch.cuda.device_count())]
    #     if len(gpus) > 1:
    #         llama_multigpu(model, gpus, gpu_dist)
    #     else:
    #         model = model.to(DEV)
    #     if args.benchmark:
    #         input_ids = next(iter(dataloader))[0][:, :args.benchmark]
    #         benchmark(model, input_ids, check=args.check)

    #if args.eval:
    #    from multimodalutils import *
    #    eval_vqav2(args, model, 'v2_OpenEnded_mscoco_val2014_questions.json')

    if args.eval is not None:
        template = 'Question: {question}\nAnswer:'
        from multimodalutils import *
        for bench in args.eval:
            if (bench == 'vqav2'):
                eval_vqav2(args, model, 'v2_OpenEnded_mscoco_val2014_questions.json', template='Answer using a single word or phrase. Question: {question} Answer:')
            elif (bench == 'seed1'):
                eval_seed1(args, model, '', template='Question: {question} Answer:')
            
    
    #if args.test_generation:
    #    gpus = [torch.device('cuda:%d' % i) for i in range(torch.cuda.device_count())]
    #    if len(gpus) > 1:
    #        blip2_multigpu(model, gpus, gpu_dist)
    #    else:
    #        model = model.to(DEV)

    #    from transformers import LlamaTokenizer, TextStreamer
    #    tokenizer = LlamaTokenizer.from_pretrained(args.model, use_fast=False)
    #    input_ids = tokenizer(["The capital of New Mexico is"], return_tensors="pt").input_ids.to(gpus[0])
    #    streamer = TextStreamer(tokenizer)
    #    with torch.no_grad():
    #        generated_ids = model.generate(input_ids, streamer=streamer)

    # if args.quant_directory is not None:
    #     export_quant_table(quantizers, args.quant_directory)

    if args.save:
        # llama_pack(model, quantizers, args.wbits, args.groupsize)
        torch.save(model.state_dict(), args.save)

    # if args.save_safetensors:
    #     llama_pack(model, quantizers, args.wbits, args.groupsize)
    #     from safetensors.torch import save_file as safe_save
    #     state_dict = model.state_dict()
    #     state_dict = {k: v.clone().contiguous() for k, v in state_dict.items()}
    #     safe_save(state_dict, args.save_safetensors)
