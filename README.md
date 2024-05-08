# GPTQ + QuIP for VLMs

Currently supports modular GPTQ for LLaVA and VQAV2 evaluation.
Code built on top of the [QuIP](https://github.com/Cornell-RelaxML/QuIP) repository.

**Requirements**: see [requirements.txt](requirements.txt)

## Quantization

### GPTQ:
```bash
python llava.py llava-hf/llava-1.5-7b-hf llava_instruct_150k --wbits 4 --nsamples 128 [--save quantized.safetensors] --quant gptq --pre_gptqH [--eval vqav2 seed1] [--skip_last_{vision,proj,language}]
```

### LDLQ (QuIP):
```bash
python llava.py llava-hf/llava-1.5-7b-hf llava_instruct_150k --wbits 4 --nsamples 128 [--save quantized.safetensors] --quant ldlq --incoh_processing [--eval vqav2 seed1] [--skip_last_{vision,proj,language}]
```
Quantization of `Conv2d` and `Conv1d` layers is not implemented by the authors for QuIP (see [original implementation](https://github.com/Cornell-RelaxML/QuIP/blob/ac92cfc7a22f6100009e2caf53bb72257d3f3184/bal.py#L24))

**Parameters:**
|Parameter|Description|
|--|--|
|`--wbits`|Number of bits for each block of the VLM, as `vision,proj,llm`. For example, `--wbits 2,4,8` quantizes the vision block to 2 bits, the projector to 4 bits, and the language model to 8 bits|
|`--skip_last_{language,proj,vision}`|if set, skips the last layer of the specified block|
|`--exclude-conv`|Whether to exclude `nn.Conv2D` layers in the quantization process (i.e. only quantize linear layers)|

**Notes:**
- `--skip_last_vision` skips `model.vision_tower.vision_model.encoder.layers.23.mlp.fc2`
- `--skip_last_proj` skips the `linear_2` layer of the projector
- `--skip_last_language` skips the `lm_head` layer

- `--incoh_processing` (necessary argument when running QuIP / LDLQ) is a "meta argument which sets the following flags `--pre_gptqH --pre_rescale --pre_proj --qfn b`" (from the original QuIP README)


## Evaluation

Pass the `--eval` flag to evaluate the model on a benchmark. Accepted values are `vqav2` and `seed1`.

### VQAv2
The evaluation requires the image dataset to be downloaded locally in a `vqav2/Images/` directory created in this repo. The `vqav2` directory should also contain the question file and annotation file, as downloaded from the [website](https://visualqa.org/download.html):
```
quip
├── README.md
├── vqav2/
    ├── Images/mscoco/{dataset, ex. 'val2014'}
    ├── Questions/{question file, ex. 'v2_OpenEnded_mscoco_val2014_questions.json'}
    ├── Annotations/{Annotation file for the questions}
```
The results of the evaluation are written to the `vqav2/Results` directory, first in a `.jsonl` file. This is so that if the evaluation gets interrupted for some reason, it can be resumed without recomputing all answers from the beginning. Once all questions have been answered, a valid `[...]_results.json` file is generated, and the accuracy evaluation is performed. Results are printed to standard output and saved in a `Results/[...]_accuracy.json` file.

### SEED-1
The benchmark requires the image dataset to be downloaded locally in `seed1/SEED-Bench-images`, and the question file to be placed alongside it in the `seed1` directory.
```
quip
├── README.md
├── seed1/
    ├── SEED-Bench-images
    ├── SEED-Bench.json
```
The results of the evaluation are written to a `.jsonl` file within the `seed1` directory. Once all questions have been answered, the script computes the accuracy of the model and writes the results in a `.txt` file.