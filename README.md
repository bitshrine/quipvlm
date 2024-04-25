# GPTQ + QuIP for VLMs

Currently supports modular GPTQ for LLaVA and VQAV2 evaluation.
Code built on top of the [QuIP](https://github.com/Cornell-RelaxML/QuIP) repository.

**Requirements**: see [requirements.txt](requirements.txt)

## Quantization

### GPTQ:
```bash
python llava.py llava-hf/llava-1.5-7b-hf llava_instruct_150k --wbits 4 --nsamples 128 [--save quantized.safetensors] --quant gptq --pre_gptqH [--eval] [--skip_last_{vision,proj,language}]
```

**Parameters:**
|Parameter|Description|
|--|--|
|`--wbits`|Number of bits for each block of the VLM, as `vision,proj,llm`. For example, `--wbits 2,4,8` quantizes the vision block to 2 bits, the projector to 4 bits, and the language model to 8 bits|
|`--skip_last_{language,proj,vision}`|if set, skips the last layer of the specified block|
|`--exclude-conv`|Whether to exclude `nn.Conv2D` layers in the quantization process (i.e. only quantize linear layers)|

Notes:
- `--skip_last_vision` skips `model.vision_tower.vision_model.encoder.layers.23.mlp.fc2`
- `--skip_last_proj` skips the `linear_2` layer of the projector
- `--skip_last_language` skips the `lm_head` layer


## Evaluation

### VQAv2
Pass the `--eval` flag to evaluate the model on the VQAv2 benchmark.
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