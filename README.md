[README.md](https://github.com/user-attachments/files/21600942/README.md)
# Prompt-to-Prompt

> *Latent Diffusion* and *Stable Diffusion* Implementation


![teaser](docs/teaser.png)
### [Project Page]([NJUST-wyx/20---prompt-to-prompt](https://github.com/NJUST-wyx/20---prompt-to-prompt))&ensp;&ensp;&ensp;[Paper](https://prompt-to-prompt.github.io/ptp_files/Prompt-to-Prompt_preprint.pdf)


## Install Requirements

This code was tested with cuda >= 11.8, Python 3.8, [jtorch](https://github.com/JittorRepos/jtorch) 2.0.0, [jittor](https://github.com/JittorRepos/jittor) and using pre-trained models through [huggingface / diffusers](https://github.com/JittorRepos/diffusers_jittor).
Specifically, we implemented our method over [Stable Diffusion](https://huggingface.co/CompVis/stable-diffusion-v1-4). Please modify the code in the  prompt-to-prompt_stable.ipynb to use your model path instead.

### 0. Prepare Env

```bash
source ~/.bashrc
conda create -n cross-attention python=3.8 -y
conda activate cross-attention

sudo apt-get update
sudo apt-get install python3.8-dev
ls /root/miniconda3/envs/cross-attention/include/python3.8/Python.h
```

### 1. Install Requirements

If encountering installation problem of the **MD5 mismatch**, you may get some help from [计图安装报错](https://discuss.jittor.org/t/topic/936)

```bash
pip install git+https://github.com/JittorRepos/jittor
pip install git+https://github.com/JittorRepos/jtorch
pip install git+https://github.com/JittorRepos/diffusers_jittor
pip install git+https://github.com/JittorRepos/transformers_jittor

cd path_to/JDiffusion
pip install -e .

pip install opencv-python
pip install ipython
nvcc --version
pip install cupy-cuda11x
pip install jupytext

pip install ipykernel
jupyter kernelspec list
python -m ipykernel install --user --name cross-attention --display-name "Python (cross-attention)"

pip install ipywidgets
```

### 2. Usage

Please manually switch the kernel.

```bash
jupyter nbconvert --to html --execute --KernelManager.kernel_name=cross-attention prompt-to-prompt_stable.ipynb
```

If the environment does not allow the use of Juypter, it can be replaced with prompt-to-prompt_stable_no_use_juypter.py

## Prompt Edits

In our notebooks, we perform our main logic by implementing the abstract class `AttentionControl` object, of the following form:

``` python
class AttentionControl(abc.ABC):
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError
```

The `forward` method is called in each attention layer of the diffusion model during the image generation, and we use it to modify the weights of the attention. Our method (See Section 3 of our [paper](https://arxiv.org/abs/2208.01626)) edits images with the procedure above, and  each different prompt edit type modifies the weights of the attention in a different manner.

The general flow of our code is as follows, with variations based on the attention control type:

``` python
prompts = ["A painting of a squirrel eating a burger", ...]
controller = AttentionControl(prompts, ...)
run_and_display(prompts, controller, ...)
```

### Replacement
In this case, the user swaps tokens of the original prompt with others, e.g., the editing the prompt `"A painting of a squirrel eating a burger"` to `"A painting of a squirrel eating a lasagna"` or `"A painting of a lion eating a burger"`. For this we define the class `AttentionReplace`.

### Refinement
In this case, the user adds new tokens to the prompt, e.g., editing the prompt `"A painting of a squirrel eating a burger"` to `"A watercolor painting of a squirrel eating a burger"`. For this we define the class `AttentionEditRefine`.

### Re-weight
In this case, the user changes the weight of certain tokens in the prompt, e.g., for the prompt `"a smiling bunny doll"`, strengthen the extent to which the word `smiling` affects the resulting image. For this we define the class `AttentionReweight`.


## Attention Control Options
 * `cross_replace_steps`: specifies the fraction of steps to edit the cross attention maps. Can also be set to a dictionary `[str:float]` which specifies fractions for different words in the prompt.
 * `self_replace_steps`: specifies the fraction of steps to replace the self attention maps.
 * `local_blend` (optional):  `LocalBlend` object which is used to make local edits. `LocalBlend` is initialized with the words from each prompt that correspond with the region in the image we want to edit.
 * `equalizer`: used for attention Re-weighting only. A vector of coefficients to multiply each cross-attention weight

## output

The results obtained using the program of Pytorch：prompt-to-prompt_stable1.html

The results obtained using theprogram of Jittor：prompt-to-prompt_stable.html

Our experimental results can be replicated, but the reason for the differences from the original results is that jtorch and jittor may have new random numbers.
