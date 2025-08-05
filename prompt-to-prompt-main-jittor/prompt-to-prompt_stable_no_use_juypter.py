#!/usr/bin/env python
# coding: utf-8

# ## Copyright 2022 Google LLC. Double-click for license information.

# In[1]:


# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# ## Prompt-to-Prompt with Stable Diffusion

# In[1]:


from typing import Optional, Union, Tuple, List, Callable, Dict
import torch
from JDiffusion import StableDiffusionPipeline
import torch.nn.functional as nnf
import numpy as np
import abc

import seq_aligner

#加入输出函数
import ptp_utils_no_use_juypter
import os

import random
import jittor


def set_all_random_seeds(seed=42):
    # 固定Python内置随机模块
    random.seed(seed)

    # 固定NumPy随机种子
    np.random.seed(seed)

    # 如果使用Jittor，固定Jittor的随机种子
    jittor.set_seed(seed)


# 立即调用，确保在所有随机操作前生效
set_all_random_seeds(seed=42)  # 建议使用固定数字（如42），方便复现
# In[ ]:


MY_TOKEN = '<replace with your token>'
LOW_RESOURCE = False 
NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
ldm_stable = StableDiffusionPipeline.from_pretrained("/root/autodl-tmp/stable-diffusion-v1-4", use_auth_token=MY_TOKEN)
tokenizer = ldm_stable.tokenizer


# ## Prompt-to-Prompt Attnetion Controllers
# Our main logic is implemented in the `forward` call in an `AttentionControl` object.
# The forward is called in each attention layer of the diffusion model and it can modify the input attnetion weights `attn`.
# 
# `is_cross`, `place_in_unet in ("down", "mid", "up")`, `AttentionControl.cur_step` help us track the exact attention layer and timestamp during the diffusion iference.
# 

# In[3]:


class LocalBlend:

    def __call__(self, x_t, attention_store):
        k = 1
        maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
        maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, MAX_NUM_WORDS) for item in maps]
        maps = torch.cat(maps, dim=1)
        maps = (maps * self.alpha_layers).sum(-1).mean(1)
        mask = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 +1), (1, 1), padding=(k, k))
        mask = nnf.interpolate(mask, size=(x_t.shape[2:]))
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        #mask = mask.gt(self.threshold)
        mask = mask > self.threshold
        
        mask = (mask[:1] + mask[1:]).float()
        x_t = x_t[:1] + mask * (x_t - x_t[:1])
        return x_t
       
    def __init__(self, prompts: List[str], words: [List[List[str]]], threshold=.3):
        alpha_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = ptp_utils_no_use_juypter.get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1
        self.alpha_layers = alpha_layers
        self.threshold = threshold


class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0
    
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

class EmptyControl(AttentionControl):
    
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        return attn
    
    
class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention


    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

        
class AttentionControlEdit(AttentionStore, abc.ABC):
    
    def step_callback(self, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t
        
    def replace_self_attention(self, attn_base, att_replace):
        if att_replace.shape[2] <= 16 ** 2:
            return attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
        else:
            return att_replace
    
    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError
    
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce = attn[0], attn[1:]
            if is_cross:
                alpha_words = self.cross_replace_alpha[self.cur_step]
                attn_repalce_new = self.replace_cross_attention(attn_base, attn_repalce) * alpha_words + (1 - alpha_words) * attn_repalce
                
                
                # print("attn[1:].shape（原4维）:", attn[1:].shape)  # 例如：(2, 4, 64, 64)
                # print("attn_repalce_new.shape（5维）:", attn_repalce_new.shape)  # 例如：(2, 3, 4, 64, 64)
                if attn_repalce_new.ndim == 5:
                    # 如果是5维，删除第0维（最开始的一维）
                    attn_repalce_new = attn_repalce_new.squeeze(0)  # 用squeeze删除大小为1的维度
                    # 额外检查：删除后是否变为4维，且形状与目标切片匹配
                    assert attn_repalce_new.ndim == 4, "删除第0维后仍不是4维！"
                    assert attn_repalce_new.shape == attn[1:].shape, "删除维度后形状仍不匹配！"
                
                attn[1:] = attn_repalce_new
            else:
                attn[1:] = self.replace_self_attention(attn_base, attn_repalce)
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn
    
    def __init__(self, prompts, num_steps: int,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 self_replace_steps: Union[float, Tuple[float, float]],
                 local_blend: Optional[LocalBlend]):
        super(AttentionControlEdit, self).__init__()
        self.batch_size = len(prompts)
        self.cross_replace_alpha = ptp_utils_no_use_juypter.get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps, tokenizer)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend

class AttentionReplace(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)
      
    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionReplace, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper = seq_aligner.get_replacement_mapper(prompts, tokenizer)
        

class AttentionRefine(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionRefine, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper, alphas = seq_aligner.get_refinement_mapper(prompts, tokenizer)
        self.mapper, alphas = self.mapper, alphas
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])


class AttentionReweight(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        if self.prev_controller is not None:
            attn_base = self.prev_controller.replace_cross_attention(attn_base, att_replace)
        attn_replace = attn_base[None, :, :, :] * self.equalizer[:, None, None, :]
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float, equalizer,
                local_blend: Optional[LocalBlend] = None, controller: Optional[AttentionControlEdit] = None):
        super(AttentionReweight, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.equalizer = equalizer
        self.prev_controller = controller


def get_equalizer(text: str, word_select: Union[int, Tuple[int, ...]], values: Union[List[float],
                  Tuple[float, ...]]):
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(len(values), 77)
    values = torch.tensor(values, dtype=torch.float32)
    for word in word_select:
        inds = ptp_utils_no_use_juypter.get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = values
    return equalizer


# In[5]:


from PIL import Image

def aggregate_attention(attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out.cpu()


def show_cross_attention(attention_store: AttentionStore, res: int, from_where: List[str], select: int = 0):
    tokens = tokenizer.encode(prompts[select])
    decoder = tokenizer.decode
    attention_maps = aggregate_attention(attention_store, res, from_where, True, select)
    images = []
    captions = []
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        image = ptp_utils_no_use_juypter.text_under_image(image, decoder(int(tokens[i])))
        images.append(image)
        captions.append(decoder(int(tokens[i])))
    ptp_utils_no_use_juypter.view_images(np.stack(images, axis=0))
    return ptp_utils_no_use_juypter.view_images_html(images, captions)
    

def show_self_attention_comp(attention_store: AttentionStore, res: int, from_where: List[str],
                        max_com=10, select: int = 0):
    attention_maps = aggregate_attention(attention_store, res, from_where, False, select).numpy().reshape((res ** 2, res ** 2))
    u, s, vh = np.linalg.svd(attention_maps - np.mean(attention_maps, axis=1, keepdims=True))
    images = []
    captions = [f"Component {i+1}" for i in range(max_com)]
    for i in range(max_com):
        image = vh[i].reshape(res, res)
        image = image - image.min()
        image = 255 * image / image.max()
        image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(np.uint8)
        image = Image.fromarray(image).resize((256, 256))
        image = np.array(image)
        images.append(image)
    ptp_utils_no_use_juypter.view_images(np.concatenate(images, axis=1))
    return ptp_utils_no_use_juypter.view_images_html(images, captions)


# In[4]:


def run_and_display(prompts, controller, latent=None, run_baseline=False, generator=None):
    if run_baseline:
        print("w.o. prompt-to-prompt")
        images, latent = run_and_display(prompts, EmptyControl(), latent=latent, run_baseline=False, generator=generator)
        print("with prompt-to-prompt")
    images, x_t = ptp_utils_no_use_juypter.text2image_ldm_stable(ldm_stable, prompts, controller, latent=latent, num_inference_steps=NUM_DIFFUSION_STEPS, guidance_scale=GUIDANCE_SCALE, generator=generator, low_resource=LOW_RESOURCE)

    # # 新增：获取图像路径
    # image_path = images[1]
    # # 新增：生成HTML文件
    # html_content = f"""
    # <!DOCTYPE html>
    # <html>
    # <head>
    #     <meta charset="UTF-8">
    #     <title>Prompt-to-Prompt Output</title>
    # </head>
    # <body>
    #     <h1>Generated Image</h1>
    #     <img src="{image_path}" alt="Generated Image">
    # </body>
    # </html>
    # """
    # output_html_path = 'output.html'
    # with open(output_html_path, 'w', encoding='utf-8') as f:
    #     f.write(html_content)
    # return images[0], x_t

    ptp_utils_no_use_juypter.view_images(images)
    return images, x_t


# ## Cross-Attention Visualization
# First let's generate an image and visualize the cross-attention maps for each word in the prompt.
# Notice, we normalize each map to 0-1.

# In[6]:


report_sections = []  # 仅包含字符串（HTML片段）的列表

# 1. 基础图像生成与注意力可视化
g_cpu = torch.Generator().manual_seed(8888)
prompts = ["A painting of a squirrel eating a burger"]
controller = AttentionStore()

# 添加标题（字符串）
#report_sections.append(ptp_utils_no_use_juypter.text_section_html("1. Base Image Generation"))

# 生成图像（获取numpy数组）
images, x_t = run_and_display(prompts, controller, run_baseline=False, generator=g_cpu)

# 关键修正：将numpy图像转换为HTML字符串后再添加
for img, prompt in zip(images, prompts):
    # 转换单张图像为HTML片段（字符串）
    img_html = ptp_utils_no_use_juypter.view_images_html([img], [prompt])
    report_sections.append(img_html)  # 添加的是字符串，不是numpy数组

# 注意力可视化（确保返回的是HTML字符串）
report_sections.append(ptp_utils_no_use_juypter.text_section_html("Cross-Attention Maps"))
attn_html = show_cross_attention(controller, res=16, from_where=("up", "down"))
# 验证返回类型（防御性编程）
if isinstance(attn_html, str):
    report_sections.append(attn_html)
else:
    report_sections.append(f"<p>Error: 注意力可视化返回非字符串类型</p>")

# 2. 替换编辑示例
prompts_replace = ["A painting of a squirrel eating a burger", "A painting of a lion eating a burger"]
controller_replace = AttentionReplace(prompts_replace, NUM_DIFFUSION_STEPS, cross_replace_steps=.8, self_replace_steps=0.4)
report_sections.append(ptp_utils_no_use_juypter.text_section_html("2. Replacement Edit Example"))

replace_images, _ = run_and_display(prompts_replace, controller_replace, latent=x_t, run_baseline=True)
# 再次确保添加的是HTML字符串
replace_html = ptp_utils_no_use_juypter.view_images_html(replace_images, prompts_replace)
report_sections.append(replace_html)

# # 生成完整HTML（此时所有元素都是字符串，可安全join）
# full_html = ptp_utils_no_use_juypter.html_wrapper("\n".join(report_sections), "Prompt-to-Prompt Results")
# with open("prompt_to_prompt_results.html", "w", encoding="utf-8") as f:
#     f.write(full_html)
# print("HTML report saved as 'prompt_to_prompt_results.html'")




# ### Modify Cross-Attention injection #steps for specific words
# Next, we can reduce the restriction on our lion by reducing the number of cross-attention injection with respect to the word "lion".

# In[8]:


# 1. 替换编辑示例（AttentionReplace）
report_sections.append(ptp_utils_no_use_juypter.text_section_html("1. 基础替换编辑"))
prompts = ["A painting of a squirrel eating a burger",
           "A painting of a lion eating a burger"]
controller = AttentionReplace(
    prompts,
    NUM_DIFFUSION_STEPS,
    cross_replace_steps={"default_": 1., "lion": .4},
    self_replace_steps=0.4
)
# 生成图像并获取结果
images, _ = run_and_display(prompts, controller, latent=x_t, run_baseline=False, generator=g_cpu)
# 转换为HTML并添加到报告
img_html = ptp_utils_no_use_juypter.view_images_html(images, prompts)
report_sections.append(img_html)


# 2. 局部编辑示例（LocalBlend + AttentionReplace）
report_sections.append(ptp_utils_no_use_juypter.text_section_html("2. 局部保留替换编辑"))
report_sections.append(ptp_utils_no_use_juypter.text_section_html("保留汉堡，仅替换松鼠为狮子"))
prompts = ["A painting of a squirrel eating a burger",
           "A painting of a lion eating a burger"]
lb = LocalBlend(prompts, ("squirrel", "lion"))
controller = AttentionReplace(
    prompts,
    NUM_DIFFUSION_STEPS,
    cross_replace_steps={"default_": 1., "lion": .4},
    self_replace_steps=0.4,
    local_blend=lb
)
images, _ = run_and_display(prompts, controller, latent=x_t, run_baseline=False, generator=g_cpu)
img_html = ptp_utils_no_use_juypter.view_images_html(images, prompts)
report_sections.append(img_html)


# 3. 局部替换食物示例
report_sections.append(ptp_utils_no_use_juypter.text_section_html("3. 局部替换食物（汉堡→千层面）"))
prompts = ["A painting of a squirrel eating a burger",
           "A painting of a squirrel eating a lasagne"]
lb = LocalBlend(prompts, ("burger", "lasagne"))
controller = AttentionReplace(
    prompts,
    NUM_DIFFUSION_STEPS,
    cross_replace_steps={"default_": 1., "lasagne": .2},
    self_replace_steps=0.4,
    local_blend=lb
)
images, _ = run_and_display(prompts, controller, latent=x_t, run_baseline=True, generator=g_cpu)
img_html = ptp_utils_no_use_juypter.view_images_html(images, prompts)
report_sections.append(img_html)


# 4. 风格优化编辑（Refinement）
report_sections.append(ptp_utils_no_use_juypter.text_section_html("4. 风格优化编辑"))
report_sections.append(ptp_utils_no_use_juypter.text_section_html("添加新古典主义风格"))
prompts = ["A painting of a squirrel eating a burger",
           "A neoclassical painting of a squirrel eating a burger"]
controller = AttentionRefine(
    prompts,
    NUM_DIFFUSION_STEPS,
    cross_replace_steps=.5,
    self_replace_steps=.2
)
images, _ = run_and_display(prompts, controller, latent=x_t, generator=g_cpu)
img_html = ptp_utils_no_use_juypter.view_images_html(images, prompts)
report_sections.append(img_html)


# 5. 场景季节变化
report_sections.append(ptp_utils_no_use_juypter.text_section_html("5. 场景季节变化"))
# 秋季
report_sections.append(ptp_utils_no_use_juypter.text_section_html("秋季场景"))
prompts = ["a photo of a house on a mountain",
           "a photo of a house on a mountain at fall"]
controller = AttentionRefine(
    prompts,
    NUM_DIFFUSION_STEPS,
    cross_replace_steps=.8,
    self_replace_steps=.4
)
images, _ = run_and_display(prompts, controller, latent=x_t, generator=g_cpu)
img_html = ptp_utils_no_use_juypter.view_images_html(images, prompts)
report_sections.append(img_html)

# 冬季
report_sections.append(ptp_utils_no_use_juypter.text_section_html("冬季场景"))
prompts = ["a photo of a house on a mountain",
           "a photo of a house on a mountain at winter"]
controller = AttentionRefine(
    prompts,
    NUM_DIFFUSION_STEPS,
    cross_replace_steps=.8,
    self_replace_steps=.4
)
images, _ = run_and_display(prompts, controller, latent=x_t, generator=g_cpu)
img_html = ptp_utils_no_use_juypter.view_images_html(images, prompts)
report_sections.append(img_html)


# 6. 食物细化编辑
report_sections.append(ptp_utils_no_use_juypter.text_section_html("6. 食物细化编辑（汤→豌豆汤）"))
prompts = ["soup", "pea soup"]
lb = LocalBlend(prompts, ("soup", "soup"))
controller = AttentionRefine(
    prompts,
    NUM_DIFFUSION_STEPS,
    cross_replace_steps=.8,
    self_replace_steps=.4,
    local_blend=lb
)
images, _ = run_and_display(prompts, controller, latent=x_t, run_baseline=False, generator=g_cpu)
img_html = ptp_utils_no_use_juypter.view_images_html(images, prompts)
report_sections.append(img_html)


# 7. 注意力重加权（Attention Re-Weighting）
report_sections.append(ptp_utils_no_use_juypter.text_section_html("7. 注意力重加权"))
# 增强"smiling"权重
report_sections.append(ptp_utils_no_use_juypter.text_section_html("增强'smiling'注意力权重（×5）"))
prompts = ["a smiling bunny doll"] * 2
equalizer = get_equalizer(prompts[1], ("smiling",), (5,))  # 权重放大5倍
controller = AttentionReweight(
    prompts,
    NUM_DIFFUSION_STEPS,
    cross_replace_steps=.8,
    self_replace_steps=.4,
    equalizer=equalizer
)
images, _ = run_and_display(prompts, controller, latent=x_t, run_baseline=False, generator=g_cpu)
img_html = ptp_utils_no_use_juypter.view_images_html(images, ["原始", "增强'smiling'注意力"])
report_sections.append(img_html)


# 8. 局部注意力调整（粉色自行车→粉色小熊）
report_sections.append(ptp_utils_no_use_juypter.text_section_html("8. 局部注意力调整（减少自行车的粉色）"))
prompts = ["pink bear riding a bicycle"] * 2
equalizer = get_equalizer(prompts[1], ("pink",), (-1,))  # 降低"pink"权重
lb = LocalBlend(prompts, ("bicycle", "bicycle"))  # 仅在自行车区域应用调整
controller = AttentionReweight(
    prompts,
    NUM_DIFFUSION_STEPS,
    cross_replace_steps=.8,
    self_replace_steps=.4,
    equalizer=equalizer,
    local_blend=lb
)
images, _ = run_and_display(prompts, controller, latent=x_t, run_baseline=False, generator=g_cpu)
img_html = ptp_utils_no_use_juypter.view_images_html(images, ["原始", "减少自行车粉色"])
report_sections.append(img_html)


# 9. 注意力增强（面包丁）
report_sections.append(ptp_utils_no_use_juypter.text_section_html("9. 注意力增强（突出面包丁）"))
# 基础编辑
report_sections.append(ptp_utils_no_use_juypter.text_section_html("基础编辑：汤→带面包丁的豌豆汤"))
prompts = ["soup", "pea soup with croutons"]
lb = LocalBlend(prompts, ("soup", "soup"))
controller_a = AttentionRefine(
    prompts,
    NUM_DIFFUSION_STEPS,
    cross_replace_steps=.8,
    self_replace_steps=.4,
    local_blend=lb
)
images_a, _ = run_and_display(prompts, controller_a, latent=x_t, run_baseline=False, generator=g_cpu)
img_html_a = ptp_utils_no_use_juypter.view_images_html(images_a, prompts)
report_sections.append(img_html_a)

# 增强"croutons"注意力
report_sections.append(ptp_utils_no_use_juypter.text_section_html("增强'croutons'注意力权重（×3）"))
equalizer = get_equalizer(prompts[1], ("croutons",), (3,))
controller = AttentionReweight(
    prompts,
    NUM_DIFFUSION_STEPS,
    cross_replace_steps=.8,
    self_replace_steps=.4,
    equalizer=equalizer,
    local_blend=lb,
    controller=controller_a
)
images, _ = run_and_display(prompts, controller, latent=x_t, run_baseline=False, generator=g_cpu)
img_html = ptp_utils_no_use_juypter.view_images_html(images, ["基础编辑", "增强面包丁注意力"])
report_sections.append(img_html)


# 10. 土豆→炸土豆（增强"fried"权重）
report_sections.append(ptp_utils_no_use_juypter.text_section_html("10. 强化烹饪状态（增强'fried'权重）"))
# 基础编辑
report_sections.append(ptp_utils_no_use_juypter.text_section_html("基础编辑：土豆→炸土豆"))
prompts = ["potatos", "fried potatos"]
lb = LocalBlend(prompts, ("potatos", "potatos"))
controller = AttentionRefine(
    prompts,
    NUM_DIFFUSION_STEPS,
    cross_replace_steps=.8,
    self_replace_steps=.4,
    local_blend=lb
)
images_b, _ = run_and_display(prompts, controller, latent=x_t, run_baseline=False, generator=g_cpu)
img_html_b = ptp_utils_no_use_juypter.view_images_html(images_b, prompts)
report_sections.append(img_html_b)

# 增强"fried"注意力
report_sections.append(ptp_utils_no_use_juypter.text_section_html("增强'fried'注意力权重（×10）"))
equalizer = get_equalizer(prompts[1], ("fried",), (10,))
controller = AttentionReweight(
    prompts,
    NUM_DIFFUSION_STEPS,
    cross_replace_steps=.8,
    self_replace_steps=.4,
    equalizer=equalizer,
    local_blend=lb,
    controller=controller  # 复用基础编辑的controller
)
images, _ = run_and_display(prompts, controller, latent=x_t, run_baseline=False, generator=g_cpu)
img_html = ptp_utils_no_use_juypter.view_images_html(images, ["基础编辑", "增强'fried'注意力"])
report_sections.append(img_html)


# 生成完整HTML报告
full_html = ptp_utils_no_use_juypter.html_wrapper("\n".join(report_sections), "Prompt-to-Prompt 完整编辑结果")
with open("prompt_to_prompt_complete_report.html", "w", encoding="utf-8") as f:
    f.write(full_html)
print("完整HTML报告已保存为 'prompt_to_prompt_complete_report.html'")


# In[ ]:




