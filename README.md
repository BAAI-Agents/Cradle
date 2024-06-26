# Cradle: Empowering Foundation Agents Towards General Computer Control

<div align="center">

[[Website]](https://baai-agents.github.io/Cradle/)
[[Arxiv]](https://arxiv.org/abs/2403.03186)
[[PDF]](https://arxiv.org/pdf/2403.03186.pdf)

[![Python Version](https://img.shields.io/badge/Python-3.10-blue.svg)]()
[![GitHub license](https://img.shields.io/badge/MIT-blue)]()

![](docs/images/cradle-intro-cr.png)

</div>

The Cradle framework empowers nascent foundation models to perform complex computer tasks
via the same unified interface humans use, i.e., screenshots as input and keyboard & mouse operations as output.

## 📢 Updates
- 2024-06-27: A major update! Cradle is extened to four games: [RDR2](https://www.rockstargames.com/reddeadredemption2), [Stardew Valley](https://www.stardewvalley.net/), [Cities: Skylines](https://www.paradoxinteractive.com/games/cities-skylines/about), and [Dealer's Life 2](https://abyteentertainment.com/dealers-life-2/) and various software, including but not limited to Chrome, Outlook, Capcut, Meitu and Feishu. We also release our latest [paper](https://arxiv.org/pdf/2403.03186). Check it out!   

<div align="center">

![](docs/images/gcc.jpg)

</div>

## Latest Videos
<div align="center">
<a alt="Watch the video" href="https://www.youtube.com/watch?v=fkkSJw1iJJ8"><img src="docs/images/RDR2_story_cover.jpg" width="33%" /></a>
&nbsp;&nbsp;
<a alt="Watch the video" href="https://www.youtube.com/watch?v=ay5gBqzPcDE"><img src="docs/images/RDR2_openended_cover.jpg" width="33%" /></a>
&nbsp;&nbsp;
<a alt="Watch the video" href="https://www.youtube.com/watch?v=regULK_60_8"><img src="docs/images/cityskyline_video_cover.png" width="33%" /></a>
&nbsp;&nbsp;
<a alt="Watch the video" href="https://www.youtube.com/watch?v=Kaiz4yJieUk"><img src="docs/images/stardew_video_cover.png" width="33%" /></a>
&nbsp;&nbsp;
<a alt="Watch the video" href="https://www.youtube.com/watch?v=WZiL_0V880M"><img src="docs/images/dealer_video_cover.png" width="33%" /></a>
&nbsp;&nbsp;
<a alt="Watch the video" href="https://www.youtube.com/watch?v=k0K_GbmTthg"><img src="docs/images/Software_cover.png" width="33%" /></a>
&nbsp;&nbsp;
</div>

Click on either of the video thumbnails above to watch them on YouTube.

# 💾 Installation

## Prepare the Environment File
We currently provide access to OpenAI's and Claude's API. Please create a `.env` file in the root of the repository to store the keys (one of them is enough). 

Sample `.env` file containing private information:
```
OA_OPENAI_KEY = "abc123abc123abc123abc123abc123ab"
OA_CLAUDE_KEY = "abc123abc123abc123abc123abc123ab"
RF_CLAUDE_AK = "abc123abc123abc123abc123abc123ab"
RF_CLAUDE_SK = "123abc123abc123abc123abc123abc12"
IDE_NAME = "Code"
```
OA_OPENAI_KEY is the OpenAI API key. You can get it from the [OpenAI](https://platform.openai.com/api-keys).

OA_CLAUDE_KEY is the Anthropic Claude API key. You can get it from the [Anthropic](https://console.anthropic.com/settings/keys).

RF_CLAUDE_AK and RF_CLAUDE_SK are AWS Restful API key and secret key for Claude API.

IDE_NAME refers to the IDE environment in which the repository's code runs, such as `PyCharm` or `Code` (VSCode). It is primarily used to enable automatic switching between the IDE and the game window.


## Setup

### Python Environment
Please setup your python environment and install the required dependencies as:
```bash
# Clone the repository
git clone https://github.com/BAAI-Agents/Cradle.git
cd Cradle

# Create a new conda environment
conda create --name uac-dev python=3.10
conda activate uac-dev
pip install -r requirements.txt
```

#### Install Grounding Dino (only needed for RDR2)
```bash
# Install torch and torchvision
pip install --upgrade torch==2.1.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html
pip install torchvision==0.16.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html

# Install the pre-compiled GroundingDino with the project dependencies
# 1. Download the weights to the cache directory
cd cache
curl -L -C - -O https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth
cd ..
# 2. Download the bert-base-uncased model from Hugging Face
mkdir hf
huggingface-cli download bert-base-uncased config.json tokenizer.json vocab.txt tokenizer_config.json model.safetensors --cache-dir hf
# 3. Install the groundingdino
cd ..
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install -r requirements.txt
pip install .
cd Cradle
```
If you encounter any issues during the installation of GroundingDINO, please refer to the official website [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) or our provided [GroundingDino Installation Guide](docs/envs/groundingdino.md).

#### Download videosubfinder （only needed for RDR2）
Download the videosubfinder from https://sourceforge.net/projects/videosubfinder/ and extract the files into the res/tool/subfinder folder. We have already created the folder for you and included a test.srt, which is a required dummy file that will not affect results.

The file structure should be like this:

```
├── res
  ├── tool
    ├── subfinder
      ├── VideoSubFinderWXW.exe
      ├── test.srt
      ├── ...
```

# 🚀 Get Started
Due to the vast differences between each game and software, we have provided the specific settings for each of them below.
1. [Red Dead Redemption 2](docs/envs/rdr2.md)
2. [Stardew Valley](docs/envs/stardew.md)
3. [Cities: Skylines](docs/envs/skylines.md)
4. [Dealer's Life 2](docs/envs/dealers.md)
5. [Software](docs/envs/software.md)

<div align="center">
<img src="docs/images/games_wheel.png" height="365" /> <img src="docs/images/applications_wheel.png" height="365" />
</div>

# 🌲 File Structure


# Citation
If you find our work useful, please consider citing us!
```
@article{weihao2024cradle,
  title     = {{Towards General Computer Control: A Multimodal Agent For Red Dead Redemption II As A Case Study}},
  author    = {Weihao Tan and Ziluo Ding and Wentao Zhang and Boyu Li and Bohan Zhou and Junpeng Yue and Haochong Xia and Jiechuan Jiang and Longtao Zheng and Xinrun Xu and Yifei Bi and Pengjie Gu and Xinrun Wang and Börje F. Karlsson and Bo An and Zongqing Lu},
  journal   = {arXiv:2403.03186},
  month     = {March},
  year      = {2024},
  primaryClass={cs.AI}
}
```
