# Cradle: Empowering Foundation Agents Towards General Computer Control

<div align="center">

[[Website]](https://baai-agents.github.io/Cradle/)
[[arXiv]](https://arxiv.org/abs/2403.03186)
[[PDF]](https://arxiv.org/pdf/2403.03186)

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
<a alt="Watch the video" href="https://www.youtube.com/watch?v=fkkSJw1iJJ8"><img src="./docs/envs/images/rdr2/RDR2_story_cover.jpg" width="33%" /></a>
&nbsp;&nbsp;
<a alt="Watch the video" href="https://www.youtube.com/watch?v=ay5gBqzPcDE"><img src="./docs/envs/images/rdr2/RDR2_openended_cover.jpg" width="33%" /></a>
&nbsp;&nbsp;
<a alt="Watch the video" href="https://www.youtube.com/watch?v=regULK_60_8"><img src="./docs/envs/images/skylines/cityskyline_video_cover.png" width="33%" /></a>
&nbsp;&nbsp;
<a alt="Watch the video" href="https://www.youtube.com/watch?v=Kaiz4yJieUk"><img src="./docs/envs/images/stardew/stardew_video_cover.png" width="33%" /></a>
&nbsp;&nbsp;
<a alt="Watch the video" href="https://www.youtube.com/watch?v=WZiL_0V880M"><img src="./docs/envs/images/dealers/dealer_video_cover.png" width="33%" /></a>
&nbsp;&nbsp;
<a alt="Watch the video" href="https://www.youtube.com/watch?v=uWgLnZmpVTM"><img src="./docs/envs/images/software/Software_cover.png" width="33%" /></a>
&nbsp;&nbsp;
</div>

Click on either of the video thumbnails above to watch them on YouTube.

# 💾 Installation

## Prepare the Environment File
We currently provide access to OpenAI's and Claude's API. Please create a `.env` file in the root of the repository to store the keys (one of them is enough).

Sample `.env` file containing private information:
```
OA_OPENAI_KEY = "abc123abc123abc123abc123abc123ab"
RF_CLAUDE_AK = "abc123abc123abc123abc123abc123ab" # Access Key for Claude
RF_CLAUDE_SK = "123abc123abc123abc123abc123abc12" # Secret Access Key for Claude
AZ_OPENAI_KEY = "123abc123abc123abc123abc123abc12"
AZ_BASE_URL = "https://abc123.openai.azure.com/"
RF_CLAUDE_AK = "abc123abc123abc123abc123abc123ab"
RF_CLAUDE_SK = "123abc123abc123abc123abc123abc12"
IDE_NAME = "Code"
```
OA_OPENAI_KEY is the OpenAI API key. You can get it from the [OpenAI](https://platform.openai.com/api-keys).

AZ_OPENAI_KEY is the Azure OpenAI API key. You can get it from the [Azure Portal](https://portal.azure.com/#view/HubsExtension/BrowseResource/resourceType/Microsoft.CognitiveServices%2Faccounts).

OA_CLAUDE_KEY is the Anthropic Claude API key. You can get it from the [Anthropic](https://console.anthropic.com/settings/keys).

RF_CLAUDE_AK and RF_CLAUDE_SK are AWS Restful API key and secret key for Claude API.

IDE_NAME refers to the IDE environment in which the repository's code runs, such as `PyCharm` or `Code` (VSCode). It is primarily used to enable automatic switching between the IDE and the target environment.


## Setup

### Python Environment
Please setup your python environment and install the required dependencies as:
```bash
# Clone the repository
git clone https://github.com/BAAI-Agents/Cradle.git
cd Cradle

# Create a new conda environment
conda create --name cradle-dev python=3.10
conda activate cradle-dev
pip install -r requirements.txt
```

### Install the OCR Tools
```
1. Option 1
# Download best-matching version of specific model for your spaCy installation
python -m spacy download en_core_web_lg

or

# pip install .tar.gz archive or .whl from path or URL
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.7.1/en_core_web_lg-3.7.1.tar.gz

2. Option 2
# Copy this url https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.7.1/en_core_web_lg-3.7.1.tar.gz
# Paste it in the browser and download the file to res/spacy/data
cd res/spacy/data
pip install en_core_web_lg-3.7.1.tar.gz
```

# 🚀 Get Started
Due to the vast differences between each game and software, we have provided the specific settings for each of them below.
1. [Red Dead Redemption 2](docs/envs/rdr2.md)
2. [Stardew Valley](docs/envs/stardew.md)
3. [Cities: Skylines](docs/envs/skylines.md)
4. [Dealer's Life 2](docs/envs/dealers.md)
5. [Software](docs/envs/software.md)

<div align="center">
<img src="./docs/images/games_wheel.png" height="365" /> <img src="./docs/images/applications_wheel.png" height="365" />
</div>

# 🌲 File Structure
Since some users may want to apply our framework to new games, this section primarily showcases the core directories and organizational structure of Cradle. We will highlight in "⭐⭐⭐" the modules related to migrating to new games, and provide detailed explanations later.
```
Cradle
├── cache # Cache the GroundingDino model and the bert-base-uncased model
├── conf # ⭐⭐⭐ The configuration files for the environment and the llm model
│   ├── env_config_dealers.json
│   ├── env_config_rdr2_main_storyline.json
│   ├── env_config_rdr2_open_ended_mission.json
│   ├── env_config_skylines.json
│   ├── env_config_stardew_cultivation.json
│   ├── env_config_stardew_farm_clearup.json
│   ├── env_config_stardew_shopping.json
│   ├── openai_config.json
│   ├── claude_config.json
│   ├── restful_claude_config.json
│   └── ...
├── deps # The dependencies for the Cradle framework, ignore this folder
├── docs # The documentation for the Cradle framework, ignore this folder
├── res # The resources for the Cradle framework
│   ├── models # Ignore this folder
│   ├── tool # Subfinder for RDR2
│   ├── [game or software] # ⭐⭐⭐ The resources for game, exmpale: rdr2, dealers, skylines, stardew, outlook, chrome, capcut, meitu, feishu
│   │   ├── prompts # The prompts for the game
│   │   │   └── templates
│   │   │       ├── action_planning.prompt
│   │   │       ├── information_gathering.prompt
│   │   │       ├── self_reflection.prompt
│   │   │       └── task_inference.prompt
│   │   ├── skills # The skills json for the game, it will be generated automatically
│   │   ├── icons # The icons difficult for GPT-4 to recognize in the game can be replaced with text for better recognition using an icon replacer
│   │   └── saves # Save files in the game
│   └── ...
├── requirements.txt # The requirements for the Cradle framework
├── runner.py # The main entry for the Cradle framework
├── cradle # Cradle's core modules
│   ├── config # The configuration for the Cradle framework
│   ├── environment # The environment for the Cradle framework
│   │   ├── [game or software] # ⭐⭐⭐ The environment for the game, exmpale: rdr2, dealers, skylines, stardew, outlook, chrome, capcut, meitu, feishu
│   │   │   ├── __init__.py # The initialization file for the environment
│   │   │   ├── atomic_skills # Atomic skills in the game. Users should customise them to suit the needs of the game or software, e.g. character movement
│   │   │   ├── composite_skills # Combination skills for atomic skills in games or software
│   │   │   ├── skill_registry.py # The skill registry for the game. Will register all atomic skills and composite skills into the registry.
│   │   │   └── ui_control.py # The UI control for the game. Define functions to pause the game and switch to the game window
│   │   └── ...
│   ├── gameio # Interfaces that directly wrap the skill registry and ui control in the environment
│   ├── log # The log for the Cradle framework
│   ├── memory # The memory for the Cradle framework
│   ├── module # Currently there is only the skill execution module. Later will migrate action planning, self-reflection and other modules from planner and provider
│   ├── planner # The planner for the Cradle framework. Unified interface for action planning, self-reflection and other modules. This module will be deleted later and will be moved to the module module.
│   ├── runner # ⭐⭐⭐ The logical flow of execution for each game and software. All game and software processes will then be unified into a single runner
│   ├── utils # Defines some helper functions such as save json and load json
│   └── provider # The provider for the Cradle framework. We have semantically decomposed most of the execution flow in the runner into providers
│       ├── augment # Methods for image augmentation
│       ├── llm # Call for the LLM model, e.g. OpenAI's GPT-4o, Claude, etc.
│       ├── module # ⭐⭐⭐ The module for the Cradle framework. e.g., action planning, self-reflection and other modules. It will be migrated to the cradle/module later.
│       ├── object_detect # Methods for object detection
│       ├── process # ⭐⭐⭐ Methods for pre-processing and post-processing for action planning, self-reflection and other modules
│       ├── video # Methods for video processing
│       ├── others # Methods for other operations, e.g., save and load coordinates for skylines
│       ├── circle_detector.py # The circle detector for the rdr2
│       ├── icon_replacer.py # Methods for replacing icons with text
│       ├── sam_provider.py # Segment anything for software
│       └── ...
└── ...
```

# 📚 Migrate to New Game
Since each game's settings and the operating systems they are compatible with are different, Cradle cannot simply replace one game name to migrate to a new game. We suggest considering each game specifically. For example, RDR2, an independent AAA game, requires real-time combat, so we need to pause the game to wait for GPT-4o's response and then unpause the game to execute the actions. Stardew has the same issue. Other games like Dealer's Life 2 and Cities: Skylines do not have real-time requirements, so they do not need to pause. If the new game is similar to the latter, we recommend copying Cities: Skylines' implementation and following its implementation path to create the corresponding modules. Although each game may differ significantly, our Cradle framework can still achieve a unified adaptation for a game. Assuming the new game's name is **newgame**, the specific migration pipeline can be found [Migrate to New Game Guide](docs/envs/new_game.md). 

# Citation
If you find our work useful, please consider citing us!
```
@article{tan2024cradle,
  title={Cradle: Empowering Foundation Agents towards General Computer Control},
  author={Weihao Tan and Wentao Zhang and Xinrun Xu and Haochong Xia and Ziluo Ding and Boyu Li and Bohan Zhou and Junpeng Yue and Jiechuan Jiang and Yewen Li and Ruyi An and Molei Qin and Chuqiao Zong and Longtao Zheng and Yujie Wu and Xiaoqiang Chai and Yifei Bi and Tianbao Xie and Pengjie Gu and Xiyun Li and Ceyao Zhang and Long Tian and Chaojie Wang and Xinrun Wang and Börje F. Karlsson and Bo An and Shuicheng Yan and Zongqing Lu},
  journal={arXiv preprint arXiv:2403.03186},
  year={2024}
}
```
[//]: # (```)
[//]: # (@article{weihao2024cradle,)
[//]: # (  title     = {{Cradle: Empowering Foundation Agents towards General Computer Control}},)
[//]: # (  author    = {Weihao Tan and Wentao Zhang and Xinrun Xu and Haochong Xia and Ziluo Ding and Boyu Li and Bohan Zhou and Junpeng Yue and Jiechuan Jiang and Yewen Li and Ruyi An and Molei Qin and Chuqiao Zong and Longtao Zheng and Yujie Wu and Xiaoqiang Chai and Yifei Bi and Tianbao Xie and Pengjie Gu and Xiyun Li and Ceyao Zhang and Long Tian and Chaojie Wang and Xinrun Wang and Börje F. Karlsson and Bo An and Shuicheng Yan and Zongqing Lu},)
[//]: # (  journal   = {arXiv:2403.03186},)
[//]: # (  month     = {March},)
[//]: # (  year      = {2024},)
[//]: # (  primaryClass={cs.AI})
[//]: # (})
[//]: # (```)

## FAQ

### What is Cradle?
Cradle is a framework that empowers foundation models to perform **General Computer Control (GCC)** - using the same unified interface humans use: screenshots as input and keyboard/mouse operations as output.

### How does Cradle differ from other agent frameworks?
| Feature | Cradle | LangChain | CrewAI |
|---------|---------|-----------|--------|
| Approach | **Screenshot → K&M** | Tool-based | Role-based |
| Interface | Human-like | API-based | Agent-based |
| Target | **Any software** | Specific tools | Defined roles |
| Game Support | ✅ RDR2/Stardew/etc | ⚠️ Limited | ❌ None |

### What games/software does Cradle support?
| Category | Examples |
|----------|----------|
| Games | RDR2, Stardew Valley, Cities: Skylines, Dealer's Life 2 |
| Software | Chrome, Outlook, Capcut, Meitu, Feishu |
| General | Any computer task via screenshots |

### What are the key components?
- **Screenshot Input** - Understands UI like humans
- **Keyboard/Mouse Output** - Controls any interface
- **Foundation Models** - LLM-powered reasoning
- **Unified Interface** - Same as human interaction

### What Python version is required?
Python 3.10+

### What license does Cradle use?
MIT License - Free for personal and commercial use.

### Where can I find more information?
- 📚 **Paper**: [arXiv](https://arxiv.org/abs/2403.03186)
- 🌐 **Website**: [baai-agents.github.io/Cradle](https://baai-agents.github.io/Cradle/)
- 🐛 **Issues**: [GitHub Issues](https://github.com/BAAI-Agents/Cradle/issues)
