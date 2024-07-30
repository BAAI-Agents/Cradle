# Migrate to New Game Guide

This is a simplified guide for migrating to a new game. Specific details need to be considered accordingly.

### 1. Implement the game's configuration
We have set up an entry point for the configuration file. Please refer to conf/env_config_skylines.json for the configuration file corresponding to newgame. Below is a brief introduction to several key parameters in the config:
- **"env_name"**: The name of the game, which is used to identify the game.
- **"skill_registry_name"**: The path to the skill registry file.
- **"ui_control_name"**: The path to the ui control file.
- **"task_description_list"**: The task description list, which is used to describe the task to be completed by the agent.
- **"skill_configs"**: The configuration of atomic skills, including the name of the atomic skill.
- **"provider_configs"**: Because we have currently split the various modules based on semantics, such as information gathering, self-reflection, and additional providers, including the augment (image) provider, their configurations are all under this parameter.

**A kind reminder: Perhaps when you first start using Cradle, you may not know the purpose and usage of each parameter in the config. Please try to refer to the format in conf/env_config_skylines.json and migrate it to the new game. Also, remember to modify the contents of the config after completing the subsequent modules.**

### 2. Implement the game's runner

You can see a series of runners in the **cradle/runner/** directory, such as cradle/runner/skylines_runner.py. This file defines the main processes for all games, including the execution logic for self-reflection and information gathering. We recommend directly copying cradle/runner/skylines_runner.py, renaming it for the new game (**newgame_runner.py**), and then modifying it as needed.

**It is worth mentioning that we will eventually unify all runners into a single runner. In the final version of Cradle, it will no longer be necessary to implement this file separately**.

### 3. Implement the game's environment
Using Cites: Skylines as an example, the game's environment mainly consists of three parts:
- **cradle/environment/skylines/atomic_skills** (definition of atomic skills)
- **cradle/environment/skylines/skill_registry.py** (registration of atomic skills in the registry)
- **cradle/environment/skylines/ui_control.py** (game interface control, such as switching games and pausing the game, which need to be defined here)

So the new game should have a similar structure:
- **cradle/environment/newgame/atomic_skills**
- **cradle/environment/newgame/skill_registry.py**
- **cradle/environment/newgame/ui_control.py**

The skill registry does not need specific adaptation. We recommend directly copying cradle/environment/skylines/skill_registry.py and changing the name to newgame.

For ui control, check whether the game needs to be paused while waiting for GPT-4o's response. If yes, implement it in the ui control. The pause_game and unpause_game functions need to be adapted to the corresponding game. If these functions are not needed, you can directly copy cradle/environment/skylines/ui_control.py and change the name to newgame.

For atomic skills, check whether the game provides enough guidance for generating necessary skills (e.g., RDR2 does not teach us the usage of WASD and how to move the view). If no, implement the missing ones. Some games do not provide any guidance, then you need to implement the skills in advance, which is just like the case in web navigation. Pay attention to the specific/wired mechanisms of the games.
Atomic skills need to be customized based on the game's specific atomic skills, which require a complete reimplementation.

### 4. Implement the game's prompt

For Cities: Skylines, prompts for various modules like information gathering and self-reflection are defined in the res/skylines/prompts/templates directory. We recommend directly copying these prompts to the new game directory and modifying the prompts for each module according to the new game. Iteratively revise the prompt, add augmented tools and run the agent until the agent can complete the task.

**It is worth mentioning that since each game is different, the prompts may be entirely different. However, we provide a unified prompt framework, and users can modify the prompts according to this format.**

**A kind reminder: Prompt engineering is a long-term process that requires continuous iteration and adjustment to adapt to various tasks. Now that you have mastered the basic workflow of Cradle, you are welcome to experience Cradle!**