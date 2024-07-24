# Migrate to New Game Guide

This is a simplified guide for migrating to a new game. Specific details need to be considered accordingly.

### 1. Implement the game's runner

You can see a series of runners in the **cradle/runner/** directory, such as cradle/runner/skylines_runner.py. This file defines the main processes for all games, including information gathering and the execution logic for self-reflection. We recommend directly copying cradle/runner/skylines_runner.py, renaming it for the new game (**newgame_runner.py**), and then modifying it as needed.

**It is worth mentioning that we will eventually unify all runners into a single runner. In the final version of CRADLE, it will no longer be necessary to implement this file separately**.

### 2. Implement the game's environment
Using Skylines as an example, the game's environment mainly consists of three parts:
- **cradle/environment/skylines/atomic_skills** (definition of atomic skills)
- **cradle/environment/skylines/skill_registry.py** (registration of atomic skills in the registry)
- **cradle/environment/skylines/ui_control.py** (game interface control, such as switching games and pausing the game, which need to be defined here)

So the new game should have a similar structure:
- **cradle/environment/newgame/atomic_skills**
- **cradle/environment/newgame/skill_registry.py**
- **cradle/environment/newgame/ui_control.py**

The skill registry does not need specific adaptation. We recommend directly copying cradle/environment/skylines/skill_registry.py and changing the name to newgame.

For ui control, the pause_game and unpause_game functions need to be adapted to the corresponding game. If these functions are not needed, you can directly copy cradle/environment/skylines/ui_control.py and change the name to newgame.

Atomic skills need to be customized based on the game's specific atomic skills, which require a complete reimplementation.

### 3. Implement the game's prompt

For Skylines, prompts for various modules like information gathering and self-reflection are defined in the res/skylines/prompts/templates directory. We recommend directly copying these prompts to the new game directory and modifying the prompts for each module according to the new game.

**It is worth mentioning that since each game is different, the prompts may be entirely different. However, we provide a unified prompt framework, and users can modify the prompts according to this format**.

