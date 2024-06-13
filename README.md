# UAC
Repository for the Universal Agent Control project.

Please setup your environment following [this page](./ENVIRONMENTS.md).

## Infra code

### OpenAI Provider

OpenAI provider now can expose embeddings and LLM from OpenAI and Azure together. Users only need to create one instance of each and pass the appropriate configuration.

Example configurations are in /conf. To avoid exposing sensitive details, keys and other private info should be defined in environmental variables.

The suggested way to do it is to create a .env file in the root of the repository (never push this file to GitHub) where variables can be defined, and then mention the variable names in the configs.

Please check the examples below.

Sample .env file containing private info that should never be on git/GitHub:
```
OA_OPENAI_KEY = "abc123abc123abc123abc123abc123ab"
AZ_OPENAI_KEY = "123abc123abc123abc123abc123abc12"
AZ_BASE_URL = "https://abc123.openai.azure.com/"
```

Sample config for an OpenAI provider:
```
{
	"key_var" : "OA_OPENAI_KEY",
	"emb_model": "text-embedding-ada-002",
	"comp_model": "gpt-4-vision-preview",
	"is_azure": false
}
```

## General Guidelines

**>>> Allways check the /main branch for the latest examples!!!**

Any file with text content in the project in the resources directory (./res) should be in UTF-8 encoding. Use the uac.utils to open/save files.


### File Structure

#### 1. Prompt Definition Files

Prompt files are located in the repo at ./res/&lt;emviornment_name&gt;/prompts. Out of the code tree.

There two types of prompt-related files: input_example, and templates. Examples are json files. Templates are text files with special markers inside. See details below.

Inside each of these directories, most files will fit into five categories: decision_making, gather_information, information_summary, self_reflection and success_detection

Files are named according to the format: ./res/&lt;emviornment_name&gt;/prompts/&lt;type&gt;/&lt;category&gt;_&lt;sub_task&gt;.&lt;ext&gt;

As shown below, such "input_example" files illustrate the **parameters** needed to fill a prompt "template".
Not all input examples need the same parameters. Only the parameters required the corresponding specific template (".prompt" file).

#### 2. Skills Definition Files

Most of our code are in the cradle/environment/.

#### cradle/environment/&lt;emviornment_name&gt;/atomic_skills:

Atomic Skills refers to basic skills or minimal skill units that form the basis of more complex skills. Such as turn() and move_forward().

#### cradle/environment/&lt;emviornment_name&gt;/composite_skills:

Composite skills refer to more complex skills that are composed of multiple atomic skills. For example, follow() is a combination of turn() and moveforward().

#### cradle/environment/&lt;emviornment_name&gt;/lifecycle/ui_control.py

Contains code for switch game and code between two desktops and take_screenshot of the game.

## Running Examples

1. Set up the environment in .vscode/launch.json. Change the "--providerConfig" and "--envConfig" to the environment you want to run.
2. Run the runner.py to see whether the environment is set up correctly.
3. Run the prototype_runner.py to run the whole pipline. Or run the skill_example.py to run one skill part.
