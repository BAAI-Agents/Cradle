# Dealer's Life 2 

Here are the settings for Dealers' Life 2 game. And this environment is primarily available on Windows.

### Change Settings Before Running the Code
#### Display settings

- Set the **Aspect Ratio** to 16:9
- Set the **Resolution** to 1920X1080
- Set the **Window Mode** to Fullscreen
If you are using Windows, set your monitor resolution to 1920X1080.  
If you are using MacOS, you need to have an additional monitor and set your external monitor as main screen and set it to 1920X1080.  
![resolution](../envs_images/dealers/resolution.png)

### Libraries for Keyboard & Mouse Control

- pyautogui: Used to simulate mouse clicks, including long mouse presses.

### Running Cradle on Dealer's Life 2
1. Ensure that the game settings are set up as described above.
2. Launch the Dealer's Life 2 game.
3. Initialize a new game and set up the character.
4. Finish the tutorial and reach the main game screen.
5. Before further haggling with first customer, launch the framework agent with the command:
```bash
python runner.py --envConfig "./conf/envs/dealers.json"
```