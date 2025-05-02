# Street Fighter II Turbo Dataset Collection API

## Running the game and executing the API code

1. For running a single bot (your bot vs CPU), open the `single-player` folder. For running two of your bots at the same time both fighting each other, open the `two-players` folder.
2. Run `EmuHawk.exe`.
3. From the File drop down, choose **Open ROM**. (Shortcut: Ctrl+O)
4. From the same `single-player` folder, choose the `Street Fighter II Turbo (U).smc` file.
5. From Tools drop down, open the **Tool Box**. (Shortcut: Shift+T)
6. Once you have performed the above steps, leave the emulator window and tool box open and open the command prompt in the directory of the API and run the following command:

   ```
   python controller.py "1" "record"
   ```

   This will allow for recording your moves.

7. After executing the code, go and select your character(s) in the game after choosing normal mode. Controls for selecting players can be set or seen from the controllers option in the config drop down of the emulator.
8. Now click on the second icon in the top row (**Gyroscope Bot**). This will cause the emulator to establish a connection with the program you ran and you will see "Connected to game" or "CONNECTED SUCCESSFULLY".
9. Congratulations, the dataset will be created for the character you are playing and it will be appended with the moves you do.

The generated dataset will be saved in the `../normalized_character_datasets` folder relative to the API directory.
