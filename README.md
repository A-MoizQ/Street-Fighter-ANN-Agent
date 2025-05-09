# Dataset Details

## Finalized Datasets

All these datasets are assumed to be finalized until further notice and their respective models will be trained on them:

- 0
- 1
- 2
- 3
- 4
- 5
- 6
- 8
- 9


## Unfinished Datasets

- 7
- 10
- 11

# Models Details

For now every model is being trained through ./gamebot-competition-master/train_models/train_individual_character.py

 **⚠️ IMPORTANT:** If you want to change the model architechture, do so in another file inside the train_models folder. Don't modify train_individual_character.py. Also when you modify dont push the changes until all the models for the finalized datasets have been trained and tested. Ensure all predictions are being played correctly. You will have to modify ./gamebot-competition-master/PythonAPI/bot.py for this

## Finalized Models

All these models are final according to the architechture of ./gamebot-competition-master/train_models/train_individual_character.py
and have been tested and are working. The combos are not working for some.

- 0
- 1 (didnt test)
- 2 (didnt test)
- 3 (didnt test)
- 4
- 5
- 6
- 8
- 9

## Models Left for Training

The models for the following character datasets need to be trained:

- 7
- 10
- 11

In train_individual_character.py, you only need to write the index of the model in the characters_to_train list. Don't add the indexes of the characters whose models have already been trained (remove them from the list if they are already there). Keep in mind if you update the dataset of an already trained model, you will need to train it again.