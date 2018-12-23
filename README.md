# Deep Space Invaders

A simple convolutional neural network which plays Space Invaders, trained on human play data.

### Prerequisites

* Python 2.7 (Anaconda)
* keras with tensorflow backend
* gym

## Usage

Run space_invaders_human.py and play Space Invaders to the best of your ability to generate data. The game 
waits for your input to progress to the next frame. Use the W, A and D keys to shoot and move, and press S
to progress a frame without any action (this input and the corresponding frame will not be included in the
data.

Repeat the above step as many times as possible - the more data the better. Then, move all of the data you
wish to include into a subfolder called Human_Play_Data, and run data_concat.py which will collect all the
files you put into the folder and create one big data file which includes all the data from each.

Now you are ready to train the model. Simply run agent_network_trainer.py and it will train a CNN for 500
epochs (currently the only way to adjust this and other parameters is to edit the code directly). A graph
will be displayed on completion showing you the training and validation accuracy. This method of AI
gameplay is a fairly innefective one - you will likely find (as I did) that the training loss continues to
decline while the validation loss remains the same or even rises, meaning that the model is simply
"memorizing" the training data instead of learning to generalize it effectively. More on this below under
"Results".

Running space_invaders_pseudohuman.py will load up your newly trained model and allow you to watch it play
a game.

si_plot.py will run the agent 100 times and display a colorful graph of the distribution of scores received
throughout. This is useful for comparing its effectiveness - for reference, a naive "random" agent is
included as space_invaders_random.py, you can run si_plot.py on this agent by replacing the import statement
at the top from "import space_invaders_pseudohuman as agent" to "import space_invaders_random as agent"

## Results

At first, human play data was collected and the model was trained on it as described above. As you can see 
from the graph below, the network overfit the data pretty severely.

![Attempt 1](/graphs/FirstCNN_overfit.PNG?raw=true "Attempt 1")

Several more attempts were made - more data was collected and different hyperparameter tweaks were made
each time. In spite of the changes, no improvement was seen. In fact, the first attempt above may have
been the most successful, as its validation loss actually decreased briefly at the beginning of training.
The loss graph of another attempt is included below.

![Attempt 2](/graphs/SecondCNN_overfit.png?raw=true "Attempt 2")

In spite of the overfitting, the model was shown to be at least performing better than a naive "random"
agent. Below is the distribution of scores received across 100 games for the random agent. 

![Random Agent](/graphs/random_agent_100_color.PNG?raw=true "Random Agent")

And here is the graph of our overfitted network's scores:

![NN Agent](/graphs/PH_agent_100.PNG?raw=true "NN Agent")

Clearly this strategy is at least somewhat effective for this purpose, but it is likely that with other
games it would not work nearly as well. Space Invaders starts you off in the same state each time, and 
so this agent is able to score a few points without having to actually know anything about the game, it
needs only remember a few moves it can make right after the start state to quickly score some points,
and watching it play you can see this is essentially what it does. A much more effective strategy to use
would be something like Deep Q Learning.
