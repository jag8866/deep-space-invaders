'''
Runs an agent 100 times and graphs the scores it receives.
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import space_invaders_pseudohuman as agent

#Run the agent 100 times and record each score
scores = []
for i in range(100):
    try:
        scores.append(agent.run())
    except:
        print("Error when running space invaders")

#Organize the data for the graph
n, bins, patches = plt.hist(np.array(scores), 30, facecolor='blue', alpha=0.5)
fracs = n / n.max()
#Normalize data for pretty colors
norm = colors.Normalize(fracs.min(), fracs.max())
#Create the graph
for thisfrac, thispatch in zip(fracs, patches):
    color = plt.cm.viridis(norm(thisfrac))
    thispatch.set_facecolor(color)

plt.xlabel('Score')
plt.ylabel('Games')
plt.title('Score Distribution for Pseudo Human Agent Across 100 Games')
#plt.plot(rand_scores)
plt.show()
