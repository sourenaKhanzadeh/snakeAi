import numpy as np
#import pandas as pd
#import seaborn as sb
import matplotlib.pyplot as plt
plt.style.use('ggplot')

if __name__ == "__main__":
   #data = pd.read_csv('test.txt', sep=" ", header=None)
   #data.columns = ["game_num", "score", "record"]

   game_num = []
   score = []
   record = []

   file = open('default_parameters_20x20grid.txt')
   for row in file:
      row = row.split(' ')
      game_num.append(int(row[0]))
      score.append(int(row[1]))
      record.append(int(row[2]))
   file.close()

   cumulativeScore = np.cumsum(score)
   
   fig = plt.figure()
   fig, ax = plt.subplots(3, 1, sharex='col', sharey='row')
   
   ax[0].set_title('Trained Model')

   ax[0].plot(score)
   ax[0].set_ylabel("Score", fontsize=10)

   ax[1].plot(cumulativeScore)
   ax[1].set_ylabel("Cumul. Score", fontsize=10)

   ax[2].plot(record)
   ax[2].set_ylabel("Record", fontsize=10)
   ax[2].set_xlabel("Game Number", fontsize=10)

   plt.show()
