import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def moving_average(arr, window_size):
   newarr = arr[:window_size]
   for i in range(len(arr) - window_size + 1):
      window = arr[i : i+window_size]
      window_average = sum(window) / window_size
      newarr.append(window_average)
   return newarr

if __name__ == "__main__":
   scoreList = []
   recordList = []

   file = open('default_parameters_20x20grid.txt')
   for row in file:
      row = row.split(' ')
      scoreList.append(int(row[1]))
      recordList.append(int(row[2]))
   file.close()

   #cumulativeScoreList = np.cumsum(scoreList)
   windowSize = 20
   movingAverageList = moving_average(scoreList, windowSize)

   fig = plt.figure()
   fig, ax = plt.subplots(3, 1, sharex='col', sharey='row')
   
   ax[0].set_title('Default Parameters 20x20 Grid')

   ax[0].plot(scoreList)
   ax[0].set_ylabel("Score", fontsize=10)

   #ax[1].plot(cumulativeScore)
   #ax[1].set_ylabel("Cumul. Score", fontsize=10)
   ax[1].plot(movingAverageList)
   ax[1].set_ylabel("Score \nMov. Avg = " + str(windowSize), fontsize=10)

   ax[2].plot(recordList)
   ax[2].set_ylabel("Record", fontsize=10)
   ax[2].set_xlabel("Game Number", fontsize=10)

   plt.show()
