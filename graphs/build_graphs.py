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

   file = open('no_training_20x20grid.txt')
   for row in file:
      row = row.split(' ')
      scoreList.append(int(row[1]))
      recordList.append(int(row[2]))
   file.close()

   #cumulativeScoreList = np.cumsum(scoreList)
   #ax[1].plot(cumulativeScore)
   #ax[1].set_ylabel("Cumul. Score", fontsize=10)

   windowSize = 20
   movingAverageList = moving_average(scoreList, windowSize)

   # Create a 3x1 plot
   fig = plt.figure()
   fig, ax = plt.subplots(3, 1, sharex='col', sharey='row')
   
   # Title
   ax[0].set_title('No Training (Random Policy) 20x20 Grid')

   # Score plot
   ax[0].plot(scoreList)
   ax[0].set_ylabel("Score", fontsize=10)
   #ax[0].set_ylim([0,100])   #y-axis for trained policies
   ax[0].set_ylim([0, 5])    #y-axis for no policy

   # Moving average plot
   ax[1].plot(movingAverageList)
   ax[1].set_ylabel("Score\n(%s Game MA)" % (str(windowSize)), fontsize=10)
   #ax[1].set_ylim([0,60])    #y-axis for trained policies
   ax[1].set_ylim([0, 5])    #y-axis for no policy

   # Record plot
   ax[2].plot(recordList)
   ax[2].set_ylabel("Record", fontsize=10)
   ax[2].set_xlabel("Game Number", fontsize=10)
   #ax[2].set_ylim([0,100])   #y-axis for trained policies
   ax[2].set_ylim([0, 5])    #y-axis for no policy

   plt.show()
