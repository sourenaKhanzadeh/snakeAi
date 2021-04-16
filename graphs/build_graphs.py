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
   score_avg = []
   record_avg = []
   for i in range(3):
      scoreList = []
      recordList = []
      file = open(f'scored__20__far__10__close__15__veryfar__5__/scored__20__far__10__close__15__veryfar__5__{i}.txt')
      j = 0
      for row in file:
         
         j += 1
         row = row.split(' ')
         scoreList.append(int(row[1]))
         recordList.append(int(row[2]))

         if j > 300:
            break
      score_avg.append(scoreList)
      record_avg.append(recordList)
      file.close()
   
   score = []
   rec =  []
   print(len(score_avg))
   for item in range(len(score_avg[0])): 
      res = score_avg[0][item] + score_avg[1][item] + score_avg[2][item]
      score.append(round(res/3, 2))

   for item in range(len(record_avg[0])): 
      res = record_avg[0][item] + record_avg[1][item] + record_avg[2][item]
      rec.append(round(res/3, 2))

         
      # cumulativeScoreList = np.cumsum(scoreList)
      # ax[1].plot(cumulativeScore)
      # ax[1].set_ylabel("Cumul. Score", fontsize=10)

   windowSize = 20
   movingAverageList = moving_average(score, windowSize)

   # Create a 3x1 plot
   fig = plt.figure()
   fig, ax = plt.subplots(3, 1, sharex='col', sharey='row')
   
   # Title scored__20__far__10__close__15__veryfar__5__
   ax[0].set_title('SCORED: 20, CLOSE: 15, FAR: 10, VERYFAR: 5')

   # Score plot
   ax[0].plot(score)
   ax[0].set_ylabel("Score", fontsize=10)
   ax[0].set_ylim([0,50])   #y-axis for trained policies
   # ax[0].set_ylim([0, 5])    #y-axis for no policy

   # Moving average plot
   ax[1].plot(movingAverageList)
   ax[1].set_ylabel("Score\n(%s Game MA)" % (str(windowSize)), fontsize=10)
   ax[1].set_ylim([0,60])    #y-axis for trained policies
   # ax[1].set_ylim([0, 5])    #y-axis for no policy

   # Record plot
   ax[2].plot(rec)
   ax[2].set_ylabel("Record", fontsize=10)
   ax[2].set_xlabel("Game Number", fontsize=10)
   # ax[2].set_ylim([0,50])   #y-axis for trained policies
   # ax[2].set_ylim([0, 5])    #y-axis for no policy

   plt.show()
