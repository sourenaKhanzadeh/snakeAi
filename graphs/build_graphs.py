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

def get_score_and_record(folder):
   score_avg = []
   record_avg = []
   for i in range(3):
      scoreList = []
      recordList = []
      file = open(f'{folder}/{folder}{i}.txt')
      j = 0
      for row in file:
         
         j += 1
         row = row.split(' ')
         scoreList.append(int(row[1]))
         recordList.append(int(row[2]))

         if j > 1000:
            break
      score_avg.append(scoreList)
      record_avg.append(recordList)
      file.close()
   
   score = []
   rec =  []

   for item in range(len(score_avg[0])): 
      res = score_avg[0][item] + score_avg[1][item] + score_avg[2][item]
      score.append(round(res/3, 2))

   for item in range(len(record_avg[0])): 
      res = record_avg[0][item] + record_avg[1][item] + record_avg[2][item]
      rec.append(round(res/3, 2))

   return score, rec

if __name__ == "__main__":
   
   # Get score and record from txt files in specified folder
   score_0, rec_0 = get_score_and_record("epsilon__0__")
   score_1, rec_1 = get_score_and_record("epsilon__100__")
   score_2, rec_2 = get_score_and_record("epsilon__200__")
   #score_3, rec_3 = get_score_and_record("folder3__")

   windowSize = 15

   movingAverageList0 = moving_average(score_0, windowSize)
   movingAverageList1 = moving_average(score_1, windowSize)
   movingAverageList2 = moving_average(score_2, windowSize)
   #movingAverageList3 = moving_average(score_3, windowSize)

   # Create a X by Y subplot, change X and Y if you change the # of subplots
   # figsize(12,5) for multi-plot... 
   # figsize(4,5) for one plot
   fig, ax = plt.subplots(2, 3, sharex='col', sharey='row', figsize=(12,5))

   # Score plot, no longer being used
   #ax[0].plot(score)
   #ax[0].set_ylabel("Score", fontsize=10)
   
   # Custom y-axis, no longer being used
   #ax[0][0].set_ylim([0, 40])   
   #ax[1][0].set_ylim([0, 100])   

   # Moving average plot
   # X O O
   # O O O
   ax[0][0].plot(movingAverageList0)
   ax[0][0].set_title('EPSILON: 0.0')
   ax[0][0].set_ylabel("Score\n(%s Game MA)" % (str(windowSize)), fontsize=10)

   # Record plot
   # O O O
   # X O O
   ax[1][0].plot(rec_0)
   ax[1][0].set_ylabel("Record", fontsize=10)
   ax[1][0].set_xlabel("Game Number", fontsize=10)

   # Moving average plot
   # O X O
   # O O O
   ax[0][1].plot(movingAverageList1)
   ax[0][1].set_title('EPSILON: 0.5')

   # Record plot
   # O O O
   # O X O
   ax[1][1].plot(rec_1)
   ax[1][1].set_xlabel("Game Number", fontsize=10)

   # Moving average plot
   # O O X
   # O O O
   ax[0][2].plot(movingAverageList2)
   ax[0][2].set_title('EPSILON: 1.0')

   # Record plot
   # O O O
   # O O X
   ax[1][2].plot(rec_2)
   ax[1][2].set_xlabel("Game Number", fontsize=10)

   # Moving average plot
   # O O O X
   # O O O O
   #ax[0][3].plot(movingAverageList3)
   #ax[0][3].set_title('C: -30, L: -30, S: 30')

   # Record plot
   # O O O O
   # O O O X
   #ax[1][3].plot(rec_3)
   #ax[1][3].set_xlabel("Game Number", fontsize=10)

   plt.show()
   
   fig.savefig('test.png', bbox_inches="tight")
