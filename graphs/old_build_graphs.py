import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np 
import math

def build_Score_Graph(title):
   # Change size of plot (in inches)
   figure(figsize=(11, 7), dpi=100)

   plt.plot(game_num, score)

   #plt.ylim([0,5])

   pfit = np.polyfit(game_num, score, 1)
   trend_line_model = np.poly1d(pfit)
   plt.plot(trend_line_model(game_num), label="Trend line")

   plt.xlabel('Game Number', fontsize=12)
   plt.ylabel('Score', fontsize=12)

   plt.title(title, fontsize=20)
   plt.legend()
   plt.show()

def build_Cumulative_Score_Graph(title):
   # Change size of plot (in inches)
   figure(figsize=(11, 7), dpi=100)

   plt.plot(game_num, cumulativeScore)

   plt.xlabel('Game Number', fontsize=12)
   plt.ylabel('Cumulative Score', fontsize=12)

   plt.title(title, fontsize=20)
   plt.show()

def build_Record_Graph(title):
   # Change size of plot (in inches)
   figure(figsize=(11, 7), dpi=100)

   plt.plot(game_num, record)
   #plt.ylim([0,5])
   plt.xlabel('Game Number', fontsize=12)
   plt.ylabel('Record Score', fontsize=12)

   plt.title(title, fontsize=20)
   plt.show()

if __name__ == "__main__":
   game_num = []
   score = []
   record = []

   file = open('test.txt')
   for row in file:
      row = row.split(' ')
      game_num.append(int(row[0]))
      score.append(int(row[1]))
      record.append(int(row[2]))

   cumulativeScore = np.cumsum(score)
      
   # For some reason you have to exit out of one graph to see the next one
   build_Score_Graph('Random Policy (no training)')
   build_Cumulative_Score_Graph('Random Policy (no training)')
   build_Record_Graph('Random Policy (no training)')


