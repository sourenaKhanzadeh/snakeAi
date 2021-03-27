# Contribution Guideline 
***
## Writing new levels
Go to <b>main.py</b> to and add levels in Window enum class
<blockquote>
write as the pattern written <br>
W1, W2, W3 ..., Wn<br>
then wirte the values as tupple <br>
W1 = (n, m, k, l)<br>
n = row tiles<br>
m = column tiles<br>
k = number of processes as rows<br>
l = number of processes as columns<br>
For instance:<br>
W1 = (10, 10, 2, 3)<br>
this will create a 10 by 10 windows  6 times 2 rows and 3 columns
</blockquote>

## Writing parameters for each levels
Go to <b>par_lev.json</b> to write parameters and levels
<blockquote>
wirte levels and parameters for each levels<br>
following the first template patterns, in file par_lev.json<br>
you will find <i>W1</i> where it is the first world, use <i>W1</i> <br>
first dictionary in the list as a template for the other parameters <br>
for window processing
</blockquote>

## Readme, License, etc.. 
please contribute on the appropiate branch, ex. Readme changes should be pushed to <b>readme</b> branch
## Generall contribution
please contribute on the <b>master</b> branch
