set xlabel 'input 1'
set ylabel 'input 2'
set zlabel 'output'
set dgrid 20 20
set datafile separator ","
splot filename using 1:2:3 with lines
pause -1
