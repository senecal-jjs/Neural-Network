set xlabel 'input'
set ylabel 'output'
set datafile separator ","
plot filename using 1:2 with linespoints
pause -1
