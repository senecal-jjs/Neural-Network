set xlabel 'x_1'
set ylabel 'x_2'
set zlabel 'output'
set dgrid 20 20
set datafile separator ","
splot filename using 1:2:3 with lines
pause -1
