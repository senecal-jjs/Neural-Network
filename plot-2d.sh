(head -n 2 $1 && tail -n +3 $1 | sort -k1 -n -t,) > /tmp/sorted.csv
gnuplot -e "filename='/tmp/sorted.csv'" plot-2d.gp
