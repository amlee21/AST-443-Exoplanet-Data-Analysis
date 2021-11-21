#! /bin/bash -u

for file in $(ls -1 corr_Toi_star.*.new) 
do
   sex ${file} -c default_check.se -CATALOG_NAME "check_${file}.cat"
done
