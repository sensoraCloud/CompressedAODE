#!/bin/bash

write_script()
{
current_dir=$1
echo $current_dir
for file in $(ls $current_dir|grep csv);
do 
current_script=tmp_batch_aode_$(basename $file .csv).m;
current_script2=batch_aode_$(basename $file .csv).m;
#batch_template=batch_aode_unsup.m


#if [ "$1" == "/homeb/corani/dataset/mdl" ]
#  then
#    echo "mdl found"
current_script2=batch_aode_$(basename $file .csv).m;
batch_template=batch_aode.m
#fi


cat $batch_template > $current_script;
sed -e "s/marker/$(basename $file .csv)/g" $current_script > $current_script2;
sed -e "s/batch_aode/batch_aode_$(basename $file .csv)/g" $current_script2 > $current_script;
mv $current_script $current_script2;
#rm -f $current_script;
done
}

#dir1=/homeb/corani/dataset/unsup
dir2=/homeb/corani/dataset
#write_script $dir1
write_script $dir2

