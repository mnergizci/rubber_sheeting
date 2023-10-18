#!/bin/bash

# 2023

m=`echo $1 | cut -d '_' -f1`
s=`echo $1 | cut -d '_' -f2`
pair=$1
outdir='RSLCRS'
mkdir -p $outdir
mkdir -p $outdir/$s
rslcdir=`pwd`/RSLC
mpar=$rslcdir/$m/$m.rslc.par
mslc=$rslcdir/$m/$m.rslc
spar=$rslcdir/$s/$s.rslc.par
sslc=$rslcdir/$s/$s.rslc
mmli=$rslcdir/$m/$m.rslc.mli.par
smli=$rslcdir/$s/$s.rslc.mli.par

###interp for burst overlap
SLC_interp_lt_ScanSAR tab/${s}R_tab $spar tab/${m}R_tab $mpar OFF/$pair/offsets.filtered.lut $mmli $smli OFF/$pair/tracking.off tab/20230210rsR_tab $outdir/$s/$s.rslc $outdir/$s/$s.rslc.par - 6 $outdir/$s/


###normal interp without producing swath par files
#SLC_interp_lt $rslcdir/$s/$s.rslc $rslcdir/$m/$m.rslc.par $rslcdir/$s/$s.rslc.par OFF/$pair/offsets.filtered.lut $mmli $smli - $outdir/$s/$s.rslc $outdir/$s/$s.rslc.par - - 5


echo 'Rubber_sheeting done, interferogram is generating...'
cd $outdir/$s; chmod 777 *

for i in 1 2 3; do
    mv "$s.IW$i.rslc.tops_par" "$s.IW$i.rslc.TOPS_par"
done
cd ../..

echo $pair >ifg.list

if [ ! -d IFG/${m}_${s}.orig ]; then
 cd IFG; mv $m'_'$s $m'_'$s.orig; mkdir $m'_'$s; cd ../
fi

if [ ! -d RSLC/${s}.orig ]; then
 cd RSLC; mv $s $s.orig; ln -s ../$outdir/$s; cd ../
fi

echo '11111111111111'
multi_look $rslcdir/$s/$s.rslc $rslcdir/$s/$s.rslc.par $rslcdir/$s/$s.rslc.mli $rslcdir/$s/$s.rslc.mli.par 20 4
chmod 777 $rslcdir/$s/*

#echo '22222222222222'
#SLC_mosaic_ScanSAR tab/${s}R_tab $rslcdir/$s/$s.rslc $rslcdir/$s/$s.rslc.par 20 4 1 tab/${m}R_tab
#chmod 777 $rslcdir/$s/*
#echo '3333333333333'
#multi_look $rslcdir/$s/$s.rslc $rslcdir/$s/$s.rslc.par $rslcdir/$s/$s.rslc.mli $rslcdir/$s/$s.rslc.mli.par 20 4

mkdir -p log
LiCSAR_03_mk_ifgs.py -d . -i ifg.list -a 4 -r 20

create_geoctiffs_to_pub.sh `pwd` $pair

create_bovl_ifg.sh $pair

#rm RSLC/$s; mv RSLC/$s.orig RSLC/$s



##after interp_lt, need to procude multi_look and create_offset.
#cd $rlscdir/$s; multi_look $s.rslc $s.rslc.par $s.rslc.mli $s.rslc.mli.par 20 4; cd ../..

#create_offset $mpar $spar ${rslcdir}/${s}/${pair}.off 1 20 4 0 
#create_offset /gws/nopw/j04/nceo_geohazards_vol1/projects/COMET/mnergizci/021D/021D_05266_252525/RSLC/20220919/20220919.rslc.par  20230210.rlsc.par 20220919_20230210.off 1 20 4 0



#S1_coreg_overlap tab/${m}R_tab tab/${s}R_tab $pair ${rslcdir}/${s}/${pair}.off ${pair}.refined.off 0.8 0.01 0.8 1 - >>result.txt

#ScanSAR_coreg_overlap.py tab/20230129R_tab tab/20230210R_tab 20230129_20230210 /gws/nopw/j04/nceo_geohazards_vol1/projects/COMET/mnergizci/021D/021D_05266_252525/RSLC/20230210/20230129_20230210.off 20230129_20230210.refined_2.off --cc 0.8 --fraction 0.01 --ph_stdev 0.8 >>resul_2.txt

#ScanSAR_coreg.py tab/${m}R_tab $m tab/${s}R_tab $s tab/${s}_out_R_tab geo/20220919.hgt 20 4 --cc 0.7 --fraction 0.01 --ph_stdev 0.8 --it1 0 --it2 2 --num_ovr 10  >>result_5.txt 

#LiCSAR_03_mk_ifgs.py -d . -i ifg.list -a 4 -r 20
