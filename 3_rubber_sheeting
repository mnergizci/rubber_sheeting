#:!/bin/bash

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
multi_look $s.rslc $s.rslc.par $s.rslc.mli $s.rslc.mli.par 20 4
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

mkdir -p log
LiCSAR_03_mk_ifgs.py -d . -i ifg.list -a 4 -r 20
#rm RSLC/$s; mv RSLC/$s.orig RSLC/$s

