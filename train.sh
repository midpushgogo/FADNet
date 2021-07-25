# dnn="${dnn:-fadnet}"
# source exp_configs/fadnet.conf

net=fadnet
loss=loss_configs/fadnet_sceneflow.json
outf_model=models/${net}-sf-newatt
logf=logs/${net}-sf-newatt.log

lr=1e-3
devices=0,1
dataset=sceneflow
datapath=data
trainlist=lists/SceneFlow.list
vallist=lists/FlyingThings3D_release_TEST.list
startR=0
startE=0
batchSize=4
maxdisp=-1
model=None 

python -W ignore main.py --cuda --net $net --loss $loss --lr $lr \
               --outf $outf_model --logFile $logf \
               --devices $devices --batch_size $batchSize \
               --datapath $datapath \
               --dataset $dataset --trainlist $trainlist --vallist $vallist \
               --startRound $startR --startEpoch $startE \
               --model $model \
               --maxdisp $maxdisp \
	       --manualSeed 1024 \
	       --attention
