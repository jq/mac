# put all your local settings here
SHELLHOME=~/etc/mac
export FAVOR_SILVER_MACHINE=5
export FAVOR_SPARK_MACHINE=2

. $SHELLHOME/vagrant.sh
. $SHELLHOME/linux.sh
. $SHELLHOME/git.sh

ff() {
	find . -name "$1"
}

autossh -M 20000 -NfD 8527 $USER@gw5.silver.musta.ch

#super faster access
alias sv='ssh gw1.silver.musta.ch'
alias sp='ssh gw1.spark.musta.ch'
alias production='ssh rc1.musta.ch'

alias gi='gradlew install'

 # setup autossh refer to https://airbnb.hackpad.com/Using-an-SSH-tunnel-UbxSgIcGVIU
