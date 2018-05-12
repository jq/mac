function tunnel {
ssh -N -f -L localhost:${1}:localhost:${1} $USER@gw${2}.silver.musta.ch
}

# Aliases for GitHub features (via hub, to which git has been aliased)
# git rev-parse --abbrev-ref HEAD is current checkout branch
function gpra() {
  git pull-request -b airbnb:${1:-master} -h airbnb:$(git rev-parse --abbrev-ref HEAD)
}



function pr_for_sha {
  git log --merges --ancestry-path --oneline $1..origin/master | grep 'pull request' | tail -n1 | awk '{print $5}' | cut -c2- | xargs -I % open https://git.musta.ch/airbnb/$(basename $(pwd -P))/pull/%
}

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
