# setup shell dev env
mkdir etc
cd etc
git clone git@github.com:jq/mac.git

DEVHOME=~/etc/mac

cd mac
# if you use default settings, do following, otherwise, use your own local settings
#ln -s $DEVHOME/local.template.sh ~/.local.sh 

ln -s $DEVHOME/zshrc ~/.zshrc
ln -s $DEVHOME/bash_profile .bash_profile
brew install tmux
ln -s $DEVHOME/tmux.conf .tmux.conf


