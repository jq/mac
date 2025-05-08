# Set PATH, MANPATH, etc., for Homebrew.
eval "$(/opt/homebrew/bin/brew shellenv)"
eval "$(_SNAPC_COMPLETE=source_zsh snapc)"
export OPENAI_API_KEY=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


#brew analytics off 2>&1 >/dev/null
#export HOMEBREW_BREWFILE=~/etc/ubrewfile/Brewfile

export PATH=.:~/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/X11/bin:$PATH


if [ -f $(brew --prefix)/etc/brew-wrap ];then
  source $(brew --prefix)/etc/brew-wrap
fi

# echo $PATH to print out
# old setting for golden key export SSH_AUTH_SOCK=$TMPDIR/ssh-agent-$USER.sock
export JAVA7_HOME=/Library/Java/JavaVirtualMachines/jdk1.7.0_75.jdk/Contents/Home
export JAVA8_HOME=/Library/Java/JavaVirtualMachines/jdk1.8.0_60.jdk/Contents/Home
export JAVA_HOME=/Library/Java/Home
# for go test for uber_timeseries_go
export LIBRARY_PATH=/usr/local/lib/gcc/7
export EDITOR='subl -w'

export GOPATH=$HOME/go
export PATH="$GOPATH/bin:$PATH"

# for fix python ascii' codec can't encode character
export LC_ALL=en_US.UTF-8  
export LANG=en_US.UTF-8

# mich settings
#ulimit -n 20000

# for xgboost
export CC=gcc-5
export CXX=g++-5

#. ~/.local.sh

# regen ssh key
alias h='ussh'


alias sl='mvn scalastyle:check'

alias bi='brew install '

# test
function mt(){
  mvn -DwildcardSuites=\*$1\* test
}

# general copy, default to spark
#copy from gateway
cpfg() {
  rsync  -avzh $USER@$1:~/$2 $3
}

cptg() {
  rsync  -avz $2 $USER@$1:~/$3
}

# copy to sjc
cps() {
	cptg hadoopgw01-sjc1 $1 $2
}

# copy from sjc to local
cpsl() {
  rsync  -avzh $USER@hadoopgw01-sjc1:~/$1 $2
}

source ~/etc/mac/.mvn.sh 
source ~/etc/mac/git.sh 
source ~/etc/mac/linux.sh 
source ~/etc/mac/.go.sh 
#source ~/etc/mac/algo.sh 
source ~/etc/mac/py.sh 
source ~/etc/mac/.gradle.sh
source ~/etc/mac/kube.sh
source ~/etc/mac/ant.sh
source ~/etc/mac/ak.sh
source ~/etc/mac/mac.sh

# install z if hits error
source /usr/local/Cellar/z/1.9/etc/profile.d/z.sh


alias pg='ps ax | grep '
#alias nport='netstat -ap tcp | grep -i "listen"'

alias rb='. ~/.zshrc'
alias sn='snapaccess credentials refresh'
alias cv='conda activate'
# run this before connect to spark UI 
#connect-proxy

# ignore autoenv for now
#source /usr/local/opt/autoenv/activate.sh
