brew analytics off 2>&1 >/dev/null
export HOMEBREW_BREWFILE=~/etc/ubrewfile/Brewfile

export PATH=/usr/local/opt/python/libexec/bin:.:~/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/X11/bin:$PATH


if [ -f $(brew --prefix)/etc/brew-wrap ];then
  source $(brew --prefix)/etc/brew-wrap
fi

# echo $PATH to print out
# old setting for golden key export SSH_AUTH_SOCK=$TMPDIR/ssh-agent-$USER.sock
export JAVA7_HOME=/Library/Java/JavaVirtualMachines/jdk1.7.0_75.jdk/Contents/Home
export JAVA8_HOME=/Library/Java/JavaVirtualMachines/jdk1.8.0_60.jdk/Contents/Home
export JAVA_HOME=/Library/Java/Home
#export SCALA_HOME=/usr/local/bin/scala
# for go test for uber_timeseries_go
export LIBRARY_PATH=/usr/local/lib/gcc/7
export EDITOR='subl -w'
export UBER_HOME=~/uber

export GOPATH=$HOME/gocode
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
alias s='subl'

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
source ~/etc/mac/algo.sh 
source ~/etc/mac/py.sh 
source ~/etc/mac/.gradle.sh
source ~/etc/mac/kube.sh
source ~/etc/mac/ant.sh
source ~/etc/mac/ak.sh

# install z if hits error
source /usr/local/Cellar/z/1.9/etc/profile.d/z.sh

# added by Anaconda3 5.0.0 installer
# export PATH="/Users/qian/anaconda3/bin:$PATH"


ƒalias pg='ps ax | grep '
#alias nport='netstat -ap tcp | grep -i "listen"'


# Add this to your .bash_profile to enable command-line setup of the athena connection proxy
# Created by @sdh 6/9/2016

connect-proxy() {
  bastion_connections=$(ps aux | grep -c 'ssh -fCND 8001 bastion01-sjc1.prod.uber.com')
  if [[ $bastion_connections -lt 2 ]] ; then 
      ssh -fCND 8001 bastion01-sjc1.prod.uber.com
  else
      echo "Already connected to bastion01-sjc1"
  fi
}

connect-athena() {
  # Start tunnel
  connect-proxy

  # Configure socks proxy
  socksproxy_config=$(networksetup -getsocksfirewallproxy "Wi-Fi")
  proxy_enabled=$(echo $socksproxy_config | grep -c "Enabled: yes")
  proxy_configured=$(echo $socksproxy_config | grep -c "Server: localhost")
  if [[ $proxy_configured -eq 0 ]] ; then
    echo "Need to configure firewall server"
    networksetup -setsocksfirewallproxy "Wi-Fi" localhost 8001
  else
    echo "Firewall server already configured"
  fi
  
  bypass_domains=$(networksetup -getproxybypassdomains "Wi-Fi")
  bypass_domains_configured=$(echo $bypass_domains | grep -E "\*\.local" | grep -cE "169\.254/16")

  if [[ $bypass_domains_configured -eq 0 ]] ; then 
    # Set bypass domains
    echo "Need to configure bypass domains"
    networksetup -setproxybypassdomains "Wi-Fi" *.local 169.254/16
  else
    echo "Bypass domains already configured"
  fi

  if [[ $proxy_enabled -eq 0 ]] ; then 
    # Enable proxy
    echo "Need to enable proxy"
    networksetup -setsocksfirewallproxystate "Wi-Fi" on
  else
    echo "Proxy already enabled"
  fi
}

disconnect-athena() {
  # Disable proxy
  networksetup -setsocksfirewallproxystate "Wi-Fi" off
  # Kill running SSH tunnel, if one exists
  pkill -f "ssh -fCND 8001 bastion01-sjc1.prod.uber.com"
}

alias rb='. ~/.zshrc'
# run this before connect to spark UI 
#connect-proxy

# ignore autoenv for now
#source /usr/local/opt/autoenv/activate.sh
