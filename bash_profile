brew analytics off 2>&1 >/dev/null

export PATH=/usr/local/opt/python/libexec/bin:/Users/qian/.nvm/v0.10.32/bin:/Users/qian/.rbenv/shims:/Users/qian/bin:/Users/qian/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/X11/bin:/usr/local/munki:/Users/qian/gocode/bin:/Users/qian/gocode/bin

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
export PATH="$PATH:$GOPATH/bin"

# mich settings
#ulimit -n 20000


#. ~/.local.sh
# gradlew
alias bs='gradle shadowJar'
alias b='gradle fatJar'

# regen ssh key
alias h='ussh'
alias s='subl'

alias sl='mvn scalastyle:check'

# test
function mt(){
  mvn -DwildcardSuites=\*$1\* test
}

alias sjc='ssh hadoopgw01-sjc1'
alias ad='ssh adhoc04-sjc1'

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
alias cpm='cps /Users/qian/Uber/michelangelo/core/target/michelangelo_core-0.4-SNAPSHOT.jar bin'
alias cpa='cps /Users/qian/src/aerosolve/demo/income_prediction/build/libs/income_prediction-1.0.0-all.jar bin'

# copy from sjc to local
cpsl() {
  rsync  -avzh $USER@hadoopgw01-sjc1:~/$1 $2
}

#for python
#eval "$(pyenv init -)"
#eval "$(pyenv virtualenv-init -)"

# require by anaconda
export PIP_REQUIRE_VIRTUALENV=false

# added by Anaconda3 4.4.0 installer
export PATH="/anaconda/bin:$PATH"

source ~/etc/mac/git.sh 
source ~/etc/mac/linux.sh 
source ~/etc/mac/.go.sh 

source /usr/local/Cellar/z/1.9/etc/profile.d/z.sh

# added by Anaconda3 5.0.0 installer
export PATH="/Users/qian/anaconda3/bin:$PATH"


alias a='arc diff'
alias pg='ps ax | grep '
#alias nport='netstat -ap tcp | grep -i "listen"'
pyv(){
  pip show $1 | grep Version 
}

alias al='arc land'

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

# $2 use -s to print log to console
tp() {
  CLAY_CONFIG=config/test.yaml py.test tests/$1 $2
}
# test janus
alias tpj='tp handlers/thrift/test_hive_schema_handler.py::test_get_hive_table_schema_response -s'
# start rest in laptop
alias rest='make lserve'
alias t='make test'

source /usr/local/opt/autoenv/activate.sh
