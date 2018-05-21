# for local error
#export LC_ALL=en_US.UTF-8  
#export LANG=en_US.UTF-8
# if you have two versions of spark comment out SPARK_HOME
#export SPARK_HOME=/usr/local/Cellar/apache-spark/2.3.0/libexec

#http://docs.python-guide.org/en/latest/dev/virtualenvs/
alias p3='pip3 install'
alias pv='virtualenv env'
alias plib='pip freeze > requirements.txt'
#install your project in development mode
alias pins='pip install -e .'
# python3 create env
alias p3e='python3.6 -m venv env'
alias pa='source env/bin/activate'
alias da='deactivate'
alias sac='≈'
alias pir='pip3 install -r requirements.txt'
# for rest project to instal pkg
alias mbs='make bootstrap'
# for algorithmic so that algo serve can run, set to export LANGUAGE_VERSION=python3 in case
# algo serve enable algo runlocal -D run much faster

# test janus
alias tpj='tp handlers/thrift/test_hive_schema_handler.py::test_get_hive_table_schema_response -s'
alias tunnel='./scripts/dev/dev_tunnels.sh'
# start rest in laptop need to open tunnel before it, and before open tunnel, you need to connect vpn
alias rest='make lserve'
alias t='make test'
alias ve='virtualenv'
export LANGUAGE_VERSION=python2

# require by http://click.pocoo.org/5/python3/
# comment out due to python unicode error
#export LC_ALL=C.UTF-8
#export LANG=C.UTF-8

pyv(){
  pip show $1 | grep Version 
}
 #for python
#eval "$(pyenv init -)"
#eval "$(pyenv virtualenv-init -)"

# require by anaconda
export PIP_REQUIRE_VIRTUALENV=false

# added by Anaconda3 4.4.0 installer
# export PATH="/anaconda/bin:$PATH"
# $2 use -s to print log to console
tp() {
  CLAY_CONFIG=config/test.yaml py.test tests/$1 $2
}

# create python auto env
# I think the whoa should be change to activate 
pyenv () {
	echo "echo 'whoa'" > $1/.env
	cd $1
}

