# python3 create env
alias p3e='python3.6 -m venv env'
alias p22='source env/bin/activate'
alias sac='≈'
alias pir='pip install -r requirements.txt'
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
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

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

