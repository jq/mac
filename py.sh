alias sac='source env/bin/activate'

# for algorithmic so that algo serve can run, set to export LANGUAGE_VERSION=python3 in case
# algo serve enable algo runlocal -D run much faster
export LANGUAGE_VERSION=python2

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