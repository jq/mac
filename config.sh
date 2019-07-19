#fish config
source ~/miniconda3/etc/fish/conf.d/conda.fish

# use omf reload to reload

function gcm 
	git add .
	git commit -m $argv
	git push 
end

function gcmu 
	git add .
	git commit -m $argv[1]
	git push origin $argv[2]
end

# for matching with develop branch
function gnd
  git checkout -t upstream/develop -b $argv[1]
end  

function gn
  git checkout -t origin/master -b $argv[1]
end 

# for xgboost
function gnx
  git checkout -t origin/ant_master -b $argv[1]
end   

function gr
  git pull --rebase
end

#python local
function g2l
	pyenv local 2.7.14
end

function g3l
	pyenv local 3.7.3
end


# -s show print log
# python -m pytest -s tests/pyspark_xgboost/test_udf.py  -k test_udf
function ptest
	python -m pytest -s tests/$argv[1] $argv[2] $argv[3]
end


# this conflict with conda: source ~/.config/fish/config.fish


export GOPATH=$HOME/go
export PATH="$GOPATH/bin:$PATH"