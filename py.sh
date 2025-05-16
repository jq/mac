# for local error
#export LC_ALL=en_US.UTF-8  
#export LANG=en_US.UTF-8
# if you have two versions of spark comment out SPARK_HOME
#export SPARK_HOME=/usr/local/Cellar/apache-spark/2.3.0/libexec

#open mpi
#export GOOGLE_APPLICATION_CREDENTIALS=/Users/jqian2/src/infra/go_test_terraform/service-account-key.json
export CLOUDSDK_PYTHON=/usr/local/Caskroom/miniconda/base/envs/d1/bin/python
export DYLD_LIBRARY_PATH=/usr/local/opt/open-mpi/lib:$DYLD_LIBRARY_PATH

export DEFAULT_INSTANCE_NAME="jqian"
export DEFAULT_INSTANCE_ZONE="us-west1-b"
export FULL_INSTANCE="jqian.snap-ads-debug.snapint"

alias python3=$(which python)
# nv
# nvcc --version  gcc --version
# docker ps -a 
# docker images
# docker start  之后 用 docker exec -it 51501fd0c88c bash
# docker run --privileged --gpus all -it -v $(pwd):/workspace/hkv gcr.io/snap-bento-training/recsys-examples:latest
# docker run --privileged --gpus all -it -v $(pwd):/workspace/hkv nvcr.io/nvidia/pytorch:24.07-py3 
# docker run --privileged --gpus all -it -v $(pwd):/workspace/tfra nvcr.io/nvidia/tensorflow:24.02-tf2-py3
# docker run --privileged --gpus all -it -v $(pwd):/workspace/tfra nvcr.io/nvidia/tensorflow:24.07-tf2-py3
# need pip install tensorflow-datasets
# docker run --privileged --gpus all -it -v $(pwd):/workspace/tfra tfra/nosla-cuda12.3-cudnn8.9-ubuntu20.04-manylinux2014-python3.9
# TFRA:  use local export PYTHONPATH="/Users/jqian2/src/recommenders-addons/:$PYTHONPATH"
# export PYTHONPATH="${PYTHONPATH}:/workspace/tfra/" 
# /usr/local/Caskroom/miniconda/base/envs/d1/bin/tensorboard --logdir=gs://zliu6/jq/logs/p1
# torch
# docker run --gpus all -it -v $(pwd):/workspace/torch nvcr.io/nvidia/pytorch:24.01-py3

#docker run --privileged --gpus all -it -v $(pwd):workspace nvcr.io/nvidia/tensorflow:22.09-tf2-py3
# docker run --gpus all -it -v $(pwd):/workspace/hkv nvcr.io/nvidia/tensorflow:22.09-tf2-py3
# docker run --gpus all -it -v $(pwd):/workspace/example nvcr.io/nvidia/tensorflow:24.03-tf2-py3

# docker run --gpus all -it -v $(pwd):/workspace/e nvcr.io/nvidia/merlin/merlin-hugectr:23.12

# docker run --gpus all -it -v $(pwd):/workspace/e nvcr.io/nvidia/merlin/merlin-hugectr:nightly

# docker run --gpus all -it -v $(pwd):/workspace/e nvcr.io/nvidia/merlin/merlin-tensorflow:nightly
# docker commit f4b7af85f6c8 hkv3
# docker tag hkv3 gcr.io/snap-bento-training/recsys-examples:latest
# docker push gcr.io/snap-bento-training/recsys-examples:latest
# docker tag recsys-examples:latest gcr.io/snap-bento-training/recsys-examples:latest
# docker tag recsys-examples:latest gcr.io/snap-bento-training/recsys-examples:latest

#  zip -r combined.zip export_dir model_dir
# gsutil
# tfra

alias k='kubectl'
alias tfrad='docker run --privileged --gpus all -it -v $(pwd):$(pwd) tfra/dev_container:latest-python3.9'
alias dk='docker stop'

alias cks="find . -name '*.py' -print0 | xargs -0 yapf --style=./.yapf -ir"

# mac python
alias cle='conda env list'
alias ce='conda activate torch-gpu'
#http://docs.python-guide.org/en/latest/dev/virtualenvs/
alias p39='conda activate python39'
alias d1='ssh jqian2@$FULL_INSTANCE'
alias pynv='conda activate torch-gpu'
alias gclg='gcloud auth application-default login'
alias p3='pip3 install'
alias pv='virtualenv env'
alias plib='pip freeze > requirements.txt'
alias d='gcloud compute instances stop $DEFAULT_INSTANCE_NAME --zone $DEFAULT_INSTANCE_ZONE'
alias jqstart='gcloud compute instances start $DEFAULT_INSTANCE_NAME --zone $DEFAULT_INSTANCE_ZONE'
# gcloud config set project snap-ads-debug
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

# del all python pyc cache, used when you move python root folder
alias pydc="find . -name '*.pyc' -delete"
alias pyt='pytest -s --genuine-v2-mode'
# require by http://click.pocoo.org/5/python3/
# comment out due to python unicode error
#export LC_ALL=C.UTF-8
#export LANG=C.UTF-8

dl() {
  curl -L -H 'Authorization: token $1' -o filename.zip $2
}

de() {
  docker exec -it $1 bash
}

gccp() {
    local instance_name=${3:-$DEFAULT_INSTANCE_NAME}  # Use global default instance name
    local zone=${4:-$DEFAULT_INSTANCE_ZONE}            # Use global default instance zone
    local src_dir=$1                                  # Source directory on VM
    local dest_dir=$2                                 # Destination directory on local machine

    if [ -z "$src_dir" ] || [ -z "$dest_dir" ]; then
        echo "Usage: gcloud_scp_copy [instance_name] [zone] <src_dir> <dest_dir>"
        echo "Default instance name: $instance_name"
        echo "Default zone: $zone"
        return 1
    fi

    gcloud compute scp --recurse "$instance_name:$src_dir" "$dest_dir" --zone "$zone"
}


cpgc() {
  scp -i ~/.ssh/id_ed25519 "$1" jqian2@jqian.snap-ads-debug.snapint:"$2"
}


w1() {
	gclg
	jqstart
	scr
  d1
}

# gb() {
# git snap pull origin "$1"
# git checkout FETCH_HEAD
# git switch -c "$1"
# } 

pyv(){
  pip show $1 | grep Version 
}

# require by anaconda
export PIP_REQUIRE_VIRTUALENV=false
export PATH="/Users/jqian2/go/bin:$PATH"

# added by Anaconda3 4.4.0 installer
# export PATH="/anaconda/bin:$PATH"
# $2 use -s to print log to console
tp() {
  CLAY_CONFIG=config/test.yaml py.test tests/$1 $2
}


# $2 -k for perticular method ""
# python -m pytest -s tests/pyspark_xgboost/test_udf.py  -k test_udf
ptest() {
	python -m pytest -s tests/$1 $2 $3
}

