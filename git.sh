# Remove a file from a Git repository without deleting it from the local filesystem
alias gk='git rm --cached '

# put last commit into working tree keep the changes in your working tree but not on the index
# need git add .
alias grh='git reset HEAD~'

# undo the last commit, but the file changes will stay in your working tree. 
# Also the changes will stay on your index, dont' need git add .
alias grt='git reset --soft HEAD~1'

# clean current working tree
alias grhd='git reset --hard'

alias gb='git branch'
alias gr='git pull --rebase'
alias gcl='git clone'

alias gc='git checkout'
alias gs='git status'
alias gss='git stash'
alias gsp='git stash pop'
# add tag
alias tag='git tag -a'
# alias grm='git rebase master'
# alias gitc='git branch --merged master | grep -v 'master$' | xargs git branch -d'
alias gp='git push origin HEAD --force'
#eval "$(hub alias -s)"
# git rebase -i HEAD~2
#git remove file from repo but not locally
alias grc='git rm --cached -r '
alias gtag='git push origin --tags'
alias gl='git log -S'
alias gsc='git submodule foreach "git checkout master; git pull"'
alias gra='git remote add upstream '

# git checkout unmerged file
gcu() {
	git reset ${1}
	git checkout ${1}
}

# syn with upstream
gsu() { 
git fetch upstream
git checkout master
git merge upstream/master
}

# You can also supply a path to only search commits that affected that path.
# Here, we find out who added the line "verify :method => [:put, :post], :only => [:create]" to UsersController
# git log -p -G'verify.*put.*create' app/controllers/users_controller.rb
# Find out who added the line 'makeMultiscoreDataRDDWithJoinedRDD'

# `git add -p` to add little changes one at a time.
gcm() {
	./cicd/type_analysis.py
	git add .
	git commit -m ${1:-'"update"'}
    git push --no-verify -f origin HEAD
}

# git add all and push
ga(){
	./cicd/type_analysis.py
	git add .
	git commit --amend --no-edit
	git push --no-verify -f origin HEAD
}

gn() {
	git checkout -t origin/${2:-master} -b $1
	git pull --rebase
}

# link with upstream current master, so that git pull --rebase syn with current master
# and git push origin $1 to update to your own git repo
gnu() {
	git checkout -t upstream/${2:-master} -b $1
	git pull --rebase
}

gpo() {
	git push origin ${1:-jq}
}

