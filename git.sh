

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
alias grm='git rebase master'
alias gitc='git branch --merged master | grep -v 'master$' | xargs git branch -d'
alias gp='git push'
#eval "$(hub alias -s)"


function tag {
  git tag -a "$1"
}

alias gtag='git push origin --tags'

# Aliases for GitHub features (via hub, to which git has been aliased)
# git rev-parse --abbrev-ref HEAD is current checkout branch
function gpra() {
  git pull-request -b airbnb:${1:-master} -h airbnb:$(git rev-parse --abbrev-ref HEAD)
}

# You can also supply a path to only search commits that affected that path.
# Here, we find out who added the line "verify :method => [:put, :post], :only => [:create]" to UsersController
# git log -p -G'verify.*put.*create' app/controllers/users_controller.rb
# Find out who added the line 'makeMultiscoreDataRDDWithJoinedRDD'
gal() {
	git log -S %1
}

# `git add -p` to add little changes one at a time.
gcm() {
	git add .
	git commit -m ${1:-'"update"'}
	git push
}

# git add all and push
ga(){
	git add .
	git commit --amend --no-edit
	git push --force
}

gd() {
	git branch -D $1
}
gc(){
	git clone $1
	
}
alias gs='git stash'
alias gsp='git stash pop'

gn() {
	git checkout -t origin/master -b $1
	git pull --rebase
}
function pr_for_sha {
  git log --merges --ancestry-path --oneline $1..origin/master | grep 'pull request' | tail -n1 | awk '{print $5}' | cut -c2- | xargs -I % open https://git.musta.ch/airbnb/$(basename $(pwd -P))/pull/%
}
