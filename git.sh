

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
alias gc='git clone'

alias gs='git stash'
alias gp='git stash pop'
# add tag
alias tag='git tag -a'
# alias grm='git rebase master'
# alias gitc='git branch --merged master | grep -v 'master$' | xargs git branch -d'
alias gph='git push'
#eval "$(hub alias -s)"

alias gtag='git push origin --tags'
alias gl='git log -S'
alias gsc='git submodule foreach "git checkout master; git pull"'

# You can also supply a path to only search commits that affected that path.
# Here, we find out who added the line "verify :method => [:put, :post], :only => [:create]" to UsersController
# git log -p -G'verify.*put.*create' app/controllers/users_controller.rb
# Find out who added the line 'makeMultiscoreDataRDDWithJoinedRDD'

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

gn() {
	git checkout -t origin/master -b $1
	git pull --rebase
}
