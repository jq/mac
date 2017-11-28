alias grg='go run main.go'
alias gor='go run '

alias gob='go build && '
# go clone
goc () {
  mkdir -p $GOPATH/src/code.uber.intern/$1
  git clone gitolite@code.uber.internal:$1 $GOPATH/src/code.uber.internal/$1 --recursive
}

alias ssql='brew services start mysql'
alias stsql='brew services stop mysql'

alias goth='make thriftc'
alias goin='go-build/glide install'
alias gobin='make bins'
alias ml='make lint'

alias got='go test -run ' 
alias rtu='./scripts/dev/dev_tunnels.sh'