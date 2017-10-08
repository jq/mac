#init gradle java project
alias gij='gradle init --type java-library'

#java build
alias gru='./gradlew run'

alias gai='gradlew assemble idea'

alias gbu='gradlew bintrayUpload'
           
alias gclean='gradlew clean'
#zsh intercept **, so use \*
gtt () {
    if [ $# -eq 0 ]
    then
        gradlew test --rerun-tasks
    else
        gradlew test --tests \*$1\* test --info
    fi
}
function gtd() {
 gradlew test --tests $1 --debug
}
