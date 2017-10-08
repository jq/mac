function tunnel {
ssh -N -f -L localhost:${1}:localhost:${1} $USER@gw${2}.silver.musta.ch
}

# Aliases for GitHub features (via hub, to which git has been aliased)
# git rev-parse --abbrev-ref HEAD is current checkout branch
function gpra() {
  git pull-request -b airbnb:${1:-master} -h airbnb:$(git rev-parse --abbrev-ref HEAD)
}



function pr_for_sha {
  git log --merges --ancestry-path --oneline $1..origin/master | grep 'pull request' | tail -n1 | awk '{print $5}' | cut -c2- | xargs -I % open https://git.musta.ch/airbnb/$(basename $(pwd -P))/pull/%
}
