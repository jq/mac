function tunnel {
ssh -N -f -L localhost:${1}:localhost:${1} $USER@gw${2}.silver.musta.ch
}
