#kube 
alias ms='minikube start'
#get url
alias mu='minikube service tf-hub-lb --url'

alias mstop='minikube stop'

alias msvc='kubectl get svc'

# List all pods in all namespaces
alias kpn='kubectl get pods --all-namespaces'

alias k='kubectl'

alias argoui='kubectl -n kubemaker port-forward deployment/argo-ui 8001:8001'

 # switch to apple compiler
CC="clang" && CXX="clang++"

# 
alias depen='dep ensure --vendor-only'
# minikube dashboard
# kubectl create ns
# kubectl get ns
#kubectl create -f manifests/k8s/argo-controller.yaml -n kubemaker
#kubectl get pods -n kubemaker
# kubectl get deployments
#kubectl run kubernetes-bootcamp --image=gcr.io/google-samples/kubernetes-bootcamp:v1 --port=8080
# kubectl proxy // setup proxy between pods and outside the cluster
# get pod name
#export POD_NAME=$(kubectl get pods -o go-template --template '{{range .items}}{{.metadata.name}}{{"\n"}}{{end}}')
# A Kubernetes Service is an abstraction layer which defines a logical set of Pods and enables external traffic exposure, load balancing and service discovery for those Pods.


export PATH=$PATH:/usr/local/kubebuilder/bin
