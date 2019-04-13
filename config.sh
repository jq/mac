#fish config

function gcm 
	git add .
	git commit -m $argv
	git push
end
