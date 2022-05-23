if keyctl show 2>&1 | grep -q "workaroundSession";
then
	echo "Already logged in"
else
	echo "Logging in with tenant id:" ${AZURE_TENANT_ID}
	keyctl session workaroundSession
	./bin/azcopy login --tenant-id=$AZURE_TENANT_ID
fi

export LOCAL_LOGBASE=logs/pretrained/
export LOGBASE=logs/pretrained/
export AZURE_STORAGE_CONTAINER="diffuser-pretrained"

## AZURE_STORAGE_CONNECTION_STRING has a substring formatted lik:
	## AccountName=${STORAGE_ACCOUNT};AccountKey= ...
export AZURE_STORAGE_ACCOUNT=`(echo $AZURE_STORAGE_CONNECTION_STRING | grep -o -P '(?<=AccountName=).*(?=;AccountKey)')`

echo "Syncing from" ${AZURE_STORAGE_ACCOUNT}"/"${AZURE_STORAGE_CONTAINER}"/"${LOGBASE}

mkdir -p ${LOCAL_LOGBASE}

./bin/azcopy sync https://${AZURE_STORAGE_ACCOUNT}.blob.core.windows.net/${AZURE_STORAGE_CONTAINER}/${LOGBASE} ${LOCAL_LOGBASE} --recursive
