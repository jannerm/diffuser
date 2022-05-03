DOWNLOAD_DIR=bin
mkdir $DOWNLOAD_DIR
wget https://aka.ms/downloadazcopy-v10-linux -O $DOWNLOAD_DIR/download.tar.gz
tar -xvf $DOWNLOAD_DIR/download.tar.gz --one-top-level=$DOWNLOAD_DIR
mv $DOWNLOAD_DIR/*/azcopy $DOWNLOAD_DIR
rm $DOWNLOAD_DIR/download.tar.gz
rm -r $DOWNLOAD_DIR/azcopy_linux*
