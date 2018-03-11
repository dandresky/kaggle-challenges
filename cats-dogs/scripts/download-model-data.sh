# Download model data from EC2 
# 	download-model-data.sh <public DNS>
scp -i ~/.ssh/admin_aws_keypair.pem ubuntu@$1:~/kaggle-data/cats-dogs/cnn_model.h5 ~/Documents/kaggle-data/cats-dogs/cnn_model.h5
scp -i ~/.ssh/admin_aws_keypair.pem ubuntu@$1:~/kaggle-data/cats-dogs/cnn_model.pkl ~/Documents/kaggle-data/cats-dogs/cnn_model.pkl
scp -i ~/.ssh/admin_aws_keypair.pem ubuntu@$1:~/kaggle-data/cats-dogs/cnn_model_predictions.npy ~/Documents/kaggle-data/cats-dogs/cnn_model_predictions.npy
