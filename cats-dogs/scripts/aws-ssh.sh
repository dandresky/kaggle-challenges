# this script takes one argument which is the path of the ecs instance since this changes every time the instance is restarted.
ssh -i ~/.ssh/admin_aws_keypair.pem ubuntu@$1
