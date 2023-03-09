
image=05c8
docker stop client_dev; docker rm client_dev;
docker run --name=client_dev -d -p 18888:8888 -v $PWD/workspace:/workspace $image jupyter-lab

docker logs -f client_dev

