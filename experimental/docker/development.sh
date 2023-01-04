docker build . -t adan-dev&&\
docker run -v $ADAN_DIR:/adan -p 3000:3000 -it adan-dev bash