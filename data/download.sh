if [ ! -f glove.840B.300d.txt ]; then
  wget http://nlp.stanford.edu/data/glove.840B.300d.zip -O glove.840B.300d.zip
  unzip glove.840B.300d.zip
fi

wget https://www.csie.ntu.edu.tw/~b08902011/raw.csv
wget https://www.csie.ntu.edu.tw/~b08902011/test.csv
