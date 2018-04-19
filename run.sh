# -*- coding: utf-8 -*-

wget https://github.com/odashi/small_parallel_enja/archive/master.zip
unzip master.zip

python create_wordidmap_file.py small_parallel_enja-master/train.en
mv id_sentences.txt id_sentences_en.txt
mv word_id_map.txt word_id_map_en.txt

python create_wordidmap_file.py small_parallel_enja-master/train.ja
mv id_sentences.txt id_sentences_ja.txt
mv word_id_map.txt word_id_map_ja.txt

python train.py id_sentences_en.txt id_sentences_ja.txt -b 100 -e 50 -g 0
python translate.py result/enc-dec_transmodel.hdf5 word_id_map_en.txt word_id_map_ja.txt small_parallel_enja-master/test.en -g 0

