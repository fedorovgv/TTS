# ASSUME TTS FOLDER

# add fast speech repo
# echo "\n========= git clone fast speech =========\n"
# git clone https://github.com/xcmyz/FastSpeech.git tmp
# mv tmp/text .
# mv tmp/audio .
# mv tmp/waveglow/* waveglow/
# mv tmp/utils.py .
# mv tmp/glow.py .
# rm -rf tmp

echo "\n========= install packages via pip =========\n"
pip install -r requirements.txt

echo "\n========= download ljspeech =========\n"
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2 -o /dev/null
mkdir -p datasets
tar -xvf LJSpeech-1.1.tar.bz2 >> /dev/null
rm LJSpeech-1.1.tar.bz2
mv LJSpeech-1.1 datasets/LJSpeech

# yandex disk loader 
chmod +x ya.py

echo "\n========= download mel specs =========\n"
./ya.py https://disk.yandex.ru/d/RItaW_Vlut5qdQ . 1> /dev/null 2> /dev/null
tar -xvf mel.tar.gz >> /dev/null
mv mels datasets/mels
rm mel.tar.gz

echo "\n========= download texts =========\n"
./ya.py https://disk.yandex.ru/d/w-CfZegZVOzi_w . 1> /dev/null 2> /dev/null
mv train.txt datasets/train.txt

echo "\n========= download alignments =========\n"
./ya.py https://disk.yandex.ru/d/ceLV3f3BNJa6qw . 1> /dev/null 2> /dev/null
unzip alignments.zip >> /dev/null
mv alignments datasets/alignments
rm alignments.zip

echo "\n========= download glow =========\n"
./ya.py https://disk.yandex.ru/d/Lm-ma0wnQAcJNQ . 1> /dev/null 2> /dev/null
mkdir -p waveglow/pretrained_model/
mv waveglow_256channels_ljs_v2.pt waveglow/pretrained_model/waveglow_256channels.pt

echo "\n========= prepare and normalize pith and energy =========\n"
python features.py

echo "\n========= download checkpoint =========\n"
./ya.py https://disk.yandex.ru/d/J8OPMC2-F1L2Uw . 1> /dev/null 2> /dev/null
mkdir -p model_ckpt/test
mv 2022-11-28_12-14.pth model_ckpt/test/2022-11-28_12-14.pth
