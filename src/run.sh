#!/bin/bash

rm ../logs/*.log

# python main.py -m CNN -b 256 --mode train --hidden 256 --dropout 0.5 --patience 5
# python main.py -m CNN -b 256 --mode test --hidden 256
# python main.py -m RNN -b 64 --mode train --hidden 128 --dropout 0.4 --patience 7 
# python main.py -m RNN -b 64 --mode test --hidden 128
python main.py -m MLP -b 32 --mode train --hidden 512 --dropout 0 --patience 1
python main.py -m MLP -b 32 --mode test --hidden 512