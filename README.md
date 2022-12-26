# ExplainableTagging

## CSIE 5460 資訊檢索與擷取
### Final Competition / 自然語言理解的解釋性資訊標記競賽

## Members
- B08902011 杜展廷
- B08902123 李晨安

## OS 
- Operating System: Arch Linux
- Kernel: Linux 5.15.67-1-lts
- Architecture: x86-64
- Hardware Vendor: ASUSTeK COMPUTER INC.
- Hardware Model: ESC8000 G4
- Firmware Version: 3601

## GPU
> We use one of the following GPUs per training.
1. NVIDIA GeForce GTX 1080 Ti
1. NVIDIA GeForce GTX 2080 Ti

## Python Version
- Python 3.9.11

## Requirements (Python package)
- Please refer to requirements.txt

## How to run
1. Install requirements.txt with the following command.
    - pip install -r requirements.txt
1. Change directory to src/
1. Run process_raw.py
    - python process_raw.py
1. Run any of the three methods
    - python run_qd_line.py
    - python run_qd_word.py
    - python run_s2s.py