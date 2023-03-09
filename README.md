# Using (NVIDIA NeMo)
# Cài 2 gói c++ yum install -y boost và yum install -y boost-devel

# Install Package
- cd nemo/script
- ./install_decoders.sh
## Sau khi cài xong bước này sẽ tạo được file /usr/local/lib/python3.7/site-packages/ctc_decoders-1.1-py3.7-linux-x86_64.egg

- pip install python-dateutil
- pip install wrapt
- pip install torch
- pip install torch_stft
- pip install torchaudio
- pip install gevent

- pip install pyOpenSSL
- pip install unidecode
- pip install inflect
- pip install frozendict
- pip install pandas
- pip install kaldi_io
- pip install ruamel_yaml
- pip install torchvision
- pip install wget
- pip install flask_socketio

#### Build language_model #### 
- cd nemo/scripts
- ./build_6-gram_OpenSLR_lm.sh

# Cách xử lý nguyên thủy
B1. Lấy ra phổ tần của các auido (Lọc Fourier trong xử lý tín hiệu số) để lọc nhiễu, chỉ lấy các tần số âm thanh của file <br/>
B2. Traning các phổ này tương ứng với text của file traning <br/>
======= Output ========= <br/>
B1: Đưa audio vào, tách các phổ tần của audio đó <br/>
B2: Nemo dự đoán text dựa trên phổ tần đã được học trước đó.


# TEST training
cd nemo <br/>
pytest tests/unclassified/test_unclassified_asr.py::TestASRPytorch::test_jasper_training --cpu \
Bạn có thể thay cpu = gpu nếu máy tính bạn hỗ trợ gpu

# Tạo các file model có đuôi mở rộng .pt 
Trong thư mục nemo/docs/sources/source/asr/tutorial.rst <br/>
https://ngc.nvidia.com/catalog/models/nvidia:quartznet_15x5_ls_sp <br/>
Các bạn có thể xem trong file tranning.py là ví dụ để tạo model cho việc nhận diện


# Truyền vào byte data từ service để nhận dạng real-time
import soundfile as sf \
rawSound, samplerate = sf.read(io.BytesIO(request.data)) \
rawSound = rawSound.transpose() \
sig_raw = librosa.to_mono(rawSound) \
sig_raw = librosa.resample(y, samplerate, 16000)
