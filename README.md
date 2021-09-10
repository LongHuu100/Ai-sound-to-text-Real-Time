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

## Xử lý real-time
Phần xử lý chuyển đổi streaming realtime và kiến trúc của Module <br/>

0. Software Structure

- Partition: Code quản lý các thread theo nhiều partition.
- Mỗi partition chứa nhiều thread để xử lý.
- Số lượng thread trong partition có thể mở rộng theo từng thời điểm, vd lúc nhiều luồng sẽ có 100 thread, không có luồng sẽ không có thread chạy.
- Mỗi thread cần config được xử lý bao nhiêu luồng audio, khuyền nghị thực tế là 10 luồng.

1. Tiền xử lý luồng.

- Xử lý đa luồng cần phải chia mỗi luồng audio vào một thread để nhất quán khi chuyển đổi.
- Mỗi thread sẽ kiểm tra đủ điều kiện để gửi nhận diện hay chưa dựa vào số byte gửi lên và thời gian của dữ liệu trong thread.
- Nếu đủ điều kiện thì nhận diện và clear data trong luồng, ngược lại thì tiếp tục chờ đủ điều kiện

## Server 64G RAM, 500GB SSD, CPU 16 CORE
Chạy module với 20 partition, mỗi partition có 30 thread, mỗi thread xử lý 10 kênh audio. \
= 20 * 30 * 10 = 6.000 luồng audio, hết ~8GB RAM \
5 Server python sử dụng GPU, chạy docker, mỗi Server docker scale 10 đầu api = 50 đầu nhận diện.

*** Sau khi xử lý xong các luồng mà không có dữ liệu thì tự động đóng thread và remove partition để không lãng phí RAM do các thread sống.

## Java. Flow.Subscriber và reactive programing
Lập trình reactive được phát triển bởi các kỹ sư trong netflix đầu tiên, sau đó được thêm vào java9 \
Nên sử dụng gói java.util.concurrent.Flow.Subscriber của java9 xử lý pub-sub để không bị block main thread. \
Để tránh việc gọi sang python thì nên build model của keras sau đó dùng dl4j import từ keras vào là chạy được.

## Java 16 đến 18 Loom có thể mở được được lớn hơn 1.000.000 thread để xử lý các job
Để làm được việc này thì java loom đưa ra khái niệm thread ảo dựa trên thuật toán ăn cắp công việc (work-steal) \
Thuật toán này dựa trên mô mình map-reduce để thực hiện hai tác vụ Fock - Join.