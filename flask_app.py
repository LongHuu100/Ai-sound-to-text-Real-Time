from flask import Flask, request
from flask_socketio import SocketIO
from infer import restore_model, load_audio
import io
import soundfile as sf
import librosa
from gevent.pywsgi import WSGIServer

app = Flask(__name__)
socketio = SocketIO(app)

#đường dẫn tới checkpoint và file config cho model
config = 'config/quartznet12x1_abcfjwz.yaml'
encoder_checkpoint = 'model_vn/JasperEncoder-STEP-1312684.pt'
decoder_checkpoint = 'model_vn/JasperDecoderForCTC-STEP-1312684.pt'

### EN chạy test file infer_en.py
#config_en = 'model_en/quartznet15x5.yaml'
#encoder_checkpoint_en = 'model_en/JasperEncoder-STEP-247400.pt'
#decoder_checkpoint_en = 'model_en/JasperDecoderForCTC-STEP-247400.pt'
#neural_factory_en = restore_model(config_en, encoder_checkpoint_en, decoder_checkpoint_en)
### END EN

neural_factory = restore_model(config, encoder_checkpoint, decoder_checkpoint)
print('========= load model checkpoint done!')

@app.route("/")
def index():
  sig = load_audio('demo_wav/00000.wav')
  print("--- TEST sig", sig)
  greedy_hypotheses, beam_hypotheses = neural_factory.infer_signal(sig)
  print('--- TEST greedy predict:{}'.format(greedy_hypotheses))
  print('--- TEST beamLM predict:{}'.format(beam_hypotheses))
  return beam_hypotheses

# raw sound data from live stream
@app.route('/stt', methods=['POST'])
def s2tapi():

    # samplerate của data stream thông thường là 41100 cần chuyển về 16000 của lúc tranning bằng 
    # librosa qua 2 hàm to_mono và resample
    rawSound, samplerate = sf.read(io.BytesIO(request.data))
    rawSound = rawSound.transpose()
    y = librosa.to_mono(rawSound)
    y = librosa.resample(y, samplerate, 16000)

    _, beam_hypotheses = neural_factory.infer_signal(y)
    return beam_hypotheses

if __name__ == '__main__':
  ####  Dev mode ####
  # socketio.run(app, host="localhost", port=8000, debug=False)
  #### Prod mode python3 flask_app.py & #####
  http_server = WSGIServer(('', 5000), app)
  http_server.serve_forever()
